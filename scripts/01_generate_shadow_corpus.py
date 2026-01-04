"""
Step 01: Shadow Corpus Generation (Consolidated)

Integrates:
- HotpotQA loading (for Paralysis style)
- FEVER loading (for Haste style)
- GSI / Meta-RSP poisoning attacks
- DeBERTa NLI fact preservation verification
- Output: poison_map.json (Map<DocID, PoisonedText>)

Usage:
    python scripts/01_generate_shadow_corpus.py --dataset hotpotqa --attack gsi --style paralysis
    python scripts/01_generate_shadow_corpus.py --dataset fever --attack gsi --style haste
    python scripts/01_generate_shadow_corpus.py --dataset hotpotqa --attack meta_rsp --style paralysis --poison-rate 0.5
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rsp.attacks import MetaRSP, GSIAttack
from rsp.attacks.styles import PARALYSIS_STYLE, HASTE_STYLE

# ============================================================
# Configuration
# ============================================================

@dataclass
class GenerationConfig:
    """Configuration for shadow corpus generation"""
    dataset: str = "hotpotqa"      # hotpotqa | fever
    attack: str = "gsi"            # gsi | meta_rsp
    style: str = "paralysis"       # paralysis | haste
    difficulty: str = "easy"       # easy | medium | hard (for hotpotqa)
    poison_rate: float = 1.0       # 0.2 | 0.5 | 1.0
    max_docs: int = 100            # Max documents to poison
    max_workers: int = 3           # Concurrent workers
    nli_threshold: float = 0.7     # DeBERTa NLI entailment threshold
    seed: int = 42

    # Paths (will be resolved based on base_dir)
    hotpotqa_path: Optional[Path] = None
    fever_path: Optional[Path] = None
    output_dir: Optional[Path] = None


# ============================================================
# DeBERTa NLI Checker
# ============================================================

class DeBERTaNLIChecker:
    """
    Fact preservation checker using DeBERTa NLI model.
    Verifies that poisoned text still entails the original facts.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-base-mnli-fever-anli"):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self._loaded = False

    def _lazy_load(self):
        """Lazy load model only when needed"""
        if self._loaded:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            print(f"[NLI] Loading DeBERTa model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()

            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            self._loaded = True
            print(f"[NLI] Model loaded on {self.device}")

        except ImportError:
            print("[WARNING] transformers not installed. NLI check disabled.")
            print("         Run: pip install transformers torch")
            self._loaded = False

    def check_entailment(self, premise: str, hypothesis: str) -> Tuple[float, str]:
        """
        Check if premise entails hypothesis.

        Returns:
            (score, label) where label is 'entailment', 'neutral', or 'contradiction'
        """
        if not self._loaded:
            self._lazy_load()

        if self.model is None:
            return 1.0, "skipped"  # Skip if model failed to load

        import torch

        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # DeBERTa MNLI labels: 0=contradiction, 1=neutral, 2=entailment
        labels = ["contradiction", "neutral", "entailment"]
        pred_idx = probs.argmax().item()

        return probs[2].item(), labels[pred_idx]  # Return entailment score

    def verify_fact_preservation(
        self,
        original_text: str,
        poisoned_text: str,
        key_facts: Optional[List[str]] = None,
        threshold: float = 0.7
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Verify that poisoned text preserves facts from original.

        Args:
            original_text: Original document text
            poisoned_text: Poisoned document text
            key_facts: Optional list of key facts to verify
            threshold: Minimum entailment score to pass

        Returns:
            (passed, avg_score, details)
        """
        if not self._loaded:
            self._lazy_load()

        if self.model is None:
            return True, 1.0, {"status": "skipped", "reason": "model not loaded"}

        # Extract key facts if not provided (use sentences from original)
        if key_facts is None:
            # Simple sentence splitting
            sentences = [s.strip() for s in original_text.replace(".", ".\n").split("\n") if len(s.strip()) > 20]
            key_facts = sentences[:5]  # Check up to 5 key sentences

        if not key_facts:
            return True, 1.0, {"status": "skipped", "reason": "no facts to verify"}

        scores = []
        details = {"facts_checked": [], "failed_facts": []}

        for fact in key_facts:
            score, label = self.check_entailment(poisoned_text, fact)
            scores.append(score)

            fact_result = {
                "fact": fact[:100] + "..." if len(fact) > 100 else fact,
                "score": score,
                "label": label
            }
            details["facts_checked"].append(fact_result)

            if score < threshold:
                details["failed_facts"].append(fact_result)

        avg_score = sum(scores) / len(scores) if scores else 1.0
        passed = avg_score >= threshold and len(details["failed_facts"]) == 0

        details["avg_score"] = avg_score
        details["passed"] = passed

        return passed, avg_score, details


# ============================================================
# Data Loaders
# ============================================================

def load_hotpotqa(path: Path, difficulty: str = "easy", max_examples: int = 100) -> Dict[str, Any]:
    """
    Load HotpotQA dataset with difficulty filtering.

    Difficulty levels based on supporting facts count:
    - easy: <= 2 hops
    - medium: 4-6 hops
    - hard: >= 6 hops

    Returns:
        {
            "qa_records": [...],
            "corpus_records": [...],  # Documents to poison
            "metadata": {...}
        }
    """
    print(f"[Data] Loading HotpotQA from {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter by difficulty
    difficulty_ranges = {
        "easy": (0, 3),      # <= 2 hops
        "medium": (4, 7),    # 4-6 hops
        "hard": (6, 999),    # >= 6 hops
    }
    min_hops, max_hops = difficulty_ranges.get(difficulty, (0, 999))

    filtered = []
    for ex in data:
        n_hops = len(ex.get("supporting_facts", []))
        if min_hops <= n_hops <= max_hops:
            filtered.append(ex)

    # Sample if needed
    if len(filtered) > max_examples:
        random.shuffle(filtered)
        filtered = filtered[:max_examples]

    # Extract QA records and corpus
    qa_records = []
    corpus_records = []
    seen_docs = set()

    for ex in filtered:
        qa_records.append({
            "qid": ex.get("_id", ""),
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "type": ex.get("type", ""),
            "level": ex.get("level", ""),
            "supporting_facts": ex.get("supporting_facts", []),
        })

        # Extract documents from context
        for title, sentences in ex.get("context", []):
            doc_id = f"hotpot_{title.replace(' ', '_')}"
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                corpus_records.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": " ".join(sentences),
                    "source": "hotpotqa",
                })

    print(f"[Data] Loaded {len(qa_records)} QA pairs, {len(corpus_records)} documents")

    return {
        "qa_records": qa_records,
        "corpus_records": corpus_records,
        "metadata": {
            "source": "hotpotqa",
            "difficulty": difficulty,
            "total_examples": len(filtered),
        }
    }


def load_fever(path: Path, max_examples: int = 100) -> Dict[str, Any]:
    """
    Load FEVER dataset for claim verification.

    FEVER format: {"id", "verifiable", "label", "claim", "evidence"}

    Returns:
        {
            "qa_records": [...],  # Claims as questions
            "corpus_records": [...],  # Evidence documents to poison
            "metadata": {...}
        }
    """
    print(f"[Data] Loading FEVER from {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter for verifiable claims with evidence
    verifiable = [r for r in records if r.get("verifiable") == "VERIFIABLE"]

    if len(verifiable) > max_examples:
        random.shuffle(verifiable)
        verifiable = verifiable[:max_examples]

    qa_records = []
    corpus_records = []
    seen_docs = set()

    for rec in verifiable:
        # Convert claim to QA format
        qa_records.append({
            "qid": f"fever_{rec.get('id', '')}",
            "question": f"Verify: {rec.get('claim', '')}",
            "answer": rec.get("label", ""),
            "type": "claim_verification",
            "evidence": rec.get("evidence", []),
        })

        # Create corpus records from evidence
        # Note: FEVER evidence format is complex, simplify for corpus
        for i, evidence_set in enumerate(rec.get("evidence", [])):
            if isinstance(evidence_set, list):
                for j, evidence in enumerate(evidence_set):
                    if isinstance(evidence, list) and len(evidence) >= 3:
                        wiki_url = evidence[2] if len(evidence) > 2 else f"evidence_{i}_{j}"
                        doc_id = f"fever_{rec.get('id')}_{wiki_url}"
                        if doc_id not in seen_docs:
                            seen_docs.add(doc_id)
                            # Use claim as proxy for document content
                            corpus_records.append({
                                "doc_id": doc_id,
                                "title": str(wiki_url),
                                "text": rec.get("claim", ""),
                                "source": "fever",
                                "label": rec.get("label", ""),
                            })

    print(f"[Data] Loaded {len(qa_records)} claims, {len(corpus_records)} evidence docs")

    return {
        "qa_records": qa_records,
        "corpus_records": corpus_records,
        "metadata": {
            "source": "fever",
            "total_examples": len(verifiable),
        }
    }


# ============================================================
# Poisoning Pipeline
# ============================================================

def poison_document_task(
    args: Tuple[Dict[str, Any], str, str, DeBERTaNLIChecker, float]
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Worker function for concurrent document poisoning.

    Args:
        args: (doc_record, attack_type, style, nli_checker, nli_threshold)

    Returns:
        (doc_id, poisoned_text or None, metadata)
    """
    doc, attack_type, style, nli_checker, nli_threshold = args
    doc_id = doc["doc_id"]
    title = doc["title"]
    text = doc["text"]

    metadata = {
        "doc_id": doc_id,
        "attack": attack_type,
        "style": style,
        "original_length": len(text),
    }

    try:
        # Create attacker
        if attack_type == "meta_rsp":
            attacker = MetaRSP(style=style)
            poisoned_text = attacker.poison_document(title, text)
            metadata["method"] = "meta_prompt_prepend"
        else:  # gsi
            attacker = GSIAttack(style=style)
            poisoned_text, attack_meta = attacker.rewrite_document(title, text)
            metadata.update(attack_meta)

        metadata["poisoned_length"] = len(poisoned_text)

        # NLI fact preservation check
        if nli_checker is not None:
            passed, score, nli_details = nli_checker.verify_fact_preservation(
                text, poisoned_text, threshold=nli_threshold
            )
            metadata["nli_passed"] = passed
            metadata["nli_score"] = score
            metadata["nli_details"] = nli_details

            if not passed:
                print(f"[NLI FAIL] {doc_id}: score={score:.3f} < {nli_threshold}")
                # Still return the poisoned text, but flag it
                metadata["nli_warning"] = True

        metadata["status"] = "success"
        return doc_id, poisoned_text, metadata

    except Exception as e:
        metadata["status"] = "error"
        metadata["error"] = str(e)
        print(f"[ERROR] {doc_id}: {e}")
        return doc_id, None, metadata


def generate_shadow_corpus(config: GenerationConfig) -> Dict[str, Any]:
    """
    Main pipeline: Load data -> Poison documents -> Verify facts -> Save poison_map.json

    Returns:
        {
            "poison_map": {doc_id: poisoned_text, ...},
            "metadata": {...},
            "qa_records": [...],
        }
    """
    print("\n" + "=" * 70)
    print("SHADOW CORPUS GENERATION")
    print("=" * 70)
    print(f"Dataset:     {config.dataset}")
    print(f"Attack:      {config.attack}")
    print(f"Style:       {config.style}")
    print(f"Poison Rate: {config.poison_rate}")
    print(f"Max Docs:    {config.max_docs}")
    print(f"Workers:     {config.max_workers}")
    print("=" * 70 + "\n")

    random.seed(config.seed)

    # 1. Load data based on dataset type
    if config.dataset == "hotpotqa":
        data = load_hotpotqa(
            config.hotpotqa_path,
            difficulty=config.difficulty,
            max_examples=config.max_docs * 2  # Load extra for corpus
        )
    elif config.dataset == "fever":
        data = load_fever(
            config.fever_path,
            max_examples=config.max_docs * 2
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    corpus_records = data["corpus_records"]
    qa_records = data["qa_records"]

    # 2. Select documents to poison based on poison_rate
    n_to_poison = min(int(len(corpus_records) * config.poison_rate), config.max_docs)
    docs_to_poison = corpus_records[:n_to_poison]

    print(f"\n[Poison] Will poison {n_to_poison} / {len(corpus_records)} documents")

    # 3. Initialize NLI checker
    nli_checker = DeBERTaNLIChecker()

    # 4. Poison documents concurrently
    poison_map = {}
    all_metadata = []

    tasks = [
        (doc, config.attack, config.style, nli_checker, config.nli_threshold)
        for doc in docs_to_poison
    ]

    start_time = time.time()
    completed = 0
    failed = 0
    nli_warnings = 0

    print(f"\n[Poison] Starting concurrent poisoning with {config.max_workers} workers...")

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(poison_document_task, task): task[0]["doc_id"]
            for task in tasks
        }

        for future in as_completed(futures):
            doc_id = futures[future]
            try:
                result_doc_id, poisoned_text, metadata = future.result()
                all_metadata.append(metadata)

                if poisoned_text is not None:
                    poison_map[result_doc_id] = poisoned_text
                    completed += 1
                    if metadata.get("nli_warning"):
                        nli_warnings += 1
                else:
                    failed += 1

                # Progress update
                total = completed + failed
                if total % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Progress: {total}/{len(tasks)} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"[ERROR] Future exception for {doc_id}: {e}")
                failed += 1

    elapsed = time.time() - start_time

    print(f"\n[Poison] Completed in {elapsed:.1f}s")
    print(f"  Success:      {completed}")
    print(f"  Failed:       {failed}")
    print(f"  NLI Warnings: {nli_warnings}")

    # 5. Prepare output
    result = {
        "poison_map": poison_map,
        "qa_records": qa_records,
        "metadata": {
            "config": asdict(config),
            "source_metadata": data["metadata"],
            "stats": {
                "total_corpus": len(corpus_records),
                "poisoned": completed,
                "failed": failed,
                "nli_warnings": nli_warnings,
                "elapsed_seconds": elapsed,
            },
            "document_metadata": all_metadata,
        }
    }

    return result


def save_outputs(result: Dict[str, Any], config: GenerationConfig):
    """Save poison_map.json and related files"""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tag for filenames
    tag = f"{config.dataset}_{config.attack}_{config.style}"
    if config.dataset == "hotpotqa":
        tag += f"_{config.difficulty}"
    tag += f"_p{int(config.poison_rate * 100)}"

    # 1. Save poison_map.json (the main output per requirements)
    poison_map_path = output_dir / f"poison_map_{tag}.json"
    with poison_map_path.open("w", encoding="utf-8") as f:
        json.dump(result["poison_map"], f, ensure_ascii=False, indent=2)
    print(f"\n[Save] poison_map.json -> {poison_map_path}")

    # 2. Save full result with metadata
    full_result_path = output_dir / f"shadow_corpus_{tag}.json"
    with full_result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Save] Full result -> {full_result_path}")

    # 3. Save QA records for experiments
    qa_path = output_dir / f"qa_records_{tag}.jsonl"
    with qa_path.open("w", encoding="utf-8") as f:
        for rec in result["qa_records"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[Save] QA records -> {qa_path}")

    print(f"\n[Done] Outputs saved to {output_dir}/")

    return {
        "poison_map": poison_map_path,
        "full_result": full_result_path,
        "qa_records": qa_path,
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate shadow corpus with reasoning-style poisoning"
    )

    parser.add_argument(
        "--dataset", type=str, default="hotpotqa",
        choices=["hotpotqa", "fever"],
        help="Dataset to use (hotpotqa for paralysis, fever for haste)"
    )
    parser.add_argument(
        "--attack", type=str, default="gsi",
        choices=["gsi", "meta_rsp"],
        help="Attack type"
    )
    parser.add_argument(
        "--style", type=str, default="paralysis",
        choices=["paralysis", "haste"],
        help="Attack style"
    )
    parser.add_argument(
        "--difficulty", type=str, default="easy",
        choices=["easy", "medium", "hard"],
        help="Difficulty level for HotpotQA"
    )
    parser.add_argument(
        "--poison-rate", type=float, default=1.0,
        help="Poison rate (0.2, 0.5, or 1.0)"
    )
    parser.add_argument(
        "--max-docs", type=int, default=100,
        help="Maximum documents to poison"
    )
    parser.add_argument(
        "--max-workers", type=int, default=3,
        help="Concurrent workers"
    )
    parser.add_argument(
        "--nli-threshold", type=float, default=0.7,
        help="DeBERTa NLI entailment threshold"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--skip-nli", action="store_true",
        help="Skip NLI verification (faster but no fact check)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    config = GenerationConfig(
        dataset=args.dataset,
        attack=args.attack,
        style=args.style,
        difficulty=args.difficulty,
        poison_rate=args.poison_rate,
        max_docs=args.max_docs,
        max_workers=args.max_workers,
        nli_threshold=args.nli_threshold if not args.skip_nli else 0.0,
        seed=args.seed,
        hotpotqa_path=data_dir / "hotpot_dev_distractor_v1.json",
        fever_path=data_dir / "shared_task_dev.jsonl",
        output_dir=base_dir / "logs" / "shadow_corpus",
    )

    # Validate paths
    if config.dataset == "hotpotqa" and not config.hotpotqa_path.exists():
        print(f"[ERROR] HotpotQA file not found: {config.hotpotqa_path}")
        print("        Download from: https://hotpotqa.github.io/")
        sys.exit(1)

    if config.dataset == "fever" and not config.fever_path.exists():
        print(f"[ERROR] FEVER file not found: {config.fever_path}")
        print("        Download from: https://fever.ai/resources.html")
        sys.exit(1)

    # Run pipeline
    result = generate_shadow_corpus(config)

    # Save outputs
    output_paths = save_outputs(result, config)

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Poison Map:  {len(result['poison_map'])} documents")
    print(f"QA Records:  {len(result['qa_records'])} questions")
    print(f"Output:      {output_paths['poison_map']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
