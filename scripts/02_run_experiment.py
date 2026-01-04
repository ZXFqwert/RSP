"""
Step 02: Unified Experiment Runner (Consolidated)

Integrates:
- MITM Retriever (hijacks search results with poisoned documents)
- Agent Execution (ReAct, Reflection, ToT)
- RSV Trajectory Extraction (V, S, A metrics per step)
- Quality Metrics (EM, F1, Semantic)
- Resume Capability (JSONL with qid-based skip)

Usage:
    # Clean baseline
    python scripts/02_run_experiment.py --attack clean --difficulty easy --agents react

    # GSI Paralysis attack
    python scripts/02_run_experiment.py --attack gsi --style paralysis --difficulty easy --agents react reflection

    # Meta-RSP with 50% poison rate
    python scripts/02_run_experiment.py --attack meta_rsp --style haste --poison-rate p50 --agents tot

    # Resume interrupted experiment
    python scripts/02_run_experiment.py --attack gsi --style paralysis --resume
"""

import json
import argparse
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rsp.agents.react import ReActAgent
from rsp.agents.reflection import ReflectionAgent
from rsp.agents.tot import ToTAgent
from rsp.agents.base import AgentResult
from rsp.retrieval.retriever import HybridRetriever
from rsp.evaluation.metrics import compute_all_quality_metrics
from rsp.evaluation.rsv_extractor import EmbeddingRSVExtractor, TraceRSV


# ============================================================
# Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """Configuration for experiment execution"""
    # Attack settings
    attack: str = "clean"           # clean | gsi | meta_rsp
    style: str = "paralysis"        # paralysis | haste
    poison_rate: str = "full"       # p20 | p50 | full (maps to 0.2, 0.5, 1.0)

    # Experiment settings
    difficulty: str = "easy"        # easy | medium | hard
    agents: List[str] = field(default_factory=lambda: ["react"])
    max_steps: int = 10             # Max reasoning steps per question
    max_eval: int = 50              # Max questions to evaluate

    # MITM settings
    hijack_k: int = 2               # Number of top-k results to replace with poisoned
    retrieval_top_k: int = 5        # Total top-k for retrieval

    # Paths (resolved based on base_dir)
    poison_map_path: Optional[Path] = None
    qa_records_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Runtime
    seed: int = 42
    resume: bool = False
    tag: str = ""                   # Custom tag for output filename
    skip_rsv: bool = False          # Skip RSV extraction for faster runs

    def get_poison_rate_float(self) -> float:
        """Convert poison rate string to float"""
        rate_map = {"p20": 0.2, "p50": 0.5, "full": 1.0}
        return rate_map.get(self.poison_rate, 1.0)


# ============================================================
# MITM Retriever (Man-in-the-Middle Attack)
# ============================================================

class MITMRetriever:
    """
    Retriever wrapper that hijacks search results with poisoned documents.

    Attack mechanism:
    1. Perform normal retrieval to get top-k documents
    2. Replace `hijack_k` of the top results with poisoned versions
    3. Return modified result set to the agent

    This simulates a RAG poisoning attack where the attacker has
    compromised the corpus with style-injected documents.
    """

    def __init__(
        self,
        base_retriever,
        poison_map: Dict[str, str],
        hijack_k: int = 2,
        hijack_positions: List[int] = None,
        verbose: bool = False
    ):
        """
        Args:
            base_retriever: Original retriever to wrap
            poison_map: Dict mapping doc_id -> poisoned_text
            hijack_k: Number of results to replace with poisoned versions
            hijack_positions: Specific positions to hijack (default: [0, 2])
            verbose: Print hijack details
        """
        self.base = base_retriever
        self.poison_map = poison_map
        self.hijack_k = hijack_k
        self.hijack_positions = hijack_positions or [0, 2]  # First and third position
        self.verbose = verbose

        # Statistics
        self.total_searches = 0
        self.total_hijacks = 0
        self.hijack_log = []

        # Pre-select which poisoned docs to use for hijacking
        self.poisoned_doc_ids = list(poison_map.keys())
        random.shuffle(self.poisoned_doc_ids)
        self._poison_idx = 0

    def _get_next_poisoned_doc(self) -> Tuple[str, str]:
        """Get next poisoned document in rotation"""
        if not self.poisoned_doc_ids:
            return None, None

        doc_id = self.poisoned_doc_ids[self._poison_idx % len(self.poisoned_doc_ids)]
        self._poison_idx += 1
        return doc_id, self.poison_map[doc_id]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hijacked search.

        Returns results with some documents replaced by poisoned versions.
        """
        self.total_searches += 1

        # Get original results
        results = self.base.search(query, top_k)

        if not self.poison_map:
            return results

        # Determine which positions to hijack
        positions_to_hijack = [p for p in self.hijack_positions if p < len(results)][:self.hijack_k]

        if not positions_to_hijack:
            return results

        # Perform hijacking
        hijacked_results = list(results)
        hijack_record = {"query": query, "hijacks": []}

        for pos in positions_to_hijack:
            doc_id, poisoned_text = self._get_next_poisoned_doc()
            if doc_id is None:
                continue

            original = hijacked_results[pos]

            # Create hijacked document
            hijacked_doc = {
                "doc_id": doc_id,
                "title": original.get("title", f"Hijacked_{doc_id}"),
                "text": poisoned_text,
                "score": original.get("score", 1.0),
                "_hijacked": True,
                "_original_doc_id": original.get("doc_id", ""),
            }

            hijacked_results[pos] = hijacked_doc
            self.total_hijacks += 1

            hijack_record["hijacks"].append({
                "position": pos,
                "original_id": original.get("doc_id", ""),
                "poisoned_id": doc_id,
            })

        if self.verbose and hijack_record["hijacks"]:
            print(f"[MITM] Hijacked {len(hijack_record['hijacks'])} docs for query: {query[:50]}...")

        self.hijack_log.append(hijack_record)
        return hijacked_results

    def get_stats(self) -> Dict[str, Any]:
        """Get MITM statistics"""
        return {
            "total_searches": self.total_searches,
            "total_hijacks": self.total_hijacks,
            "hijack_rate": self.total_hijacks / max(1, self.total_searches * self.hijack_k),
            "poison_pool_size": len(self.poison_map),
        }


# ============================================================
# Agent Factory
# ============================================================

def create_agent(
    agent_type: str,
    llm,
    retriever,
) -> Any:
    """
    Factory function to create agents.

    Args:
        agent_type: "react" | "reflection" | "tot"
        llm: Language model instance
        retriever: Retriever (can be MITMRetriever)

    Returns:
        Agent instance
    """
    agent_map = {
        "react": ReActAgent,
        "reflection": ReflectionAgent,
        "tot": ToTAgent,
    }

    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from: {list(agent_map.keys())}")

    return agent_map[agent_type](llm=llm, retriever=retriever)


# ============================================================
# Experiment Result
# ============================================================

@dataclass
class ExperimentRecord:
    """Single experiment record for one question"""
    qid: str
    question: str
    gold_answer: str
    pred_answer: str

    # Quality metrics
    quality_em: float = 0.0
    quality_f1: float = 0.0
    quality_semantic: float = 0.0

    # Behavioral metrics
    steps: int = 0
    search_calls: int = 0
    total_tokens: int = 0
    elapsed_time: float = 0.0

    # RSV trajectory
    rsv_v_mean: float = 0.0
    rsv_s_mean: float = 0.0
    rsv_a_mean: float = 0.0
    rsv_trajectory: Dict[str, List[float]] = field(default_factory=dict)
    rsv_pattern: str = "unknown"

    # Trace data
    trace: List[Dict[str, Any]] = field(default_factory=list)

    # Status
    status: str = "pending"  # pending | success | error
    error_message: str = ""

    # Metadata
    agent_type: str = ""
    attack_type: str = ""
    difficulty: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        # Simplify trace for storage (keep only essential fields)
        if d.get("trace"):
            d["trace"] = [
                {k: v for k, v in step.items() if k in ["step", "thought", "action", "observation"]}
                for step in d["trace"]
            ]
        return d


# ============================================================
# Experiment Runner
# ============================================================

class ExperimentRunner:
    """
    Main experiment execution engine.

    Handles:
    - Loading poison map and QA records
    - Setting up MITM retriever
    - Running agents on questions
    - Extracting RSV trajectories
    - Computing quality metrics
    - Saving results with resume capability
    """

    def __init__(self, config: ExperimentConfig, llm=None):
        self.config = config
        self.llm = llm

        # Will be initialized in setup()
        self.poison_map: Dict[str, str] = {}
        self.qa_records: List[Dict[str, Any]] = []
        self.base_retriever = None
        self.rsv_extractor = None

        # Results tracking
        self.results: Dict[str, List[ExperimentRecord]] = {}  # agent_type -> records
        self.completed_qids: set = set()

    def setup(self) -> bool:
        """Initialize all components"""
        print("\n" + "=" * 70)
        print("EXPERIMENT SETUP")
        print("=" * 70)

        # 1. Load poison map (if not clean)
        if self.config.attack != "clean":
            if not self._load_poison_map():
                return False
        else:
            print("[Setup] Clean baseline - no poison map needed")

        # 2. Load QA records
        if not self._load_qa_records():
            return False

        # 3. Initialize retriever
        print("[Setup] Initializing base retriever...")
        try:
            self.base_retriever = HybridRetriever()
            print("[Setup] Base retriever ready")
        except Exception as e:
            print(f"[ERROR] Failed to initialize retriever: {e}")
            return False

        # 4. Initialize RSV extractor
        if not self.config.skip_rsv:
            print("[Setup] Initializing RSV extractor...")
            try:
                self.rsv_extractor = EmbeddingRSVExtractor()
                print("[Setup] RSV extractor ready")
            except Exception as e:
                print(f"[WARNING] RSV extractor failed: {e}")
                print("         Continuing without RSV extraction")
                self.rsv_extractor = None

        # 5. Load existing results for resume
        if self.config.resume:
            self._load_existing_results()

        print(f"\n[Setup] Configuration:")
        print(f"  Attack:      {self.config.attack}")
        print(f"  Style:       {self.config.style}")
        print(f"  Poison Rate: {self.config.poison_rate}")
        print(f"  Difficulty:  {self.config.difficulty}")
        print(f"  Agents:      {self.config.agents}")
        print(f"  Max Eval:    {self.config.max_eval}")
        print(f"  Max Steps:   {self.config.max_steps}")
        print(f"  Poison Docs: {len(self.poison_map)}")
        print(f"  QA Records:  {len(self.qa_records)}")
        print(f"  Resume Mode: {self.config.resume}")
        print(f"  Completed:   {len(self.completed_qids)}")

        return True

    def _load_poison_map(self) -> bool:
        """Load poison map from file"""
        poison_path = self.config.poison_map_path

        if not poison_path or not poison_path.exists():
            print(f"[ERROR] Poison map not found: {poison_path}")
            print("        Run 01_generate_shadow_corpus.py first")
            return False

        print(f"[Setup] Loading poison map from {poison_path.name}...")

        with poison_path.open("r", encoding="utf-8") as f:
            self.poison_map = json.load(f)

        print(f"[Setup] Loaded {len(self.poison_map)} poisoned documents")
        return True

    def _load_qa_records(self) -> bool:
        """Load QA records from file"""
        qa_path = self.config.qa_records_path

        if not qa_path or not qa_path.exists():
            print(f"[ERROR] QA records not found: {qa_path}")
            print("        Run 01_generate_shadow_corpus.py first")
            return False

        print(f"[Setup] Loading QA records from {qa_path.name}...")

        self.qa_records = []
        with qa_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.qa_records.append(json.loads(line))

        # Limit to max_eval
        if len(self.qa_records) > self.config.max_eval:
            self.qa_records = self.qa_records[:self.config.max_eval]

        print(f"[Setup] Loaded {len(self.qa_records)} QA records")
        return True

    def _load_existing_results(self):
        """Load existing results for resume capability"""
        for agent_type in self.config.agents:
            output_path = self._get_output_path(agent_type)

            if output_path.exists():
                print(f"[Resume] Loading existing results from {output_path.name}...")

                with output_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                rec = json.loads(line)
                                if rec.get("status") == "success":
                                    self.completed_qids.add(rec.get("qid", ""))
                            except json.JSONDecodeError:
                                continue

                print(f"[Resume] Found {len(self.completed_qids)} completed records")

    def _get_output_path(self, agent_type: str) -> Path:
        """Generate output file path"""
        tag = self.config.tag
        if not tag:
            tag = f"{self.config.difficulty}_{self.config.attack}_{self.config.style}_{self.config.poison_rate}"

        filename = f"experiment_{agent_type}_{tag}.jsonl"
        return self.config.output_dir / filename

    def _create_mitm_retriever(self) -> MITMRetriever:
        """Create MITM retriever with current poison map"""
        if self.config.attack == "clean":
            # For clean baseline, return base retriever wrapped with empty poison map
            return MITMRetriever(
                base_retriever=self.base_retriever,
                poison_map={},
                hijack_k=0,
            )

        return MITMRetriever(
            base_retriever=self.base_retriever,
            poison_map=self.poison_map,
            hijack_k=self.config.hijack_k,
            hijack_positions=[0, 2],  # Hijack 1st and 3rd position
            verbose=False,
        )

    def run_single(
        self,
        agent,
        qa_record: Dict[str, Any],
    ) -> ExperimentRecord:
        """
        Run experiment on a single question.

        Returns:
            ExperimentRecord with all metrics
        """
        qid = qa_record.get("qid", "unknown")
        question = qa_record.get("question", "")
        gold_answer = qa_record.get("answer", "")

        record = ExperimentRecord(
            qid=qid,
            question=question,
            gold_answer=gold_answer,
            pred_answer="",
            agent_type=agent.agent_type,
            attack_type=self.config.attack,
            difficulty=self.config.difficulty,
        )

        try:
            # Run agent
            start_time = time.time()
            result: AgentResult = agent.run(question, max_steps=self.config.max_steps)
            elapsed = time.time() - start_time

            # Extract basic metrics
            record.pred_answer = result.answer
            record.steps = result.steps
            record.search_calls = result.search_calls
            record.total_tokens = result.total_tokens
            record.elapsed_time = elapsed
            record.trace = result.trace

            # Compute quality metrics
            quality = compute_all_quality_metrics(result.answer, gold_answer, self.llm)
            record.quality_em = quality.get("em", 0.0)
            record.quality_f1 = quality.get("f1", 0.0)
            record.quality_semantic = quality.get("semantic", 0.0)

            # Extract RSV trajectory
            if self.rsv_extractor and result.trace:
                try:
                    trace_rsv: TraceRSV = self.rsv_extractor.extract_trace(result.trace, question)
                    record.rsv_v_mean = trace_rsv.v_mean
                    record.rsv_s_mean = trace_rsv.s_mean
                    record.rsv_a_mean = trace_rsv.a_mean
                    record.rsv_pattern = trace_rsv.pattern
                    record.rsv_trajectory = {
                        "v": trace_rsv.v_trajectory,
                        "s": trace_rsv.s_trajectory,
                        "a": trace_rsv.a_trajectory,
                    }
                except Exception as e:
                    print(f"[WARNING] RSV extraction failed for {qid}: {e}")

            record.status = "success"

        except Exception as e:
            record.status = "error"
            record.error_message = str(e)
            print(f"[ERROR] {qid}: {e}")

        return record

    def run_agent(self, agent_type: str) -> List[ExperimentRecord]:
        """
        Run experiment for a specific agent type.

        Returns:
            List of ExperimentRecord
        """
        print(f"\n{'=' * 70}")
        print(f"RUNNING: {agent_type.upper()}")
        print(f"{'=' * 70}")

        # Create MITM retriever
        mitm_retriever = self._create_mitm_retriever()

        # Create agent
        agent = create_agent(agent_type, self.llm, mitm_retriever)

        # Setup output
        output_path = self._get_output_path(agent_type)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = []

        # Open file in append mode for resume
        mode = "a" if self.config.resume else "w"

        with output_path.open(mode, encoding="utf-8") as f:
            for i, qa_record in enumerate(self.qa_records):
                qid = qa_record.get("qid", "unknown")

                # Skip if already completed (resume mode)
                if qid in self.completed_qids:
                    print(f"  [{i+1}/{len(self.qa_records)}] {qid} - SKIPPED (already completed)")
                    continue

                print(f"  [{i+1}/{len(self.qa_records)}] {qid}...", end=" ", flush=True)

                # Run experiment
                record = self.run_single(agent, qa_record)
                records.append(record)

                # Save immediately (real-time JSONL)
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                f.flush()

                # Print status
                if record.status == "success":
                    print(f"✓ EM={record.quality_em:.2f} Steps={record.steps} Time={record.elapsed_time:.1f}s")
                else:
                    print(f"✗ {record.error_message[:50]}")

        # Print MITM statistics
        mitm_stats = mitm_retriever.get_stats()
        print(f"\n[MITM Stats] Searches: {mitm_stats['total_searches']}, "
              f"Hijacks: {mitm_stats['total_hijacks']}, "
              f"Hijack Rate: {mitm_stats['hijack_rate']:.2%}")

        return records

    def run(self) -> Dict[str, Any]:
        """
        Run full experiment across all configured agents.

        Returns:
            Summary dictionary
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT EXECUTION")
        print("=" * 70)

        all_results = {}
        start_time = time.time()

        for agent_type in self.config.agents:
            records = self.run_agent(agent_type)
            all_results[agent_type] = {
                "records": records,
                "stats": self._compute_stats(records),
            }

        elapsed = time.time() - start_time

        # Save summary
        summary = {
            "config": asdict(self.config),
            "results": {
                agent: {
                    "stats": data["stats"],
                    "count": len(data["records"]),
                }
                for agent, data in all_results.items()
            },
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = self.config.output_dir / f"summary_{self.config.tag or 'experiment'}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n[Summary] Saved to {summary_path}")

        return summary

    def _compute_stats(self, records: List[ExperimentRecord]) -> Dict[str, float]:
        """Compute aggregate statistics for a set of records"""
        if not records:
            return {}

        success_records = [r for r in records if r.status == "success"]

        if not success_records:
            return {"success_rate": 0.0}

        return {
            "success_rate": len(success_records) / len(records),
            "em_mean": sum(r.quality_em for r in success_records) / len(success_records),
            "f1_mean": sum(r.quality_f1 for r in success_records) / len(success_records),
            "semantic_mean": sum(r.quality_semantic for r in success_records) / len(success_records),
            "steps_mean": sum(r.steps for r in success_records) / len(success_records),
            "search_calls_mean": sum(r.search_calls for r in success_records) / len(success_records),
            "time_mean": sum(r.elapsed_time for r in success_records) / len(success_records),
            "rsv_v_mean": sum(r.rsv_v_mean for r in success_records) / len(success_records),
            "rsv_s_mean": sum(r.rsv_s_mean for r in success_records) / len(success_records),
            "rsv_a_mean": sum(r.rsv_a_mean for r in success_records) / len(success_records),
        }


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RSP experiments with MITM retrieval hijacking"
    )

    # Attack settings
    parser.add_argument(
        "--attack", type=str, default="clean",
        choices=["clean", "gsi", "meta_rsp"],
        help="Attack type (clean=baseline)"
    )
    parser.add_argument(
        "--style", type=str, default="paralysis",
        choices=["paralysis", "haste"],
        help="Attack style"
    )
    parser.add_argument(
        "--poison-rate", type=str, default="full",
        choices=["p20", "p50", "full"],
        help="Poison rate"
    )

    # Experiment settings
    parser.add_argument(
        "--difficulty", type=str, default="easy",
        choices=["easy", "medium", "hard"],
        help="HotpotQA difficulty level"
    )
    parser.add_argument(
        "--agents", type=str, nargs="+", default=["react"],
        choices=["react", "reflection", "tot"],
        help="Agent types to evaluate"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10,
        help="Maximum reasoning steps per question"
    )
    parser.add_argument(
        "--max-eval", type=int, default=50,
        help="Maximum questions to evaluate"
    )

    # MITM settings
    parser.add_argument(
        "--hijack-k", type=int, default=2,
        help="Number of retrieval results to hijack"
    )

    # Runtime settings
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing results"
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Custom tag for output filename"
    )
    parser.add_argument(
        "--skip-rsv", action="store_true",
        help="Skip RSV extraction for faster runs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    shadow_dir = base_dir / "logs" / "shadow_corpus"
    output_dir = base_dir / "logs" / "traces"

    # Build poison map path based on attack type
    if args.attack == "clean":
        poison_map_path = None
        qa_records_path = data_dir / "hotpot_dev_distractor_v1.json"  # Use raw data for clean
    else:
        # Construct filename based on settings
        tag = f"hotpotqa_{args.attack}_{args.style}_easy"  # Default to easy for poison map
        rate_suffix = {"p20": "_p20", "p50": "_p50", "full": "_p100"}.get(args.poison_rate, "_p100")
        tag += rate_suffix

        poison_map_path = shadow_dir / f"poison_map_{tag}.json"
        qa_records_path = shadow_dir / f"qa_records_{tag}.jsonl"

    # For clean baseline, we need to handle QA records differently
    if args.attack == "clean":
        # Create temporary QA records from HotpotQA
        qa_records_path = shadow_dir / "qa_records_clean.jsonl"
        if not qa_records_path.exists():
            print("[Setup] Creating clean QA records from HotpotQA...")
            hotpot_path = data_dir / "hotpot_dev_distractor_v1.json"
            if hotpot_path.exists():
                qa_records_path.parent.mkdir(parents=True, exist_ok=True)
                with hotpot_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                with qa_records_path.open("w", encoding="utf-8") as f:
                    for ex in data[:args.max_eval]:
                        record = {
                            "qid": ex.get("_id", ""),
                            "question": ex.get("question", ""),
                            "answer": ex.get("answer", ""),
                            "type": ex.get("type", ""),
                            "level": ex.get("level", ""),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[Setup] Created {args.max_eval} clean QA records")

    # Set random seed
    random.seed(args.seed)

    # Create config
    config = ExperimentConfig(
        attack=args.attack,
        style=args.style,
        poison_rate=args.poison_rate,
        difficulty=args.difficulty,
        agents=args.agents,
        max_steps=args.max_steps,
        max_eval=args.max_eval,
        hijack_k=args.hijack_k,
        poison_map_path=poison_map_path,
        qa_records_path=qa_records_path,
        output_dir=output_dir,
        seed=args.seed,
        resume=args.resume,
        tag=args.tag,
        skip_rsv=args.skip_rsv,
    )

    print("\n" + "=" * 70)
    print("RSP EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Attack:      {config.attack}")
    print(f"Style:       {config.style}")
    print(f"Poison Rate: {config.poison_rate}")
    print(f"Difficulty:  {config.difficulty}")
    print(f"Agents:      {config.agents}")
    print("=" * 70)

    # Initialize LLM
    print("\n[Init] Initializing LLM...")
    try:
        from rsp.llm.ollama_client import OllamaClient
        llm = OllamaClient(model="llama3.1:8b")
        print("[Init] LLM ready")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM: {e}")
        sys.exit(1)

    # Run experiment
    runner = ExperimentRunner(config, llm)

    if not runner.setup():
        print("[ERROR] Setup failed. Exiting.")
        sys.exit(1)

    summary = runner.run()

    # Print final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    for agent_type, agent_data in summary.get("results", {}).items():
        stats = agent_data.get("stats", {})
        print(f"\n{agent_type.upper()}:")
        print(f"  Count:      {agent_data.get('count', 0)}")
        print(f"  Success:    {stats.get('success_rate', 0):.1%}")
        print(f"  EM:         {stats.get('em_mean', 0):.3f}")
        print(f"  F1:         {stats.get('f1_mean', 0):.3f}")
        print(f"  Steps:      {stats.get('steps_mean', 0):.1f}")
        print(f"  RSV V:      {stats.get('rsv_v_mean', 0):.3f}")
        print(f"  RSV S:      {stats.get('rsv_s_mean', 0):.3f}")
        print(f"  RSV A:      {stats.get('rsv_a_mean', 0):.3f}")

    print(f"\nTotal time: {summary.get('elapsed_seconds', 0) / 60:.1f} minutes")
    print(f"Results:    {config.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
