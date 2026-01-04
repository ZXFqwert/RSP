"""
Step 03: RSV (Reasoning-State Vector) Calculation (Consolidated)

Integrates:
- Embedding-based RSV calculation (via Ollama bge-m3)
- Heuristic RSV calculation (for fast analysis without embeddings)
- Post-hoc analysis of existing experiment traces
- Batch processing with parallel execution

RSV Metrics:
- V (Verification): Measures cumulative verification effort (search redundancy)
- S (Self-confidence): Measures certainty vs hedging in reasoning text
- A (Attention): Measures focus/dispersion in entity attention

Usage:
    # Post-hoc analysis of experiment traces (heuristic mode - fast)
    python scripts/03_calculate_rsv.py --input logs/traces/experiment_react_*.jsonl --mode heuristic

    # Embedding-based RSV calculation (more accurate, requires Ollama)
    python scripts/03_calculate_rsv.py --input logs/traces/experiment_react_*.jsonl --mode embedding

    # Single file analysis
    python scripts/03_calculate_rsv.py --input logs/traces/experiment_easy_gsi_paralysis_full.jsonl

    # Override existing RSV values
    python scripts/03_calculate_rsv.py --input logs/traces/*.jsonl --overwrite
"""

import json
import argparse
import sys
import math
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ============================================================
# Configuration
# ============================================================

@dataclass
class RSVConfig:
    """Configuration for RSV calculation"""
    mode: str = "heuristic"         # heuristic | embedding
    input_pattern: str = ""         # Glob pattern for input files
    output_dir: Optional[Path] = None
    overwrite: bool = False         # Overwrite existing RSV values
    max_workers: int = 4            # For parallel processing
    embedding_model: str = "bge-m3:latest"  # Ollama model for embeddings
    verbose: bool = False


# ============================================================
# Data Structures
# ============================================================

@dataclass
class StepRSV:
    """RSV metrics for a single reasoning step"""
    step: int
    v: float = 0.0          # Verification (0-1)
    s: float = 0.0          # Self-confidence (0-1)
    a: float = 0.0          # Attention focus (0-1)
    action_type: str = ""   # "search", "finish", "other"
    query: str = ""
    thought_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceRSV:
    """RSV trajectory for an entire reasoning trace"""
    qid: str
    question: str
    n_steps: int

    # Trajectories
    v_trajectory: List[float] = field(default_factory=list)
    s_trajectory: List[float] = field(default_factory=list)
    a_trajectory: List[float] = field(default_factory=list)

    # Aggregate metrics
    v_mean: float = 0.0
    s_mean: float = 0.0
    a_mean: float = 0.0

    # Pattern classification
    pattern: str = "unknown"  # clean | paralysis | haste | unknown

    # Step details
    steps: List[StepRSV] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "qid": self.qid,
            "question": self.question,
            "n_steps": self.n_steps,
            "v_trajectory": self.v_trajectory,
            "s_trajectory": self.s_trajectory,
            "a_trajectory": self.a_trajectory,
            "v_mean": self.v_mean,
            "s_mean": self.s_mean,
            "a_mean": self.a_mean,
            "pattern": self.pattern,
        }
        if self.steps:
            d["steps"] = [s.to_dict() for s in self.steps]
        return d


# ============================================================
# Word Lists for S (Self-confidence) Calculation
# ============================================================

HEDGING_WORDS = [
    # Uncertainty markers
    'may', 'might', 'could', 'possibly', 'perhaps', 'likely', 'unlikely',
    'probably', 'apparently', 'seemingly', 'supposedly', 'presumably',
    # Doubt expressions
    'not sure', 'uncertain', 'unclear', 'unsure', 'doubt', 'doubtful',
    'questionable', 'debatable', 'ambiguous', 'tentative',
    # Verification needs
    'need to verify', 'need to check', 'should confirm', 'requires verification',
    'let me check', 'let me verify', 'need more information',
    # Incompleteness markers
    'fragmentary', 'insufficient', 'incomplete', 'partial', 'limited',
    # Approximation
    'approximately', 'roughly', 'about', 'around', 'estimate',
]

CERTAINTY_WORDS = [
    # Strong certainty
    'definitely', 'certainly', 'obviously', 'clearly', 'undoubtedly',
    'unquestionably', 'absolutely', 'surely', 'indeed', 'truly',
    # Confirmation
    'confirmed', 'verified', 'established', 'proven', 'demonstrated',
    'evidenced', 'substantiated', 'validated',
    # Knowledge claims
    'know', 'known', 'fact', 'factual', 'true', 'correct', 'accurate',
    # Conclusion markers
    'therefore', 'thus', 'hence', 'consequently', 'conclude', 'conclusion',
    'determined', 'found that', 'discovered',
    # Direct assertions
    'is a', 'was a', 'are', 'were', 'will be',
]


# ============================================================
# Heuristic RSV Calculator
# ============================================================

class HeuristicRSVCalculator:
    """
    Fast RSV calculation using heuristic rules.
    No external dependencies (embeddings) required.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def extract_trace(
        self,
        trace: List[Dict[str, Any]],
        question: str,
        qid: str = ""
    ) -> TraceRSV:
        """
        Calculate RSV trajectory for a trace using heuristics.

        Args:
            trace: List of step dictionaries with thought/action/observation
            question: The original question
            qid: Question ID

        Returns:
            TraceRSV with complete trajectory
        """
        if not trace:
            return TraceRSV(qid=qid, question=question, n_steps=0)

        steps_rsv = []
        v_trajectory = []
        s_trajectory = []
        a_trajectory = []

        for i, step in enumerate(trace):
            # Calculate V, S, A for this step
            v, s, a = self._calc_step_metrics(trace[:i+1], step, question)

            # Extract action type
            action = step.get("action", "") or ""
            if "Search" in action or "search" in action.lower():
                action_type = "search"
            elif "Finish" in action or "finish" in action.lower():
                action_type = "finish"
            else:
                action_type = "other"

            # Extract query from search action
            query = ""
            if action_type == "search":
                # Try to extract search query from action
                match = re.search(r'Search\[([^\]]+)\]', action)
                if match:
                    query = match.group(1)
                else:
                    query = action

            # Get thought snippet
            thought = step.get("thought", "") or step.get("reflection", "") or ""
            thought_snippet = thought[:100] + "..." if len(thought) > 100 else thought

            step_rsv = StepRSV(
                step=i,
                v=v,
                s=s,
                a=a,
                action_type=action_type,
                query=query,
                thought_snippet=thought_snippet,
            )

            steps_rsv.append(step_rsv)
            v_trajectory.append(v)
            s_trajectory.append(s)
            a_trajectory.append(a)

        # Calculate means
        v_mean = sum(v_trajectory) / len(v_trajectory) if v_trajectory else 0
        s_mean = sum(s_trajectory) / len(s_trajectory) if s_trajectory else 0
        a_mean = sum(a_trajectory) / len(a_trajectory) if a_trajectory else 0

        # Classify pattern
        pattern = self._classify_pattern(v_mean, s_mean, a_mean, len(trace))

        return TraceRSV(
            qid=qid,
            question=question,
            n_steps=len(trace),
            v_trajectory=v_trajectory,
            s_trajectory=s_trajectory,
            a_trajectory=a_trajectory,
            v_mean=v_mean,
            s_mean=s_mean,
            a_mean=a_mean,
            pattern=pattern,
            steps=steps_rsv,
        )

    def _calc_step_metrics(
        self,
        trace_slice: List[Dict[str, Any]],
        current_step: Dict[str, Any],
        question: str
    ) -> Tuple[float, float, float]:
        """
        Calculate V, S, A for a single step using heuristics.

        Uses prefix model: each step's metrics consider all previous steps.
        """
        # === V: Verification (Cumulative Search Redundancy) ===
        # Count all search actions up to this point
        search_actions = []
        for s in trace_slice:
            action = s.get("action", "") or ""
            if "Search" in action or "search" in action.lower():
                search_actions.append(action)

        n_searches = len(search_actions)

        # Check for query redundancy (repeated searches)
        unique_queries = set()
        for action in search_actions:
            match = re.search(r'Search\[([^\]]+)\]', action)
            if match:
                unique_queries.add(match.group(1).lower().strip())
            else:
                unique_queries.add(action.lower().strip())

        redundancy_factor = 1.0
        if n_searches > 0 and len(unique_queries) < n_searches:
            redundancy_factor = 1.2  # Boost V for redundant searches

        # Use tanh to map to 0-1 range
        # 5 searches = ~0.91, 3 searches = ~0.75, 1 search = ~0.38
        v = math.tanh(n_searches * redundancy_factor / 2.5)

        # === S: Self-confidence (Certainty vs Hedging) ===
        thought = (current_step.get("thought", "") or "") + " " + \
                  (current_step.get("reflection", "") or "")
        thought_lower = thought.lower()

        # Count hedging and certainty words
        hedge_count = sum(thought_lower.count(w) for w in HEDGING_WORDS)
        certainty_count = sum(thought_lower.count(w) for w in CERTAINTY_WORDS)

        # Calculate S: base 0.5, adjusted by word counts
        total_markers = hedge_count + certainty_count
        if total_markers == 0:
            s = 0.5  # Neutral baseline
        else:
            # Range: -1 to 1, then mapped to 0-1
            raw_score = (certainty_count - hedge_count) / (total_markers + 1)
            s = 0.5 + (raw_score * 0.5)
            s = max(0.0, min(1.0, s))  # Clamp to [0, 1]

        # === A: Attention / Progress ===
        action = current_step.get("action", "") or ""

        if "Finish" in action or "finish" in action.lower():
            a = 1.0  # Reached conclusion
        else:
            # Progress estimation based on step index
            step_idx = current_step.get("step", len(trace_slice) - 1)
            # Gradual increase: step 0 = 0.1, step 5 = 0.85
            a = min(0.9, 0.1 + (step_idx * 0.15))

        return v, s, a

    def _classify_pattern(
        self,
        v_mean: float,
        s_mean: float,
        a_mean: float,
        n_steps: int
    ) -> str:
        """
        Classify reasoning pattern based on RSV signature.

        Paralysis: High V, Low S, Low A (many steps, uncertain, doesn't finish)
        Haste: Low V, High S, High A (few steps, overconfident, finishes quickly)
        Clean: Balanced metrics
        """
        # Thresholds (can be tuned based on empirical data)
        V_HIGH = 0.6
        V_LOW = 0.3
        S_HIGH = 0.6
        S_LOW = 0.4
        STEPS_HIGH = 6
        STEPS_LOW = 3

        # Paralysis pattern: High verification, low confidence, many steps
        if v_mean > V_HIGH and s_mean < S_LOW and n_steps > STEPS_HIGH:
            return "paralysis"

        # Haste pattern: Low verification, high confidence, few steps
        if v_mean < V_LOW and s_mean > S_HIGH and n_steps < STEPS_LOW:
            return "haste"

        # Clean pattern: Balanced
        if V_LOW <= v_mean <= V_HIGH and S_LOW <= s_mean <= S_HIGH:
            return "clean"

        return "unknown"


# ============================================================
# Embedding-based RSV Calculator (wrapper for rsv_extractor.py)
# ============================================================

class EmbeddingRSVCalculator:
    """
    RSV calculation using embeddings via Ollama.
    More accurate but requires Ollama server with bge-m3 model.
    """

    def __init__(self, model: str = "bge-m3:latest", verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self._extractor = None
        self._loaded = False

    def _lazy_load(self):
        """Lazy load the embedding extractor"""
        if self._loaded:
            return

        try:
            from rsp.evaluation.rsv_extractor import EmbeddingRSVExtractor
            self._extractor = EmbeddingRSVExtractor(
                ollama_model=self.model,
                verbose=self.verbose
            )
            self._loaded = True
            print(f"[RSV] Loaded embedding extractor with model: {self.model}")
        except Exception as e:
            print(f"[WARNING] Failed to load embedding extractor: {e}")
            print("         Falling back to heuristic mode")
            self._extractor = None
            self._loaded = True

    def extract_trace(
        self,
        trace: List[Dict[str, Any]],
        question: str,
        qid: str = ""
    ) -> TraceRSV:
        """
        Calculate RSV trajectory using embeddings.
        Falls back to heuristic if embedding extractor not available.
        """
        self._lazy_load()

        if self._extractor is None:
            # Fallback to heuristic
            heuristic = HeuristicRSVCalculator(verbose=self.verbose)
            return heuristic.extract_trace(trace, question, qid)

        # Use embedding-based extraction
        try:
            result = self._extractor.extract_trace(trace, question)

            # Convert to our TraceRSV format
            return TraceRSV(
                qid=qid,
                question=question,
                n_steps=len(trace),
                v_trajectory=result.v_trajectory,
                s_trajectory=result.s_trajectory,
                a_trajectory=result.a_trajectory,
                v_mean=result.v_mean,
                s_mean=result.s_mean,
                a_mean=result.a_mean,
                pattern=result.pattern,
                steps=[StepRSV(
                    step=s.step,
                    v=s.v,
                    s=s.s,
                    a=s.a,
                    action_type=s.action_type,
                    query=s.query,
                    thought_snippet=s.thought_snippet,
                ) for s in result.steps]
            )
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Embedding extraction failed: {e}")
            # Fallback to heuristic
            heuristic = HeuristicRSVCalculator(verbose=self.verbose)
            return heuristic.extract_trace(trace, question, qid)


# ============================================================
# Batch Processing
# ============================================================

def process_experiment_file(
    file_path: Path,
    calculator,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Process a single experiment file and add RSV metrics.

    Args:
        file_path: Path to experiment JSONL or JSON file
        calculator: RSVCalculator instance
        overwrite: Whether to overwrite existing RSV values

    Returns:
        Dict with processing results and statistics
    """
    print(f"\n[Processing] {file_path.name}")

    results = {
        "file": str(file_path),
        "records_processed": 0,
        "records_updated": 0,
        "records_skipped": 0,
        "errors": [],
        "trajectories": [],
    }

    # Determine file format
    if file_path.suffix == ".jsonl":
        records = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        results["errors"].append(f"JSON decode error: {e}")
    else:
        # JSON format (experiment summary)
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Navigate to records (handle nested structure)
        if "results" in data:
            if isinstance(data["results"], list):
                records = data["results"]
            elif isinstance(data["results"], dict):
                # Flatten from {agent: {results: [...]}}
                records = []
                for agent_key, agent_data in data["results"].items():
                    if isinstance(agent_data, dict) and "results" in agent_data:
                        records.extend(agent_data["results"])
                    elif isinstance(agent_data, list):
                        records.extend(agent_data)
        else:
            records = [data]

    # Process each record
    updated_records = []

    for record in records:
        results["records_processed"] += 1

        # Check if RSV already exists
        if not overwrite and record.get("rsv_trajectory"):
            results["records_skipped"] += 1
            updated_records.append(record)
            continue

        # Get trace
        trace = record.get("trace", [])
        if not trace:
            results["records_skipped"] += 1
            updated_records.append(record)
            continue

        try:
            # Calculate RSV
            qid = record.get("qid", "unknown")
            question = record.get("question", "")

            trace_rsv = calculator.extract_trace(trace, question, qid)

            # Update record with RSV metrics
            record["rsv_v_mean"] = trace_rsv.v_mean
            record["rsv_s_mean"] = trace_rsv.s_mean
            record["rsv_a_mean"] = trace_rsv.a_mean
            record["rsv_pattern"] = trace_rsv.pattern
            record["rsv_trajectory"] = {
                "v": trace_rsv.v_trajectory,
                "s": trace_rsv.s_trajectory,
                "a": trace_rsv.a_trajectory,
            }

            results["records_updated"] += 1
            results["trajectories"].append(trace_rsv.to_dict())

        except Exception as e:
            results["errors"].append(f"Record {record.get('qid', '?')}: {e}")

        updated_records.append(record)

    # Save updated file
    if results["records_updated"] > 0:
        if file_path.suffix == ".jsonl":
            with file_path.open("w", encoding="utf-8") as f:
                for record in updated_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            # For JSON files, update in place
            if "results" in data:
                if isinstance(data["results"], list):
                    data["results"] = updated_records
                elif isinstance(data["results"], dict):
                    # Need to redistribute records back to agents
                    # For simplicity, just update the first agent
                    for agent_key in data["results"]:
                        if isinstance(data["results"][agent_key], dict):
                            data["results"][agent_key]["results"] = updated_records
                            break

            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  Processed: {results['records_processed']}, "
          f"Updated: {results['records_updated']}, "
          f"Skipped: {results['records_skipped']}")

    return results


def run_batch_analysis(config: RSVConfig) -> Dict[str, Any]:
    """
    Run RSV analysis on multiple files.

    Args:
        config: RSVConfig with input pattern and settings

    Returns:
        Summary of all processing results
    """
    print("\n" + "=" * 70)
    print("RSV CALCULATION")
    print("=" * 70)
    print(f"Mode:        {config.mode}")
    print(f"Input:       {config.input_pattern}")
    print(f"Overwrite:   {config.overwrite}")
    print("=" * 70)

    # Find input files
    input_files = []
    for pattern in config.input_pattern.split(","):
        pattern = pattern.strip()
        input_files.extend([Path(f) for f in glob.glob(pattern)])

    # Remove duplicates and sort
    input_files = sorted(set(input_files))

    if not input_files:
        print(f"[ERROR] No files found matching: {config.input_pattern}")
        return {"error": "No files found"}

    print(f"\nFound {len(input_files)} files to process")

    # Create calculator
    if config.mode == "embedding":
        calculator = EmbeddingRSVCalculator(
            model=config.embedding_model,
            verbose=config.verbose
        )
    else:
        calculator = HeuristicRSVCalculator(verbose=config.verbose)

    # Process files
    all_results = []
    total_processed = 0
    total_updated = 0
    total_skipped = 0
    all_errors = []

    for file_path in input_files:
        try:
            result = process_experiment_file(
                file_path,
                calculator,
                overwrite=config.overwrite
            )
            all_results.append(result)
            total_processed += result["records_processed"]
            total_updated += result["records_updated"]
            total_skipped += result["records_skipped"]
            all_errors.extend(result["errors"])
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
            all_errors.append(f"File {file_path}: {e}")

    # Save summary
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "config": asdict(config),
            "timestamp": datetime.now().isoformat(),
            "files_processed": len(input_files),
            "total_records_processed": total_processed,
            "total_records_updated": total_updated,
            "total_records_skipped": total_skipped,
            "errors": all_errors,
            "file_results": [
                {k: v for k, v in r.items() if k != "trajectories"}
                for r in all_results
            ],
        }

        summary_path = config.output_dir / f"rsv_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n[Summary] Saved to {summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RSV CALCULATION COMPLETE")
    print("=" * 70)
    print(f"Files Processed:   {len(input_files)}")
    print(f"Records Processed: {total_processed}")
    print(f"Records Updated:   {total_updated}")
    print(f"Records Skipped:   {total_skipped}")
    if all_errors:
        print(f"Errors:           {len(all_errors)}")
    print("=" * 70)

    return {
        "files_processed": len(input_files),
        "total_processed": total_processed,
        "total_updated": total_updated,
        "total_skipped": total_skipped,
        "errors": all_errors,
    }


# ============================================================
# Standalone Analysis (for individual traces)
# ============================================================

def analyze_single_trace(
    trace: List[Dict[str, Any]],
    question: str,
    mode: str = "heuristic"
) -> TraceRSV:
    """
    Analyze a single trace and return RSV metrics.

    Useful for programmatic use or debugging.
    """
    if mode == "embedding":
        calculator = EmbeddingRSVCalculator()
    else:
        calculator = HeuristicRSVCalculator()

    return calculator.extract_trace(trace, question)


def print_trajectory_ascii(trace_rsv: TraceRSV):
    """Print ASCII visualization of RSV trajectory"""
    print(f"\nRSV Trajectory for: {trace_rsv.qid}")
    print(f"Question: {trace_rsv.question[:60]}...")
    print(f"Pattern: {trace_rsv.pattern}")
    print(f"Steps: {trace_rsv.n_steps}")
    print()

    # ASCII bar chart
    width = 30

    print("Step  V         S         A")
    print("-" * 50)

    for i, (v, s, a) in enumerate(zip(
        trace_rsv.v_trajectory,
        trace_rsv.s_trajectory,
        trace_rsv.a_trajectory
    )):
        v_bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
        s_bar = "█" * int(s * 10) + "░" * (10 - int(s * 10))
        a_bar = "█" * int(a * 10) + "░" * (10 - int(a * 10))
        print(f"{i:4d}  {v_bar} {s_bar} {a_bar}")

    print("-" * 50)
    print(f"Mean  V={trace_rsv.v_mean:.3f}  S={trace_rsv.s_mean:.3f}  A={trace_rsv.a_mean:.3f}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate RSV (Reasoning-State Vector) metrics from experiment traces"
    )

    parser.add_argument(
        "--input", type=str, required=True,
        help="Input file pattern (glob), e.g., 'logs/traces/*.jsonl'"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for summary (default: same as input)"
    )
    parser.add_argument(
        "--mode", type=str, default="heuristic",
        choices=["heuristic", "embedding"],
        help="RSV calculation mode (default: heuristic)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing RSV values"
    )
    parser.add_argument(
        "--embedding-model", type=str, default="bge-m3:latest",
        help="Ollama model for embedding mode (default: bge-m3:latest)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / "logs" / "rsv"

    config = RSVConfig(
        mode=args.mode,
        input_pattern=args.input,
        output_dir=output_dir,
        overwrite=args.overwrite,
        embedding_model=args.embedding_model,
        verbose=args.verbose,
    )

    # Run analysis
    run_batch_analysis(config)


if __name__ == "__main__":
    main()
