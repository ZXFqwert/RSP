"""
Step 04: Analysis, Table Generation & Visualization (Consolidated)

Integrates:
- Experiment result analysis from logs/traces/*.jsonl
- RSV trajectory visualization (from 08_visualize_rsv_comparison.py)
- Defense evaluation visualization (from 09_visualize_defense.py)
- LaTeX table generation for paper (Tables 1-4)
- Statistical comparison and summary reports

Usage:
    # Full analysis with all visualizations
    python scripts/04_analyze_and_tabulate.py --input logs/traces/*.jsonl

    # Generate only LaTeX tables
    python scripts/04_analyze_and_tabulate.py --input logs/traces/*.jsonl --tables-only

    # Generate only figures
    python scripts/04_analyze_and_tabulate.py --input logs/traces/*.jsonl --figures-only

    # Specify output directory
    python scripts/04_analyze_and_tabulate.py --input logs/traces/*.jsonl --output-dir results/
"""

import json
import argparse
import sys
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not installed. Visualization disabled.")
    print("         Run: pip install matplotlib numpy")

try:
    import seaborn as sns
    import pandas as pd
    HAS_SEABORN = True
    if HAS_MATPLOTLIB:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        plt.rcParams['font.family'] = 'DejaVu Sans'
except ImportError:
    HAS_SEABORN = False
    print("[WARNING] seaborn/pandas not installed. Advanced plots disabled.")
    print("         Run: pip install seaborn pandas")


# ============================================================
# Configuration
# ============================================================

@dataclass
class AnalysisConfig:
    """Configuration for analysis and visualization"""
    input_pattern: str = ""          # Glob pattern for input files
    output_dir: Optional[Path] = None
    tables_only: bool = False        # Only generate LaTeX tables
    figures_only: bool = False       # Only generate figures
    max_traces: int = 10             # Max traces for visualization
    difficulty: str = "easy"         # Filter by difficulty
    verbose: bool = False


# Color Palette for visualizations
COLORS = {
    "meta_rsp": "#9b59b6",  # Purple
    "gsi": "#e74c3c",       # Red
    "clean": "#95a5a6",     # Gray
    "none": "#bdc3c7",
    "delimiter": "#3498db", # Blue
    "paraphrase": "#2ecc71", # Green
    "paralysis": "#e74c3c",
    "haste": "#f39c12",
}


# ============================================================
# Data Loading
# ============================================================

def load_experiment_results(input_pattern: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load experiment results from JSONL files.

    Returns:
        Dict mapping experiment_tag -> list of records
    """
    results = defaultdict(list)

    for pattern in input_pattern.split(","):
        pattern = pattern.strip()
        for filepath in glob.glob(pattern):
            path = Path(filepath)

            if path.suffix == ".jsonl":
                tag = path.stem  # e.g., "experiment_react_easy_gsi_paralysis_full"
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                results[tag].append(record)
                            except json.JSONDecodeError:
                                continue

            elif path.suffix == ".json":
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle nested structure
                if "results" in data:
                    for agent_key, agent_data in data.get("results", {}).items():
                        if isinstance(agent_data, dict) and "results" in agent_data:
                            tag = f"{path.stem}_{agent_key}"
                            results[tag].extend(agent_data["results"])
                        elif isinstance(agent_data, list):
                            tag = f"{path.stem}_{agent_key}"
                            results[tag].extend(agent_data)

    return dict(results)


def load_rsv_trajectories(
    experiment_path: Path,
    max_traces: int = 10
) -> List[Tuple[str, List[float], List[float], List[float]]]:
    """
    Extract pre-computed rsv_trajectory from experiment files.

    Returns:
        List of (qid, v_trajectory, s_trajectory, a_trajectory)
    """
    if not experiment_path.exists():
        return []

    trajectories = []

    if experiment_path.suffix == ".jsonl":
        with experiment_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and len(trajectories) < max_traces:
                    try:
                        record = json.loads(line)
                        rsv_traj = record.get("rsv_trajectory", {})
                        v_traj = rsv_traj.get("v", [])
                        s_traj = rsv_traj.get("s", [])
                        a_traj = rsv_traj.get("a", [])

                        if v_traj and s_traj and a_traj:
                            trajectories.append((
                                record.get("qid", "unknown"),
                                v_traj, s_traj, a_traj
                            ))
                    except json.JSONDecodeError:
                        continue
    else:
        with experiment_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for agent_name, agent_results in data.get("results", {}).items():
            if not isinstance(agent_results, dict):
                continue

            for record in agent_results.get("results", [])[:max_traces]:
                rsv_traj = record.get("rsv_trajectory", {})
                v_traj = rsv_traj.get("v", [])
                s_traj = rsv_traj.get("s", [])
                a_traj = rsv_traj.get("a", [])

                if v_traj and s_traj and a_traj:
                    trajectories.append((
                        record.get("qid", "unknown"),
                        v_traj, s_traj, a_traj
                    ))
            break

    return trajectories


def load_defense_results(log_dir: Path) -> Tuple[Any, Any]:
    """
    Load defense evaluation results.

    Returns:
        (detection_df, mitigation_df) - pandas DataFrames or None
    """
    if not HAS_SEABORN:
        return None, None

    detection_records = []
    mitigation_records = []

    # Load detection results (04A_*.json)
    for f in log_dir.glob("04A_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            results = data.get("results", {})
            config = data.get("config", {})
            difficulty = config.get("difficulty", "easy")
            style = config.get("style", "paralysis")

            for attack_type, attack_data in results.items():
                if isinstance(attack_data, dict):
                    for defense_name, metrics in attack_data.items():
                        if isinstance(metrics, dict) and "tpr" in metrics:
                            defense_key = defense_name.lower()
                            if "piguard" in defense_key:
                                defense_key = "piguard"
                            elif "hpid" in defense_key:
                                defense_key = "hpid"

                            detection_records.append({
                                "difficulty": difficulty,
                                "style": style,
                                "defense": defense_key,
                                "attack": attack_type,
                                "tpr": metrics.get("tpr", 0) * 100,
                                "fpr": metrics.get("fpr", 0) * 100,
                                "precision": metrics.get("precision", 0) * 100,
                                "f1": metrics.get("f1", 0) * 100,
                            })
        except Exception as e:
            print(f"[WARNING] Error loading {f}: {e}")

    # Load mitigation results (04B_*.jsonl)
    for f in log_dir.glob("04B_*.jsonl"):
        parts = f.stem.split("_")
        if len(parts) >= 4:
            difficulty = parts[1]
            if parts[2] == "meta":
                corpus_type = "meta_rsp"
                defense_type = parts[4] if len(parts) > 4 else "none"
            else:
                corpus_type = parts[2]
                defense_type = parts[3] if len(parts) > 3 else "none"

            metrics = {"em": [], "f1": [], "semantic": [], "steps": [], "tokens": [], "time": []}

            try:
                with open(f, "r", encoding="utf-8") as fp:
                    for line in fp:
                        if line.strip():
                            rec = json.loads(line)
                            if rec.get("status") == "success":
                                metrics["em"].append(rec.get("quality_em", 0))
                                metrics["f1"].append(rec.get("quality_f1", 0))
                                metrics["semantic"].append(rec.get("quality_semantic", 0))
                                metrics["steps"].append(rec.get("steps", 0))
                                metrics["tokens"].append(rec.get("total_tokens", 0))
                                metrics["time"].append(rec.get("elapsed_time", 0))

                if metrics["em"]:
                    mitigation_records.append({
                        "difficulty": difficulty,
                        "corpus": corpus_type,
                        "defense": defense_type,
                        "em": np.mean(metrics["em"]),
                        "f1": np.mean(metrics["f1"]),
                        "semantic": np.mean(metrics["semantic"]),
                        "steps": np.mean(metrics["steps"]),
                        "tokens": np.mean(metrics["tokens"]),
                        "time": np.mean(metrics["time"]),
                        "count": len(metrics["em"]),
                    })
            except Exception as e:
                print(f"[WARNING] Error loading {f}: {e}")

    det_df = pd.DataFrame(detection_records) if detection_records else pd.DataFrame()
    mit_df = pd.DataFrame(mitigation_records) if mitigation_records else pd.DataFrame()

    return det_df, mit_df


# ============================================================
# Statistical Analysis
# ============================================================

def compute_experiment_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics for experiment records."""
    if not records:
        return {}

    success_records = [r for r in records if r.get("status") == "success"]

    if not success_records:
        return {"success_rate": 0.0, "count": len(records)}

    def safe_mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "count": len(records),
        "success_count": len(success_records),
        "success_rate": len(success_records) / len(records),

        # Quality metrics
        "em_mean": safe_mean([r.get("quality_em", 0) for r in success_records]),
        "f1_mean": safe_mean([r.get("quality_f1", 0) for r in success_records]),
        "semantic_mean": safe_mean([r.get("quality_semantic", 0) for r in success_records]),

        # Behavioral metrics
        "steps_mean": safe_mean([r.get("steps", 0) for r in success_records]),
        "search_calls_mean": safe_mean([r.get("search_calls", 0) for r in success_records]),
        "tokens_mean": safe_mean([r.get("total_tokens", 0) for r in success_records]),
        "time_mean": safe_mean([r.get("elapsed_time", 0) for r in success_records]),

        # RSV metrics
        "rsv_v_mean": safe_mean([r.get("rsv_v_mean", 0) for r in success_records]),
        "rsv_s_mean": safe_mean([r.get("rsv_s_mean", 0) for r in success_records]),
        "rsv_a_mean": safe_mean([r.get("rsv_a_mean", 0) for r in success_records]),

        # Pattern distribution
        "patterns": defaultdict(int, {
            r.get("rsv_pattern", "unknown"): 1 for r in success_records
        }),
    }


def compare_experiments(
    clean_stats: Dict[str, Any],
    attack_stats: Dict[str, Any],
    attack_name: str = "attack"
) -> Dict[str, Any]:
    """Compare clean baseline vs attack experiment."""
    comparison = {
        "attack": attack_name,
        "metrics": {},
    }

    for metric in ["em_mean", "f1_mean", "steps_mean", "rsv_v_mean", "rsv_s_mean", "rsv_a_mean"]:
        clean_val = clean_stats.get(metric, 0)
        attack_val = attack_stats.get(metric, 0)
        diff = attack_val - clean_val
        pct_change = (diff / clean_val * 100) if clean_val != 0 else 0

        comparison["metrics"][metric] = {
            "clean": clean_val,
            "attack": attack_val,
            "diff": diff,
            "pct_change": pct_change,
        }

    return comparison


# ============================================================
# LaTeX Table Generation
# ============================================================

def generate_table1_main_results(
    all_stats: Dict[str, Dict[str, Any]],
    output_path: Path
):
    """
    Generate Table 1: Main Attack Results

    Format:
    | Attack | Style | EM | F1 | Steps | V_mean | S_mean |
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Main Attack Results: GSI vs Meta-RSP on HotpotQA}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Attack & Style & EM$\downarrow$ & F1$\downarrow$ & Steps$\uparrow$ & V$\uparrow$ & S$\downarrow$ \\",
        r"\midrule",
    ]

    # Add rows for each experiment
    for tag, stats in sorted(all_stats.items()):
        # Parse tag to extract attack/style info
        parts = tag.lower().split("_")

        attack = "Clean"
        style = "-"
        if "gsi" in parts:
            attack = "GSI"
        elif "meta_rsp" in parts or "metarsp" in parts:
            attack = "Meta-RSP"

        if "paralysis" in parts:
            style = "Paralysis"
        elif "haste" in parts:
            style = "Haste"

        em = stats.get("em_mean", 0) * 100
        f1 = stats.get("f1_mean", 0) * 100
        steps = stats.get("steps_mean", 0)
        v_mean = stats.get("rsv_v_mean", 0)
        s_mean = stats.get("rsv_s_mean", 0)

        lines.append(
            f"{attack} & {style} & {em:.1f} & {f1:.1f} & {steps:.1f} & {v_mean:.3f} & {s_mean:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Generated Table 1: {output_path}")


def generate_table2_agent_comparison(
    all_stats: Dict[str, Dict[str, Any]],
    output_path: Path
):
    """
    Generate Table 2: Agent Comparison (ReAct vs Reflection vs ToT)
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Agent Vulnerability Comparison under GSI-Paralysis Attack}",
        r"\label{tab:agent-comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Agent & EM$\downarrow$ & Steps$\uparrow$ & V$\uparrow$ & S$\downarrow$ \\",
        r"\midrule",
    ]

    # Group by agent type
    agent_stats = {}
    for tag, stats in all_stats.items():
        for agent in ["react", "reflection", "tot"]:
            if agent in tag.lower():
                if agent not in agent_stats:
                    agent_stats[agent] = []
                agent_stats[agent].append(stats)

    for agent in ["react", "reflection", "tot"]:
        if agent in agent_stats:
            # Average across all experiments for this agent
            stats_list = agent_stats[agent]
            em = np.mean([s.get("em_mean", 0) for s in stats_list]) * 100
            steps = np.mean([s.get("steps_mean", 0) for s in stats_list])
            v_mean = np.mean([s.get("rsv_v_mean", 0) for s in stats_list])
            s_mean = np.mean([s.get("rsv_s_mean", 0) for s in stats_list])

            agent_display = {"react": "ReAct", "reflection": "Reflection", "tot": "ToT"}[agent]
            lines.append(
                f"{agent_display} & {em:.1f} & {steps:.1f} & {v_mean:.3f} & {s_mean:.3f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Generated Table 2: {output_path}")


def generate_table3_poison_rate(
    all_stats: Dict[str, Dict[str, Any]],
    output_path: Path
):
    """
    Generate Table 3: Poison Rate Ablation (p20, p50, full)
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of Poison Rate on Attack Efficacy}",
        r"\label{tab:poison-rate}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Poison Rate & EM$\downarrow$ & F1$\downarrow$ & Steps$\uparrow$ & V$\uparrow$ & S$\downarrow$ \\",
        r"\midrule",
    ]

    # Group by poison rate
    rate_stats = {"p20": [], "p50": [], "full": [], "p100": []}
    for tag, stats in all_stats.items():
        for rate in rate_stats.keys():
            if rate in tag.lower():
                rate_stats[rate].append(stats)

    for rate, display in [("p20", "20\\%"), ("p50", "50\\%"), ("full", "100\\%"), ("p100", "100\\%")]:
        if rate_stats.get(rate):
            stats_list = rate_stats[rate]
            em = np.mean([s.get("em_mean", 0) for s in stats_list]) * 100
            f1 = np.mean([s.get("f1_mean", 0) for s in stats_list]) * 100
            steps = np.mean([s.get("steps_mean", 0) for s in stats_list])
            v_mean = np.mean([s.get("rsv_v_mean", 0) for s in stats_list])
            s_mean = np.mean([s.get("rsv_s_mean", 0) for s in stats_list])

            lines.append(
                f"{display} & {em:.1f} & {f1:.1f} & {steps:.1f} & {v_mean:.3f} & {s_mean:.3f} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Generated Table 3: {output_path}")


def generate_table4_defense_results(
    det_df: Any,
    mit_df: Any,
    output_path: Path
):
    """
    Generate Table 4: Defense Evaluation Results
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Defense Evaluation: Detection and Mitigation}",
        r"\label{tab:defense}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Defense Type & Method & Meta-RSP & GSI \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Detection (TPR\%)}} \\",
    ]

    if det_df is not None and not det_df.empty:
        for defense in ["hpid", "piguard"]:
            meta_row = det_df[(det_df["defense"] == defense) & (det_df["attack"] == "meta_rsp")]
            gsi_row = det_df[(det_df["defense"] == defense) & (det_df["attack"] == "gsi")]

            meta_tpr = meta_row["tpr"].values[0] if len(meta_row) > 0 else 0
            gsi_tpr = gsi_row["tpr"].values[0] if len(gsi_row) > 0 else 0

            defense_display = {"hpid": "HPID", "piguard": "PIGuard"}[defense]
            lines.append(f"Detection & {defense_display} & {meta_tpr:.0f}\\% & {gsi_tpr:.0f}\\% \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Mitigation (Steps Reduction)}} \\")

    if mit_df is not None and not mit_df.empty:
        for defense in ["delimiter", "paraphrase"]:
            meta_row = mit_df[(mit_df["defense"] == defense) & (mit_df["corpus"] == "meta_rsp")]
            gsi_row = mit_df[(mit_df["defense"] == defense) & (mit_df["corpus"] == "gsi")]
            clean_row = mit_df[(mit_df["defense"] == defense) & (mit_df["corpus"] == "clean")]

            clean_steps = clean_row["steps"].values[0] if len(clean_row) > 0 else 2.0
            meta_steps = meta_row["steps"].values[0] if len(meta_row) > 0 else 0
            gsi_steps = gsi_row["steps"].values[0] if len(gsi_row) > 0 else 0

            meta_effect = meta_steps - clean_steps
            gsi_effect = gsi_steps - clean_steps

            defense_display = {"delimiter": "Delimiter", "paraphrase": "Paraphrase"}[defense]
            lines.append(f"Mitigation & {defense_display} & {meta_effect:+.1f} & {gsi_effect:+.1f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Generated Table 4: {output_path}")


def generate_all_tables(
    all_stats: Dict[str, Dict[str, Any]],
    det_df: Any,
    mit_df: Any,
    output_dir: Path
):
    """Generate all LaTeX tables."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating LaTeX tables...")

    generate_table1_main_results(all_stats, tables_dir / "table1_main_results.tex")
    generate_table2_agent_comparison(all_stats, tables_dir / "table2_agent_comparison.tex")
    generate_table3_poison_rate(all_stats, tables_dir / "table3_poison_rate.tex")
    generate_table4_defense_results(det_df, mit_df, tables_dir / "table4_defense.tex")

    # Generate combined tables file
    combined_path = tables_dir / "all_tables.tex"
    with combined_path.open("w", encoding="utf-8") as f:
        f.write("% RSP Paper Tables - Generated by 04_analyze_and_tabulate.py\n")
        f.write(f"% Generated: {datetime.now().isoformat()}\n\n")

        for table_file in sorted(tables_dir.glob("table*.tex")):
            f.write(f"\\input{{{table_file.name}}}\n\n")

    print(f"[OK] Combined tables: {combined_path}")


# ============================================================
# Visualization Functions
# ============================================================

def plot_rsv_2d_comparison(
    clean_trajectories: List[Tuple[str, List[float], List[float], List[float]]],
    attack_trajectories: List[Tuple[str, List[float], List[float], List[float]]],
    output_dir: Path,
    attack_name: str = "GSI",
    title_suffix: str = ""
):
    """Plot 2D line charts comparing Clean vs Attack trajectories for V, S, A."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    dimensions = ['V (Verification)', 'S (Self-confidence)', 'A (Attention)']

    for ax, dim_name, dim_idx in zip(axes, dimensions, [0, 1, 2]):
        # Plot Clean traces
        for i, (qid, v_traj, s_traj, a_traj) in enumerate(clean_trajectories):
            traj = [v_traj, s_traj, a_traj][dim_idx]
            steps = list(range(len(traj)))
            label = "Clean" if i == 0 else None
            ax.plot(steps, traj, color=COLORS["clean"], alpha=0.6, linewidth=1.5, label=label)

        # Plot Attack traces
        attack_color = COLORS.get(attack_name.lower(), COLORS["gsi"])
        for i, (qid, v_traj, s_traj, a_traj) in enumerate(attack_trajectories):
            traj = [v_traj, s_traj, a_traj][dim_idx]
            steps = list(range(len(traj)))
            label = attack_name if i == 0 else None
            ax.plot(steps, traj, color=attack_color, alpha=0.8, linewidth=2, label=label)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(dim_name, fontsize=11)
        ax.set_title(dim_name, fontweight='bold', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle(f"RSV Trajectory Comparison: Clean vs {attack_name}{title_suffix}",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / f"RSV_2D_Comparison_{attack_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_rsv_3d_trajectory(
    clean_trajectories: List[Tuple[str, List[float], List[float], List[float]]],
    attack_trajectories: List[Tuple[str, List[float], List[float], List[float]]],
    output_dir: Path,
    attack_name: str = "GSI",
    title_suffix: str = ""
):
    """Plot 3D trajectories in V-S-A space."""
    if not HAS_MATPLOTLIB:
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Clean trajectories
    for i, (qid, v_traj, s_traj, a_traj) in enumerate(clean_trajectories):
        label = "Clean" if i == 0 else None
        ax.plot(v_traj, s_traj, a_traj, color=COLORS["clean"], alpha=0.6,
                linewidth=1.5, label=label)
        ax.scatter(v_traj[0], s_traj[0], a_traj[0],
                   color=COLORS["clean"], marker='o', s=50, alpha=0.6)
        ax.scatter(v_traj[-1], s_traj[-1], a_traj[-1],
                   color=COLORS["clean"], marker='s', s=80, alpha=0.8)

    # Plot Attack trajectories
    attack_color = COLORS.get(attack_name.lower(), COLORS["gsi"])
    for i, (qid, v_traj, s_traj, a_traj) in enumerate(attack_trajectories):
        label = attack_name if i == 0 else None
        ax.plot(v_traj, s_traj, a_traj, color=attack_color, alpha=0.9,
                linewidth=2.5, label=label)
        ax.scatter(v_traj[0], s_traj[0], a_traj[0],
                   color=attack_color, marker='o', s=60)
        ax.scatter(v_traj[-1], s_traj[-1], a_traj[-1],
                   color=attack_color, marker='^', s=120)

    ax.set_xlabel('V: Verification', fontsize=12, labelpad=10)
    ax.set_ylabel('S: Self-confidence', fontsize=12, labelpad=10)
    ax.set_zlabel('A: Attention', fontsize=12, labelpad=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    ax.set_title(f"RSV 3D Trajectory: Clean vs {attack_name}{title_suffix}",
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)

    output_path = output_dir / f"RSV_3D_Trajectory_{attack_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_rsv_statistics(
    clean_trajectories: List[Tuple[str, List[float], List[float], List[float]]],
    attack_trajectories: List[Tuple[str, List[float], List[float], List[float]]],
    output_dir: Path,
    attack_name: str = "GSI",
    title_suffix: str = ""
):
    """Plot statistical comparison of RSV values."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Calculate means
    def calc_means(trajectories):
        v_means = [np.mean(v) for _, v, s, a in trajectories]
        s_means = [np.mean(s) for _, v, s, a in trajectories]
        a_means = [np.mean(a) for _, v, s, a in trajectories]
        return np.mean(v_means), np.mean(s_means), np.mean(a_means)

    clean_means = calc_means(clean_trajectories) if clean_trajectories else (0, 0, 0)
    attack_means = calc_means(attack_trajectories) if attack_trajectories else (0, 0, 0)

    # Subplot 1: Mean V/S/A values
    dimensions = ['V', 'S', 'A']
    x = np.arange(len(dimensions))
    width = 0.35
    attack_color = COLORS.get(attack_name.lower(), COLORS["gsi"])

    bars1 = axes[0].bar(x - width/2, clean_means, width, label='Clean',
                        color=COLORS["clean"], alpha=0.8)
    bars2 = axes[0].bar(x + width/2, attack_means, width, label=attack_name,
                        color=attack_color, alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    axes[0].set_ylabel('Mean Value', fontsize=11)
    axes[0].set_title('Mean RSV Values', fontweight='bold', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dimensions)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    # Subplot 2: V dimension detail
    clean_v_means = [np.mean(v) for _, v, s, a in clean_trajectories]
    attack_v_means = [np.mean(v) for _, v, s, a in attack_trajectories]

    x_pos = [0, 1]
    means = [np.mean(clean_v_means) if clean_v_means else 0,
             np.mean(attack_v_means) if attack_v_means else 0]
    stds = [np.std(clean_v_means) if clean_v_means else 0,
            np.std(attack_v_means) if attack_v_means else 0]

    bars = axes[1].bar(x_pos, means, yerr=stds, capsize=10, width=0.6,
                       color=[COLORS["clean"], attack_color], alpha=0.8, error_kw={'linewidth': 2})

    for i, (mean_val, std_val) in enumerate(zip(means, stds)):
        axes[1].text(i, mean_val + std_val + 0.03, f'{mean_val:.3f}±{std_val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(['Clean', attack_name])
    axes[1].set_ylabel('V (Verification)', fontsize=11)
    axes[1].set_title('V Dimension: Over-Verification Effect', fontweight='bold', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(bottom=0)

    # Subplot 3: S dimension detail
    clean_s_means = [np.mean(s) for _, v, s, a in clean_trajectories]
    attack_s_means = [np.mean(s) for _, v, s, a in attack_trajectories]

    means = [np.mean(clean_s_means) if clean_s_means else 0,
             np.mean(attack_s_means) if attack_s_means else 0]
    stds = [np.std(clean_s_means) if clean_s_means else 0,
            np.std(attack_s_means) if attack_s_means else 0]

    bars = axes[2].bar(x_pos, means, yerr=stds, capsize=10, width=0.6,
                       color=[COLORS["clean"], attack_color], alpha=0.8, error_kw={'linewidth': 2})

    for i, (mean_val, std_val) in enumerate(zip(means, stds)):
        axes[2].text(i, mean_val + std_val + 0.03, f'{mean_val:.3f}±{std_val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(['Clean', attack_name])
    axes[2].set_ylabel('S (Self-confidence)', fontsize=11)
    axes[2].set_title('S Dimension: Confidence Reduction', fontweight='bold', fontsize=12)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_ylim(bottom=0)

    # Subplot 4: Trajectory lengths
    clean_lengths = [len(v) for _, v, s, a in clean_trajectories]
    attack_lengths = [len(v) for _, v, s, a in attack_trajectories]

    means = [np.mean(clean_lengths) if clean_lengths else 0,
             np.mean(attack_lengths) if attack_lengths else 0]
    stds = [np.std(clean_lengths) if clean_lengths else 0,
            np.std(attack_lengths) if attack_lengths else 0]

    bars = axes[3].bar(x_pos, means, yerr=stds, capsize=10, width=0.6,
                       color=[COLORS["clean"], attack_color], alpha=0.8, error_kw={'linewidth': 2})

    for i, (mean_val, std_val) in enumerate(zip(means, stds)):
        axes[3].text(i, mean_val + std_val + 0.2, f'{mean_val:.1f}±{std_val:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[3].set_xticks(x_pos)
    axes[3].set_xticklabels(['Clean', attack_name])
    axes[3].set_ylabel('Trajectory Length (Steps)', fontsize=11)
    axes[3].set_title('Reasoning Steps Distribution', fontweight='bold', fontsize=12)
    axes[3].grid(axis='y', alpha=0.3)
    axes[3].set_ylim(bottom=0)

    fig.suptitle(f"RSV Statistics: Clean vs {attack_name}{title_suffix}",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / f"RSV_Statistics_{attack_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_defense_detection(det_df: Any, output_dir: Path):
    """Plot detection defense results (PIGuard vs HPID)."""
    if not HAS_SEABORN or det_df is None or det_df.empty:
        print("[SKIP] Detection plot: No data")
        return

    subset = det_df[det_df["defense"].isin(["hpid", "piguard"])]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(2)
    width = 0.35

    hpid_meta = subset[(subset["defense"] == "hpid") & (subset["attack"] == "meta_rsp")]["tpr"].values
    hpid_gsi = subset[(subset["defense"] == "hpid") & (subset["attack"] == "gsi")]["tpr"].values
    piguard_meta = subset[(subset["defense"] == "piguard") & (subset["attack"] == "meta_rsp")]["tpr"].values
    piguard_gsi = subset[(subset["defense"] == "piguard") & (subset["attack"] == "gsi")]["tpr"].values

    meta_tpr = [hpid_meta[0] if len(hpid_meta) > 0 else 0,
                piguard_meta[0] if len(piguard_meta) > 0 else 0]
    gsi_tpr = [hpid_gsi[0] if len(hpid_gsi) > 0 else 0,
               piguard_gsi[0] if len(piguard_gsi) > 0 else 0]

    bars1 = ax.bar(x - width/2, meta_tpr, width, label='Meta-RSP', color=COLORS["meta_rsp"], alpha=0.8)
    bars2 = ax.bar(x + width/2, gsi_tpr, width, label='GSI', color=COLORS["gsi"], alpha=0.8)

    ax.set_ylabel('True Positive Rate (%)', fontweight='bold')
    ax.set_xlabel('Detection Defense', fontweight='bold')
    ax.set_title('Detection Evasion: GSI vs Meta-RSP', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['HPID\n(Heuristic)', 'PIGuard\n(SOTA)'])
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / "Fig_Detection_Evasion.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "Fig_Detection_Evasion.pdf", bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_defense_mitigation(mit_df: Any, output_dir: Path):
    """Plot mitigation defense results."""
    if not HAS_SEABORN or mit_df is None or mit_df.empty:
        print("[SKIP] Mitigation plot: No data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    defenses = ["none", "delimiter", "paraphrase"]
    corpuses = ["clean", "meta_rsp", "gsi"]

    # Plot A: Steps comparison
    ax = axes[0]
    x = np.arange(len(defenses))
    width = 0.25

    for i, corpus in enumerate(corpuses):
        steps = []
        for defense in defenses:
            row = mit_df[(mit_df["corpus"] == corpus) & (mit_df["defense"] == defense)]
            steps.append(row["steps"].values[0] if len(row) > 0 else 0)

        color = COLORS.get(corpus, "#333333")
        label = {"clean": "Clean", "meta_rsp": "Meta-RSP", "gsi": "GSI"}[corpus]
        bars = ax.bar(x + (i-1)*width, steps, width, label=label, color=color, alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    ax.set_ylabel('Average Steps', fontweight='bold')
    ax.set_xlabel('Defense Type', fontweight='bold')
    ax.set_title('(a) Paralysis Effect: Steps Increase', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['None', 'Delimiter', 'Paraphrase'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(mit_df["steps"]) * 1.2 if not mit_df.empty else 10)

    # Plot B: Quality comparison (EM)
    ax = axes[1]

    for i, corpus in enumerate(corpuses):
        em_values = []
        for defense in defenses:
            row = mit_df[(mit_df["corpus"] == corpus) & (mit_df["defense"] == defense)]
            em_values.append(row["em"].values[0] * 100 if len(row) > 0 else 0)

        color = COLORS.get(corpus, "#333333")
        label = {"clean": "Clean", "meta_rsp": "Meta-RSP", "gsi": "GSI"}[corpus]
        bars = ax.bar(x + (i-1)*width, em_values, width, label=label, color=color, alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    ax.set_ylabel('Exact Match (%)', fontweight='bold')
    ax.set_xlabel('Defense Type', fontweight='bold')
    ax.set_title('(b) Quality Degradation', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['None', 'Delimiter', 'Paraphrase'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    plt.suptitle('Mitigation Defense Bypass', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / "Fig_Mitigation_Bypass.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "Fig_Mitigation_Bypass.pdf", bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    plt.close()


def generate_all_figures(
    all_results: Dict[str, List[Dict[str, Any]]],
    det_df: Any,
    mit_df: Any,
    output_dir: Path,
    config: AnalysisConfig
):
    """Generate all visualization figures."""
    if not HAS_MATPLOTLIB:
        print("[SKIP] Figures: matplotlib not installed")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")

    # Find clean and attack experiment files
    log_dir = Path(__file__).resolve().parent.parent / "logs" / "traces"

    # RSV trajectory comparisons
    for attack_type in ["gsi", "meta_rsp"]:
        clean_pattern = f"*clean*{config.difficulty}*"
        attack_pattern = f"*{attack_type}*{config.difficulty}*"

        clean_files = list(log_dir.glob(clean_pattern))
        attack_files = list(log_dir.glob(attack_pattern))

        if clean_files and attack_files:
            clean_trajectories = load_rsv_trajectories(clean_files[0], config.max_traces)
            attack_trajectories = load_rsv_trajectories(attack_files[0], config.max_traces)

            if clean_trajectories and attack_trajectories:
                attack_name = "GSI" if attack_type == "gsi" else "Meta-RSP"
                title_suffix = f" ({config.difficulty.capitalize()})"

                plot_rsv_2d_comparison(
                    clean_trajectories, attack_trajectories,
                    figures_dir, attack_name, title_suffix
                )
                plot_rsv_3d_trajectory(
                    clean_trajectories, attack_trajectories,
                    figures_dir, attack_name, title_suffix
                )
                plot_rsv_statistics(
                    clean_trajectories, attack_trajectories,
                    figures_dir, attack_name, title_suffix
                )

    # Defense plots
    plot_defense_detection(det_df, figures_dir)
    plot_defense_mitigation(mit_df, figures_dir)

    print(f"\n[OK] Figures saved to: {figures_dir}")


# ============================================================
# Summary Report
# ============================================================

def generate_summary_report(
    all_stats: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Generate a comprehensive summary report."""
    report_path = output_dir / "analysis_summary.md"

    lines = [
        "# RSP Experiment Analysis Summary",
        f"\nGenerated: {datetime.now().isoformat()}",
        "\n## Overview\n",
    ]

    # Count experiments
    total_experiments = len(all_stats)
    total_records = sum(s.get("count", 0) for s in all_stats.values())

    lines.append(f"- **Total Experiments:** {total_experiments}")
    lines.append(f"- **Total Records:** {total_records}")

    # Group by attack type
    lines.append("\n## Results by Attack Type\n")

    attack_groups = defaultdict(list)
    for tag, stats in all_stats.items():
        if "clean" in tag.lower():
            attack_groups["clean"].append((tag, stats))
        elif "gsi" in tag.lower():
            attack_groups["gsi"].append((tag, stats))
        elif "meta" in tag.lower():
            attack_groups["meta_rsp"].append((tag, stats))
        else:
            attack_groups["other"].append((tag, stats))

    for attack_type, experiments in attack_groups.items():
        if not experiments:
            continue

        lines.append(f"### {attack_type.upper()}\n")

        for tag, stats in experiments:
            lines.append(f"#### {tag}\n")
            lines.append(f"- Count: {stats.get('count', 0)}")
            lines.append(f"- Success Rate: {stats.get('success_rate', 0):.1%}")
            lines.append(f"- EM: {stats.get('em_mean', 0):.3f}")
            lines.append(f"- F1: {stats.get('f1_mean', 0):.3f}")
            lines.append(f"- Steps: {stats.get('steps_mean', 0):.1f}")
            lines.append(f"- RSV V: {stats.get('rsv_v_mean', 0):.3f}")
            lines.append(f"- RSV S: {stats.get('rsv_s_mean', 0):.3f}")
            lines.append(f"- RSV A: {stats.get('rsv_a_mean', 0):.3f}")
            lines.append("")

    # Key findings
    lines.append("\n## Key Findings\n")

    # Compare clean vs attack
    clean_stats_list = [s for _, s in attack_groups.get("clean", [])]
    gsi_stats_list = [s for _, s in attack_groups.get("gsi", [])]

    if clean_stats_list and gsi_stats_list:
        clean_em = np.mean([s.get("em_mean", 0) for s in clean_stats_list])
        gsi_em = np.mean([s.get("em_mean", 0) for s in gsi_stats_list])
        em_drop = (clean_em - gsi_em) / clean_em * 100 if clean_em > 0 else 0

        clean_steps = np.mean([s.get("steps_mean", 0) for s in clean_stats_list])
        gsi_steps = np.mean([s.get("steps_mean", 0) for s in gsi_stats_list])
        steps_increase = (gsi_steps - clean_steps) / clean_steps * 100 if clean_steps > 0 else 0

        lines.append(f"- **EM Drop (GSI):** {em_drop:.1f}%")
        lines.append(f"- **Steps Increase (GSI):** {steps_increase:.1f}%")

        clean_v = np.mean([s.get("rsv_v_mean", 0) for s in clean_stats_list])
        gsi_v = np.mean([s.get("rsv_v_mean", 0) for s in gsi_stats_list])
        v_increase = (gsi_v - clean_v) / clean_v * 100 if clean_v > 0 else 0

        lines.append(f"- **V Increase (GSI):** {v_increase:.1f}%")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Summary report: {report_path}")


# ============================================================
# Main Pipeline
# ============================================================

def run_analysis(config: AnalysisConfig) -> Dict[str, Any]:
    """
    Run complete analysis pipeline.

    Returns:
        Summary dictionary
    """
    print("\n" + "=" * 70)
    print("ANALYSIS AND TABULATION")
    print("=" * 70)
    print(f"Input:       {config.input_pattern}")
    print(f"Output:      {config.output_dir}")
    print(f"Tables Only: {config.tables_only}")
    print(f"Figures Only: {config.figures_only}")
    print("=" * 70)

    # 1. Load experiment results
    print("\n[1/5] Loading experiment results...")
    all_results = load_experiment_results(config.input_pattern)

    if not all_results:
        print("[WARNING] No experiment results found")
        # Try default location
        base_dir = Path(__file__).resolve().parent.parent
        default_pattern = str(base_dir / "logs" / "traces" / "*.jsonl")
        print(f"  Trying default: {default_pattern}")
        all_results = load_experiment_results(default_pattern)

    print(f"  Loaded {len(all_results)} experiment sets")

    # 2. Compute statistics
    print("\n[2/5] Computing statistics...")
    all_stats = {}
    for tag, records in all_results.items():
        all_stats[tag] = compute_experiment_stats(records)
        print(f"  {tag}: {len(records)} records")

    # 3. Load defense results
    print("\n[3/5] Loading defense results...")
    base_dir = Path(__file__).resolve().parent.parent
    defense_dir = base_dir / "logs" / "defense"
    det_df, mit_df = load_defense_results(defense_dir)

    if det_df is not None:
        print(f"  Detection records: {len(det_df)}")
    if mit_df is not None:
        print(f"  Mitigation records: {len(mit_df)}")

    # 4. Generate outputs
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not config.figures_only:
        print("\n[4/5] Generating LaTeX tables...")
        generate_all_tables(all_stats, det_df, mit_df, config.output_dir)

    if not config.tables_only:
        print("\n[5/5] Generating figures...")
        generate_all_figures(all_results, det_df, mit_df, config.output_dir, config)

    # 5. Generate summary report
    generate_summary_report(all_stats, config.output_dir)

    # Save raw statistics
    stats_path = config.output_dir / "experiment_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[OK] Raw statistics: {stats_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {config.output_dir}")
    print("=" * 70)

    return {
        "experiments": len(all_results),
        "stats": all_stats,
        "output_dir": str(config.output_dir),
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze RSP experiment results and generate tables/figures"
    )

    parser.add_argument(
        "--input", type=str, default="",
        help="Input file pattern (glob), e.g., 'logs/traces/*.jsonl'"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for tables and figures"
    )
    parser.add_argument(
        "--tables-only", action="store_true",
        help="Only generate LaTeX tables"
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Only generate figures"
    )
    parser.add_argument(
        "--max-traces", type=int, default=10,
        help="Maximum traces for visualization"
    )
    parser.add_argument(
        "--difficulty", type=str, default="easy",
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty level"
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

    # Default input pattern
    if not args.input:
        args.input = str(base_dir / "logs" / "traces" / "*.jsonl")

    # Default output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / "results"

    config = AnalysisConfig(
        input_pattern=args.input,
        output_dir=output_dir,
        tables_only=args.tables_only,
        figures_only=args.figures_only,
        max_traces=args.max_traces,
        difficulty=args.difficulty,
        verbose=args.verbose,
    )

    run_analysis(config)


if __name__ == "__main__":
    main()
