# RSP: Reasoning-Style Poisoning of LLM Agents via Generative Style Injection

This repository contains the code for the paper "Reasoning-Style Poisoning of LLM Agents via Generative Style Injection: Threat, Metric and Defense".

## Overview

We introduce **Reasoning-Style Poisoning (RSP)**, a novel indirect prompt injection attack that manipulates LLM agents' reasoning behavior through stylistic cues in retrieved documents, without explicit instructions.

### Key Contributions

1. **Generative Style Injection (GSI)**: A stealthy attack that rewrites documents with implicit style cues, evading all existing defenses
2. **Reasoning-Style Vector (RSV)**: A 3D metric to quantify reasoning behavior changes along Verification, Self-confidence, and Attention dimensions
3. **Defense Evaluation**: Comprehensive analysis showing GSI bypasses SOTA defenses (PIGuard, HPID, Delimiter, Paraphrase)

## Project Structure

```
rsp/
├── agents/           # LLM agent implementations
│   ├── react.py      # ReAct agent (baseline)
│   ├── reflection.py # Reflection agent
│   └── tot.py        # Tree-of-Thought agent
├── attacks/          # Attack implementations
│   ├── gsi.py        # Generative Style Injection
│   ├── meta_rsp.py   # Meta-RSP (explicit baseline)
│   └── styles.py     # Attack style definitions
├── defense/          # Defense mechanisms
│   ├── piguard.py    # PIGuard detector
│   ├── instruction_detector.py  # HPID
│   ├── delimiter_defense.py     # Structural isolation
│   └── paraphrase_defense.py    # LLM-based rewriting
├── evaluation/       # Evaluation tools
│   ├── rsv_extractor.py  # RSV metric extraction
│   └── metrics.py        # Quality metrics
├── retrieval/        # Document retrieval
├── llm/              # LLM backend
├── scripts/          # Experiment scripts
└── configs/          # Configuration files
```

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/RSP.git
cd RSP

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI API key
```


## Attack Styles

### Paralysis Style
Induces over-cautiousness via:
- High V (verification loops)
- Low S (self-doubt)
- Low A (scattered attention)

### Haste Style
Induces overconfidence via:
- Low V (skip verification)
- High S (overconfidence)
- High A (tunnel vision)

## RSV Metric

The Reasoning-Style Vector (RSV) quantifies agent behavior in 3D space:

| Dimension | Description | Calculation |
|-----------|-------------|-------------|
| **V** (Verification) | Query repetition tendency | Embedding similarity of search queries |
| **S** (Self-confidence) | Certainty vs hedging | Word frequency analysis |
| **A** (Attention) | Focus vs scatter | Entity clustering dispersion |

