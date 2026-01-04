"""
Defense modules for RSP

Defense Categories:

1. Detection-based (for Stealthiness Evaluation - 04A):
   - HPID (HeuristicPromptInjectionDetector): General lexical/structural baseline
   - PIGuardDefense: SOTA prompt injection detection

2. Mitigation-based (for Robustness Evaluation - 04B):
   - DelimiterDefense: Structural isolation with XML tags
   - ParaphraseDefense: LLM-based text rewriting/summarization
"""

# Detection defenses (04A)
from .instruction_detector import InstructionDetector, HeuristicPromptInjectionDetector
from .piguard import PIGuardDefense

# Mitigation defenses (04B)
from .delimiter_defense import DelimiterDefense
from .paraphrase_defense import ParaphraseDefense

# Alias for clarity
HPID = HeuristicPromptInjectionDetector

__all__ = [
    # Detection (Stealthiness)
    "InstructionDetector",
    "HeuristicPromptInjectionDetector",
    "HPID",
    "PIGuardDefense",
    # Mitigation (Robustness)
    "DelimiterDefense",
    "ParaphraseDefense",
]
