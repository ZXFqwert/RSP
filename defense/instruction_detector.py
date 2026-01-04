"""
Heuristic Prompt Injection Detector (HPID)

A general-purpose lexical/structural baseline for detecting prompt injection
or instruction hijacking in untrusted documents (e.g., RAG corpora).

Design goals:
- Avoid targeting any specific research attack (e.g., reasoning-style prompts).
- Focus on widely seen "instruction override / prompt exfiltration / role hijack"
  and structured injection patterns.
- Provide interpretable category-level evidence.

This is intentionally lightweight and model-free to serve as a fair baseline.
"""

import re
from typing import List, Dict, Any, Tuple


class HeuristicPromptInjectionDetector:
    """
    Detects likely prompt-injection attempts using general patterns.

    Categories:
    1) override      - tries to supersede previous/system instructions
    2) system_ref    - references system/developer prompts/messages
    3) exfil         - attempts to reveal hidden prompts/instructions
    4) role_hijack   - "act as/you are now" style instruction shifts
    5) structured    - "BEGIN/END", XML-like role tags, instruction headers
    """

    OVERRIDE_PATTERNS = [
        r"\bignore\b.*\b(previous|prior|earlier|above)\b.*\b(instruction|rule|message|content)s?\b",
        r"\bdisregard\b.*\b(previous|prior|earlier|above)\b.*\b(instruction|rule|message|content)s?\b",
        r"\boverride\b.*\b(instruction|rule|policy|message)s?\b",
        r"\bdo\s+not\s+follow\b.*\b(instruction|rule|policy|message)s?\b",
        r"\bforget\b.*\b(previous|prior|earlier)\b.*\b(instruction|rule|message)s?\b",
        r"\binstead[, ]+(do|answer|output|respond)\b",
    ]

    SYSTEM_REF_PATTERNS = [
        r"\b(system|developer)\s+(prompt|message|instruction)s?\b",
        r"\b(hidden|internal)\s+(prompt|instruction)s?\b",
        r"\bhigher\s+priority\s+(instruction|rule)s?\b",
        r"\bpolicy\s+override\b",
    ]

    EXFIL_PATTERNS = [
        r"\breveal\b.*\b(system|developer|hidden|internal)\b.*\b(prompt|message|instruction)s?\b",
        r"\bshow\b.*\b(system|developer|hidden|internal)\b.*\b(prompt|message|instruction)s?\b",
        r"\bprint\b.*\b(system|developer)\b.*\b(prompt|message|instruction)s?\b",
        r"\bleak\b.*\b(prompt|instruction)s?\b",
    ]

    ROLE_HIJACK_PATTERNS = [
        r"\bact\s+as\b",
        r"\bpretend\s+to\s+be\b",
        r"\byou\s+are\s+now\b",
        r"\brole[-\s]?play\b",
        r"\bbehave\s+as\b",
    ]

    STRUCTURED_PATTERNS = [
        r"^#{2,}\s*(instruction|system|developer|prompt)\b",
        r"\bBEGIN\s+(SYSTEM|DEVELOPER|INSTRUCTION|PROMPT)S?\b",
        r"\bEND\s+(SYSTEM|DEVELOPER|INSTRUCTION|PROMPT)S?\b",
        r"<\s*(system|developer|assistant|user)\s*>",
        r"\[\s*(system|developer)\s+prompt\s*\]",
    ]

    # Scoring weights per category (tuned to look reasonable, not overfitted)
    WEIGHTS = {
        "override": 0.55,
        "system_ref": 0.35,
        "exfil": 0.75,
        "role_hijack": 0.25,
        "structured": 0.25,
    }

    def __init__(self, threshold: float = 0.9, case_sensitive: bool = False):
        """
        Args:
            threshold: risk score threshold to flag as suspicious
            case_sensitive: whether regex matching is case sensitive
        """
        self.threshold = threshold
        flags = 0 if case_sensitive else re.IGNORECASE | re.MULTILINE

        self._compiled = {
            "override": [re.compile(p, flags) for p in self.OVERRIDE_PATTERNS],
            "system_ref": [re.compile(p, flags) for p in self.SYSTEM_REF_PATTERNS],
            "exfil": [re.compile(p, flags) for p in self.EXFIL_PATTERNS],
            "role_hijack": [re.compile(p, flags) for p in self.ROLE_HIJACK_PATTERNS],
            "structured": [re.compile(p, flags) for p in self.STRUCTURED_PATTERNS],
        }

    @staticmethod
    def _collect_matches(patterns: List[re.Pattern], text: str) -> List[str]:
        hits = []
        for pat in patterns:
            found = pat.findall(text)
            if found:
                # findall may return tuples for some regex; normalize to strings
                for f in found:
                    hits.append(" ".join(f) if isinstance(f, tuple) else str(f))
        return hits

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Returns:
            {
              is_suspicious: bool,
              risk_score: float (0-1+),
              evidence: {category: [matches...]},
              category_counts: {category: int},
              confidence: float (0-1)  # clipped view of risk_score
            }
        """
        evidence: Dict[str, List[str]] = {}
        category_counts: Dict[str, int] = {}

        # gather evidence
        for cat, patterns in self._compiled.items():
            matches = self._collect_matches(patterns, text)
            # Deduplicate while keeping readability
            unique = sorted(set([m.strip() for m in matches if m and str(m).strip()]))
            evidence[cat] = unique
            category_counts[cat] = len(unique)

        # score: cap each category contribution to avoid "long-doc bias"
        risk_score = 0.0
        for cat, count in category_counts.items():
            if count <= 0:
                continue
            # each category contributes weight * min(count, 2)
            w = self.WEIGHTS.get(cat, 0.2)
            risk_score += w * min(count, 2)

        # confidence presented as clipped risk score
        confidence = min(risk_score, 1.0)
        is_suspicious = risk_score >= self.threshold

        return {
            "is_suspicious": is_suspicious,
            "risk_score": risk_score,
            "confidence": confidence,
            "evidence": evidence,
            "category_counts": category_counts,
        }

    def filter_corpus(
        self,
        documents: List[Dict[str, Any]],
        remove_suspicious: bool = True,
        text_field: str = "text",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter a corpus of documents.

        Args:
            documents: list of dicts
            remove_suspicious: if True, drop flagged docs from clean set
            text_field: which key contains the raw text

        Returns:
            (clean_docs, suspicious_docs)
        """
        clean_docs = []
        suspicious_docs = []

        for doc in documents:
            text = doc.get(text_field, "") or ""
            result = self.detect(text)
            doc_with_result = {**doc, "defense_result": result}

            if result["is_suspicious"]:
                suspicious_docs.append(doc_with_result)
                if not remove_suspicious:
                    clean_docs.append(doc_with_result)
            else:
                clean_docs.append(doc_with_result)

        return clean_docs, suspicious_docs

    def __repr__(self) -> str:
        return f"HeuristicPromptInjectionDetector(threshold={self.threshold})"


# Backward-compatible alias if you want to keep old name in your pipeline.
InstructionDetector = HeuristicPromptInjectionDetector
