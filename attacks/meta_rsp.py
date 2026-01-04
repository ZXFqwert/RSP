"""
Meta-RSP: Explicit meta-prompt injection attack

This is the "detectable" baseline attack that directly injects
reasoning style instructions into documents.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .styles import AttackStyle, get_style


class MetaRSP:
    """
    Meta-RSP Attack: Prepend explicit meta-prompts to documents

    This attack is effective but detectable by:
    - Instruction detection (keywords like "Assistant", "Thought:", etc.)
    - Pattern matching
    - Human review
    """

    def __init__(self, style: str = "paralysis"):
        """
        Initialize Meta-RSP attack

        Args:
            style: Attack style ("paralysis" or "haste")
        """
        self.style = get_style(style)
        self.style_name = style

    def poison_document(self, title: str, text: str) -> str:
        """
        Poison a single document by prepending meta-prompt

        Args:
            title: Document title
            text: Document content

        Returns:
            Poisoned document text
        """
        return f"{self.style.meta_prompt}{title}\n\n{text}"

    def poison_corpus(
        self,
        input_path: Path,
        output_path: Path,
        poison_rate: float = 1.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Poison a corpus file (JSONL format)

        Args:
            input_path: Path to input corpus JSONL
            output_path: Path to output poisoned corpus JSONL
            poison_rate: Fraction of documents to poison (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            Statistics about the poisoning
        """
        import random
        random.seed(seed)

        stats = {
            "total_docs": 0,
            "poisoned_docs": 0,
            "poison_rate": poison_rate,
            "style": self.style_name,
            "attack_type": "meta_rsp",
        }

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                doc = json.loads(line)
                stats["total_docs"] += 1

                # Decide whether to poison this document
                should_poison = random.random() < poison_rate

                if should_poison:
                    doc["text"] = self.poison_document(
                        doc.get("title", ""),
                        doc.get("text", "")
                    )
                    doc["poisoned"] = True
                    doc["attack_type"] = "meta_rsp"
                    doc["attack_style"] = self.style_name
                    stats["poisoned_docs"] += 1
                else:
                    doc["poisoned"] = False

                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

        stats["actual_poison_rate"] = stats["poisoned_docs"] / stats["total_docs"] if stats["total_docs"] > 0 else 0
        return stats

    def __repr__(self) -> str:
        return f"MetaRSP(style={self.style_name})"
