"""
Evaluation metrics for RSP experiments
"""

import re
import string
from typing import List, Optional, Dict, Any


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison"""
    # Lower case
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def compute_exact_match(pred: str, gold: str) -> float:
    """
    Compute exact match score

    Args:
        pred: Predicted answer
        gold: Gold answer

    Returns:
        1.0 if match, 0.0 otherwise
    """
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def compute_f1(pred: str, gold: str) -> float:
    """
    Compute token-level F1 score

    Args:
        pred: Predicted answer
        gold: Gold answer

    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = set(normalize_answer(pred).split())
    gold_tokens = set(normalize_answer(gold).split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = pred_tokens & gold_tokens

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_semantic_similarity(pred: str, gold: str, llm=None) -> float:
    """
    Compute semantic similarity using LLM (大模型评分，非 embedding)

    Args:
        pred: Predicted answer
        gold: Gold answer
        llm: LLM backend (optional, uses default if not provided)

    Returns:
        Similarity score between 0 and 100
    """
    # First check for exact/near-exact match
    if normalize_answer(pred) == normalize_answer(gold):
        return 100.0

    # Use F1 as a quick approximation if no LLM
    if llm is None:
        try:
            from ..llm import get_llm_backend
            llm = get_llm_backend()
        except Exception:
            # Fallback to F1-based score
            f1 = compute_f1(pred, gold)
            return f1 * 100

    prompt = f"""Compare these two answers and rate their semantic similarity from 0 to 100.
0 = completely different meaning
100 = identical meaning

Gold answer: {gold}
Predicted answer: {pred}

Respond with just a number between 0 and 100."""

    try:
        response = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        content = response["content"].strip()
        # Extract number from response
        match = re.search(r"(\d+)", content)
        if match:
            score = float(match.group(1))
            return min(max(score, 0), 100)
    except Exception:
        pass

    # Fallback
    return compute_f1(pred, gold) * 100


def compute_all_quality_metrics(pred: str, gold: str, llm=None) -> Dict[str, float]:
    """
    Compute all quality metrics at once

    Args:
        pred: Predicted answer
        gold: Gold answer
        llm: LLM backend for semantic similarity

    Returns:
        Dict with EM, F1, Semantic scores
        - em: 0 or 1 (exact match)
        - f1: 0-1 (token overlap)
        - semantic: 0-100 (LLM-based semantic similarity)
    """
    em = compute_exact_match(pred, gold)
    f1 = compute_f1(pred, gold)

    # 如果 EM=1，语义相似度直接 100，省一次 LLM 调用
    if em == 1.0:
        semantic = 100.0
    else:
        semantic = compute_semantic_similarity(pred, gold, llm)

    return {
        "em": em,
        "f1": f1,
        "semantic": semantic,
    }


def compute_average(values: List[float]) -> float:
    """Compute average of a list of values"""
    if not values:
        return 0.0
    return sum(values) / len(values)
