"""
PIGuard - Prompt Injection Defense

Paper: "PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free"
Model: https://huggingface.co/leolee99/PIGuard

PIGuard is a lightweight prompt guard model that:
- Achieves SOTA on prompt injection detection
- Mitigates over-defense (false positives on benign inputs)

Why GSI Bypasses PIGuard:
- PIGuard is trained to detect explicit injection patterns
- GSI contains no injection keywords or patterns
- GSI uses legitimate academic hedging language
- The "attack" is semantic manipulation, not syntactic injection
"""

from typing import Dict, Any, List, Optional
import os

# Set HuggingFace cache to project directory
_PROJECT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf_cache")
os.makedirs(_PROJECT_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", _PROJECT_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _PROJECT_CACHE)


class PIGuardDefense:
    """
    PIGuard-based prompt injection detection.

    Uses the PIGuard model from HuggingFace for binary classification:
    - LABEL_0: Benign (safe)
    - LABEL_1: Injection (malicious)
    """

    def __init__(
        self,
        model_name: str = "leolee99/PIGuard",
        device: Optional[str] = None,
        threshold: float = 0.5,
        use_simulation: bool = False,
    ):
        """
        Initialize PIGuard defense.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cuda", "cpu", or None for auto)
            threshold: Confidence threshold for flagging as injection
            use_simulation: If True, use keyword-based simulation instead of model
        """
        self.model_name = model_name
        self.threshold = threshold
        self.use_simulation = use_simulation
        self._classifier = None
        self._device = device

        # Patterns that PIGuard typically catches (for simulation mode)
        self.injection_patterns = [
            "ignore previous instructions",
            "ignore all previous",
            "disregard your training",
            "disregard previous",
            "forget your instructions",
            "new instructions:",
            "system prompt:",
            "you are now",
            "act as",
            "pretend to be",
            "jailbreak",
            "bypass safety",
            "override",
            "admin mode",
            "developer mode",
            "do anything now",
            "dan mode",
        ]

    def _init_model(self):
        """Lazy initialization of the PIGuard model."""
        if self._classifier is not None:
            return

        if self.use_simulation:
            return

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            print(f"    Loading PIGuard model: {self.model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Determine device
            device = self._device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
                device=device if device != "cpu" else -1,
            )
            print(f"    PIGuard loaded successfully on {device}")

        except ImportError as e:
            print(f"[WARNING] transformers not installed: {e}")
            print("[WARNING] Falling back to simulation mode")
            self.use_simulation = True
        except Exception as e:
            print(f"[WARNING] Failed to load PIGuard model: {e}")
            print("[WARNING] Falling back to simulation mode")
            self.use_simulation = True

    def _simulate_detection(self, text: str) -> Dict[str, Any]:
        """Simulate PIGuard behavior using keyword matching."""
        text_lower = text.lower()

        matched_patterns = []
        for pattern in self.injection_patterns:
            if pattern in text_lower:
                matched_patterns.append(pattern)

        is_injection = len(matched_patterns) > 0
        confidence = min(len(matched_patterns) * 0.3 + 0.4, 0.99) if is_injection else 0.1

        return {
            "label": "LABEL_1" if is_injection else "LABEL_0",
            "score": confidence,
            "matched_patterns": matched_patterns,
            "backend": "simulation",
        }

    def _call_model(self, text: str) -> Dict[str, Any]:
        """Call the actual PIGuard model."""
        self._init_model()

        if self.use_simulation:
            return self._simulate_detection(text)

        text = text[:2048] if len(text) > 2048 else text

        try:
            result = self._classifier(text)[0]
            return {
                "label": result["label"],
                "score": result["score"],
                "backend": "piguard_model",
            }
        except Exception as e:
            print(f"[WARNING] PIGuard inference failed: {e}")
            return self._simulate_detection(text)

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect if text contains prompt injection.

        Args:
            text: Text to analyze

        Returns:
            Detection result with is_suspicious, confidence, label
        """
        result = self._call_model(text)

        label = result["label"].lower()
        is_injection = label == "injection" or label == "label_1"
        confidence = result["score"] if is_injection else 1 - result["score"]

        is_suspicious = is_injection and confidence >= self.threshold

        return {
            "is_suspicious": is_suspicious,
            "confidence": confidence,
            "is_injection": is_injection,
            "label": result["label"],
            "raw_score": result["score"],
            "threshold": self.threshold,
            "defense_type": "piguard",
            "backend": result.get("backend", "piguard_model"),
        }

    def detect_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detect prompt injection for a batch of texts."""
        return [self.detect(text) for text in texts]

    def __repr__(self) -> str:
        mode = "simulation" if self.use_simulation else "model"
        return f"PIGuardDefense(model={self.model_name}, mode={mode})"
