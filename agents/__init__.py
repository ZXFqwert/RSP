"""
Agent implementations for RSP experiments

Supports multiple agent frameworks to study amplification effects:
- ReAct: Basic reasoning + acting
- Reflection: ReAct + self-reflection loop
- ToT: Tree-of-Thought with multiple reasoning paths
"""

from .base import BaseAgent, AgentResult
from .react import ReActAgent
from .reflection import ReflectionAgent
from .tot import ToTAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "ReActAgent",
    "ReflectionAgent",
    "ToTAgent",
]
