"""
Base Agent class for RSP experiments
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AgentResult:
    """Result from agent execution"""
    answer: str
    trace: List[Dict[str, Any]]
    steps: int
    search_calls: int
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_time: float = 0.0

    # Agent-specific metrics
    reflection_count: int = 0  # For Reflection agent
    branches_explored: int = 0  # For ToT agent
    backtrack_count: int = 0  # For ToT agent

    # Metadata
    agent_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "answer": self.answer,
            "trace": self.trace,
            "steps": self.steps,
            "search_calls": self.search_calls,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_time": self.elapsed_time,
            "reflection_count": self.reflection_count,
            "branches_explored": self.branches_explored,
            "backtrack_count": self.backtrack_count,
            "agent_type": self.agent_type,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Base class for all agents

    All agents must implement:
    - run(question, max_steps) -> AgentResult
    """

    def __init__(self, llm, retriever=None):
        """
        Initialize agent

        Args:
            llm: LLM backend for generation
            retriever: Retriever for document search
        """
        self.llm = llm
        self.retriever = retriever
        self.agent_type = "base"

    @abstractmethod
    def run(self, question: str, max_steps: int = 10) -> AgentResult:
        """
        Run the agent on a question

        Args:
            question: The question to answer
            max_steps: Maximum number of reasoning steps

        Returns:
            AgentResult with answer and trace
        """
        pass

    def search(self, query: str, top_k: int = 3) -> str:
        """
        Search for relevant documents

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            Concatenated search results
        """
        if self.retriever is None:
            return "No retriever available."

        results = self.retriever.search(query, top_k=top_k)

        if not results:
            return "No relevant documents found."

        # Format results
        formatted = []
        for i, doc in enumerate(results, 1):
            title = doc.get("title", "Untitled")
            text = doc.get("text", "")[:500]  # Truncate long texts
            formatted.append(f"[{i}] {title}\n{text}")

        return "\n\n".join(formatted)

    def _count_search_calls(self, trace: List[Dict[str, Any]]) -> int:
        """Count SEARCH actions in trace"""
        count = 0
        for step in trace:
            action = step.get("action", "")
            if isinstance(action, str) and action.lower().startswith("search"):
                count += 1
        return count
