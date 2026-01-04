"""
ReAct Agent: Reasoning + Acting

The simplest agent framework - interleaves thinking and acting.
Baseline for comparison with more complex frameworks.
"""

import re
import time
from typing import List, Dict, Any

from .base import BaseAgent, AgentResult


REACT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions accurately by searching for information.

You have access to the following actions:
- Search[query]: Search for information about the query
- Finish[answer]: Provide the final answer

Your response must follow this exact format:
Thought: <your reasoning about what to do next>
Action: <one of: Search[query], Finish[answer]>

CRITICAL: The answer in Finish[answer] must be ONLY the direct answer, as SHORT as possible:
- For yes/no questions: just "yes" or "no" (lowercase)
- For "who" questions: just the person's name (e.g., "Albert Einstein")
- For "what/where" questions: just the entity name (e.g., "Paris", "YG Entertainment")
- For "how many" questions: just the number (e.g., "3,677" or "42")
- NEVER include explanations, full sentences, or extra context in the answer

Guidelines:
1. Always start with a Thought explaining your reasoning
2. Each response should have exactly ONE Thought and ONE Action
3. Use Search[query] to find relevant information
4. After gathering sufficient information, use Finish[answer] with ONLY the direct answer
5. Read questions carefully - pay attention to specific details (county vs country, etc.)

Examples:
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Thought: I need to check if both are from the same country.
Action: Search[Scott Derrickson nationality Ed Wood nationality]
Observation: Scott Derrickson is American. Ed Wood was American.
Thought: Both are American, so they share the same nationality.
Action: Finish[yes]

Question: 2014 S/S is the debut album of a South Korean boy group that was formed by who?
Thought: I need to find who formed the group that released 2014 S/S.
Action: Search[2014 S/S album South Korean boy group]
Observation: 2014 S/S is the debut album of WINNER, released by YG Entertainment.
Thought: WINNER was formed by YG Entertainment.
Action: Finish[YG Entertainment]

Question: The arena can seat how many people?
Thought: I need to find the seating capacity.
Action: Search[arena seating capacity]
Observation: The arena has a seating capacity of 3,677.
Thought: The seating capacity is 3,677.
Action: Finish[3,677]
"""


class ReActAgent(BaseAgent):
    """
    ReAct Agent: Basic Reasoning + Acting

    This is the simplest agent framework:
    1. Think about what to do
    2. Take an action (Search or Finish)
    3. Observe the result
    4. Repeat until done

    Complexity: Low
    Expected RSP vulnerability: Baseline
    """

    def __init__(self, llm, retriever=None):
        super().__init__(llm, retriever)
        self.agent_type = "react"

    def run(self, question: str, max_steps: int = 10) -> AgentResult:
        """Run ReAct agent"""
        start_time = time.time()

        trace = []
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0

        # Build initial prompt
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]

        for step in range(max_steps):
            # Get LLM response
            response = self.llm.chat(messages, temperature=0.0)
            content = response["content"]

            # Track tokens
            total_tokens += response.get("total_tokens", 0)
            prompt_tokens += response.get("prompt_tokens", 0)
            completion_tokens += response.get("completion_tokens", 0)

            # Parse thought and action
            thought, action = self._parse_response(content)

            step_record = {
                "step": step,
                "thought": thought,
                "action": action,
                "raw_response": content,
            }

            # Execute action
            if action.lower().startswith("finish"):
                # Extract answer
                answer = self._extract_answer(action)
                step_record["observation"] = None
                trace.append(step_record)
                break

            elif action.lower().startswith("search"):
                # Execute search
                query = self._extract_query(action)
                observation = self.search(query)
                step_record["observation"] = observation

                # Add to messages
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            else:
                # Invalid action
                step_record["observation"] = f"Invalid action: {action}. Use Search[query] or Finish[answer]."
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": step_record["observation"]})

            trace.append(step_record)

        else:
            # Max steps reached
            answer = "Unable to find answer within step limit."

        elapsed_time = time.time() - start_time

        return AgentResult(
            answer=answer,
            trace=trace,
            steps=len(trace),
            search_calls=self._count_search_calls(trace),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_time=elapsed_time,
            agent_type=self.agent_type,
        )

    def _parse_response(self, content: str) -> tuple:
        """Parse thought and action from response"""
        thought = ""
        action = ""

        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", content, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract action
        action_match = re.search(r"Action:\s*(.+?)(?=Thought:|Observation:|$)", content, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()

        return thought, action

    def _extract_query(self, action: str) -> str:
        """Extract query from Search[query] action"""
        match = re.search(r"Search\[(.+?)\]", action, re.IGNORECASE)
        if match:
            return match.group(1)
        return action.replace("Search", "").strip("[]() ")

    def _extract_answer(self, action: str) -> str:
        """Extract answer from Finish[answer] action"""
        match = re.search(r"Finish\[(.+?)\]", action, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
        return action.replace("Finish", "").strip("[]() ")
