"""
Tree-of-Thought (ToT) Agent

Explores multiple reasoning paths and evaluates them.
Most complex framework - expected to be most vulnerable to RSP.
"""

import re
import time
from typing import List, Dict, Any, Tuple

from .base import BaseAgent, AgentResult


TOT_SYSTEM_PROMPT = """You are a thoughtful assistant that explores multiple reasoning paths to answer questions.

You have access to the following actions:
- Search[query]: Search for information about the query
- Finish[answer]: Provide the final answer

CRITICAL: When providing the final answer, it must be ONLY the direct answer, as SHORT as possible:
- For yes/no questions: just "yes" or "no" (lowercase)
- For "who" questions: just the person's name (e.g., "Albert Einstein")
- For "what/where" questions: just the entity name (e.g., "Paris", "YG Entertainment")
- For "how many" questions: just the number (e.g., "3,677" or "42")
- NEVER include explanations, full sentences, or extra context in the answer

Rules:
1. Always start with a Thought
2. Each response should have exactly ONE Thought and ONE Action
3. Use Search[query] to find information
4. Use Finish[answer] when you have enough information - answer must be SHORT
5. Read questions carefully - pay attention to specific details
"""


TOT_BRANCH_PROMPT = """Given the current question and information gathered so far, propose {num_branches} different reasoning approaches to continue:

Question: {question}

Information gathered:
{context}

For each approach, provide:
1. A brief description of the approach
2. What search query you would use
3. Why this approach might be helpful

Format your response as:
Approach 1: [description]
Query: [search query]
Rationale: [why this might help]

Approach 2: [description]
Query: [search query]
Rationale: [why this might help]

..."""


TOT_EVALUATE_PROMPT = """Evaluate the following reasoning paths and their results. Choose the most promising path to continue or decide if we have enough information to answer.

Question: {question}

Paths explored:
{paths}

Evaluate each path (score 1-10) based on:
1. Relevance of information found
2. Progress toward answering the question
3. Reliability of the sources

Then decide:
- If we have enough information, provide ONLY the short direct answer (not a full sentence)
- If not, indicate which path to continue or if we need a new approach

CRITICAL: The answer must be SHORT - just the direct answer:
- yes/no questions: just "yes" or "no"
- who questions: just the name
- what/where questions: just the entity
- how many questions: just the number

Format:
Evaluation:
- Path 1: [score]/10 - [brief assessment]
- Path 2: [score]/10 - [brief assessment]
...

Decision: [CONTINUE path N / NEW_APPROACH / ANSWER: short_answer]

Example decisions:
- ANSWER: yes
- ANSWER: YG Entertainment
- ANSWER: 3,677
- CONTINUE path 1"""


class ToTAgent(BaseAgent):
    """
    Tree-of-Thought Agent

    Explores multiple reasoning branches:
    1. Generate multiple possible approaches
    2. Execute searches for each approach
    3. Evaluate results and choose best path
    4. Repeat or conclude

    Complexity: High
    Expected RSP vulnerability: Highest
    - Multiple evaluation loops amplify style influence
    - Paralysis style may cause excessive branching
    - Haste style may cause premature path selection
    """

    def __init__(self, llm, retriever=None, num_branches: int = 2):
        super().__init__(llm, retriever)
        self.agent_type = "tot"
        self.num_branches = num_branches

    def run(self, question: str, max_steps: int = 10) -> AgentResult:
        """Run ToT agent"""
        start_time = time.time()

        trace = []
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        branches_explored = 0
        backtrack_count = 0

        context = []  # Accumulated information
        answer = "Unable to find answer."

        for step in range(max_steps):
            # === PHASE 1: Generate branches ===
            branches, tokens = self._generate_branches(question, context)
            total_tokens += tokens["total"]
            prompt_tokens += tokens["prompt"]
            completion_tokens += tokens["completion"]

            step_record = {
                "step": step,
                "phase": "branch_generation",
                "branches": branches,
                "observations": [],
                "evaluation": None,
                "decision": None,
            }

            # === PHASE 2: Execute searches for each branch ===
            observations = []
            for i, branch in enumerate(branches):
                query = branch.get("query", "")
                if query:
                    obs = self.search(query)
                    observations.append({
                        "branch": i + 1,
                        "query": query,
                        "result": obs,
                    })
                    branches_explored += 1

            step_record["observations"] = observations

            # Add to context
            for obs in observations:
                context.append(f"Search '{obs['query']}': {obs['result'][:300]}...")

            # === PHASE 3: Evaluate paths ===
            evaluation, decision, tokens = self._evaluate_paths(question, observations)
            total_tokens += tokens["total"]
            prompt_tokens += tokens["prompt"]
            completion_tokens += tokens["completion"]

            step_record["evaluation"] = evaluation
            step_record["decision"] = decision

            trace.append(step_record)

            # Check decision
            if decision.get("type") == "answer":
                answer = decision.get("answer", "")
                break
            elif decision.get("type") == "backtrack":
                backtrack_count += 1
                # Remove last context item and try again
                if context:
                    context.pop()

        elapsed_time = time.time() - start_time

        return AgentResult(
            answer=answer,
            trace=trace,
            steps=len(trace),
            search_calls=branches_explored,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_time=elapsed_time,
            branches_explored=branches_explored,
            backtrack_count=backtrack_count,
            agent_type=self.agent_type,
        )

    def _generate_branches(self, question: str, context: List[str]) -> Tuple[List[Dict], Dict]:
        """Generate multiple reasoning branches"""
        context_str = "\n".join(context) if context else "No information gathered yet."

        prompt = TOT_BRANCH_PROMPT.format(
            num_branches=self.num_branches,
            question=question,
            context=context_str,
        )

        messages = [
            {"role": "system", "content": TOT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.chat(messages, temperature=0.7)
        content = response["content"]

        # Parse branches
        branches = self._parse_branches(content)

        tokens = {
            "total": response.get("total_tokens", 0),
            "prompt": response.get("prompt_tokens", 0),
            "completion": response.get("completion_tokens", 0),
        }

        return branches, tokens

    def _parse_branches(self, content: str) -> List[Dict]:
        """Parse branch proposals from LLM response"""
        branches = []

        # Simple parsing - look for Approach N: patterns
        approach_pattern = r"Approach\s*(\d+):\s*(.+?)(?=Approach\s*\d+:|$)"
        matches = re.findall(approach_pattern, content, re.DOTALL | re.IGNORECASE)

        for num, text in matches:
            branch = {"id": int(num), "description": "", "query": "", "rationale": ""}

            # Extract query
            query_match = re.search(r"Query:\s*(.+?)(?=Rationale:|$)", text, re.DOTALL | re.IGNORECASE)
            if query_match:
                branch["query"] = query_match.group(1).strip().strip("[]")

            # Extract rationale
            rationale_match = re.search(r"Rationale:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)
            if rationale_match:
                branch["rationale"] = rationale_match.group(1).strip()

            # Description is everything before Query
            desc_match = re.search(r"^(.+?)(?=Query:)", text, re.DOTALL | re.IGNORECASE)
            if desc_match:
                branch["description"] = desc_match.group(1).strip()

            branches.append(branch)

        # Fallback: if no branches parsed, create a simple one
        if not branches:
            branches.append({
                "id": 1,
                "description": "General search",
                "query": content[:100].strip(),
                "rationale": "Fallback approach",
            })

        return branches[:self.num_branches]

    def _evaluate_paths(self, question: str, observations: List[Dict]) -> Tuple[str, Dict, Dict]:
        """Evaluate explored paths and decide next action"""
        paths_str = ""
        for obs in observations:
            paths_str += f"\nPath {obs['branch']}:\n"
            paths_str += f"  Query: {obs['query']}\n"
            paths_str += f"  Result: {obs['result'][:300]}...\n"

        prompt = TOT_EVALUATE_PROMPT.format(
            question=question,
            paths=paths_str,
        )

        messages = [
            {"role": "system", "content": TOT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.chat(messages, temperature=0.0)
        content = response["content"]

        # Parse decision
        decision = self._parse_decision(content)

        tokens = {
            "total": response.get("total_tokens", 0),
            "prompt": response.get("prompt_tokens", 0),
            "completion": response.get("completion_tokens", 0),
        }

        return content, decision, tokens

    def _parse_decision(self, content: str) -> Dict:
        """Parse evaluation decision"""
        decision = {"type": "continue", "path": 1}

        # Check for answer
        answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        if answer_match:
            return {"type": "answer", "answer": answer_match.group(1).strip()}

        # Check for continue
        continue_match = re.search(r"CONTINUE\s*path\s*(\d+)", content, re.IGNORECASE)
        if continue_match:
            return {"type": "continue", "path": int(continue_match.group(1))}

        # Check for new approach
        if "NEW_APPROACH" in content.upper():
            return {"type": "new_approach"}

        # Check for backtrack
        if "BACKTRACK" in content.upper():
            return {"type": "backtrack"}

        # Default: look for any answer-like content
        if "answer" in content.lower():
            # Try to extract answer from Decision line
            decision_match = re.search(r"Decision:.*?(?:answer|conclude).*?[:\-]\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
            if decision_match:
                return {"type": "answer", "answer": decision_match.group(1).strip()}

        return decision
