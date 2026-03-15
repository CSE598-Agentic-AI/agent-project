# Copyright Sierra
# FACT (Follow-up Question ACTing) Agent: detects contradictions between new user input
# and the original instruction, and generates context-aware clarification questions.
# Addresses User Instruction Hallucination (PDF: Discrepancy Detection Layer).

import os
from typing import Optional, Tuple

FACT_SYSTEM = """You are a FACT (Follow-up Question ACTing) agent. Your job is to detect when a user's NEW message contradicts or conflicts with their ORIGINAL request, and to generate a helpful clarification question.

Contradictions include:
- Data conflicts: e.g. original asked for "refund 3 items" but now says "refund 2"
- Scope changes: e.g. original was about order A, new message references order B without clarification
- Conflicting instructions: e.g. original said "use credit card" but new message says "use gift card" for the same transaction
- Numeric/value mismatches: quantities, IDs, or amounts that disagree with the original request

You must respond in this exact format:
- First line: either "NO_CONFLICT" or "CONFLICT"
- If CONFLICT: on the next line, provide "CLARIFICATION: " followed by a single, context-aware clarification question that:
  1. Specifically identifies the conflict (what differs between original and new message)
  2. Asks the user to confirm which is correct, or to clarify their intent
  3. Offers disambiguation options when multiple interpretations are plausible (e.g. "Did you mean X or Y?")
  4. Is concise and natural—something a helpful agent would actually say

Do NOT flag minor clarifications, confirmations, or complementary information as conflicts. Only flag when the new message would lead to incorrect tool use or violate the original task."""


class FACTAgent:
    """
    LLM-based FACT agent. Compares new user input against the original instruction (from
    Instruction Vault) and, when a contradiction is detected, generates a context-aware
    clarification question instead of a generic prompt.
    """

    def __init__(
        self,
        model: str = "local",
        provider: str = "openai",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def check(
        self, original_instruction: str, new_user_message: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Compare new user message against the original instruction.
        API base is read from FACT_API_BASE or OPENAI_API_BASE (assistant base) at call time.
        Returns (has_contradiction, clarification_question_or_none).
        If no contradiction, returns (False, None).
        If contradiction, returns (True, "<generated clarification question>").
        """
        if not original_instruction.strip():
            return False, None

        api_base = os.getenv("FACT_API_BASE") or os.getenv("OPENAI_API_BASE")

        user_content = f"""Original user request (source of truth):
{original_instruction}

New user message:
{new_user_message}

Does the new message contradict or conflict with the original request in a way that would require clarification before proceeding? Respond with NO_CONFLICT or CONFLICT, and if CONFLICT provide CLARIFICATION: <your question>."""

        messages = [
            {"role": "system", "content": FACT_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        try:
            from litellm import completion

            kwargs = {
                "model": self.model,
                "messages": messages,
                "custom_llm_provider": self.provider,
                "temperature": self.temperature,
            }
            if api_base:
                kwargs["api_base"] = api_base.rstrip("/")
            res = completion(**kwargs)
            reply = (res.choices[0].message.content or "").strip()
        except Exception:
            return False, None

        first_line = reply.split("\n")[0].strip().upper()
        if first_line == "CONFLICT" or (
            first_line.startswith("CONFLICT") and "NO_CONFLICT" not in first_line
        ):
            clarification = None
            if "CLARIFICATION:" in reply:
                idx = reply.find("CLARIFICATION:")
                clarification = reply[idx + len("CLARIFICATION:") :].strip()
                clarification = clarification.split("\n")[0].strip()
            if not clarification:
                clarification = (
                    "I notice a possible conflict with your original request. "
                    "Could you please clarify what you'd like me to do?"
                )
            return True, clarification
        return False, None
