# Copyright Sierra
# Protocol for inner agents that can be wrapped by MultiAgentAgent.

from typing import Any, Dict, List, Optional, Protocol, Tuple

from tau_bench.types import Action


class InnerAgentForMultiAgent(Protocol):
    """
    Protocol for agents that MultiAgentAgent can wrap.
    Inner agents must implement get_system_content, generate_next_step, and get_tool_result_message.
    Optional: prepare(env, task_index) for agents that need setup (e.g. few-shot sampling).
    """

    def get_system_content(self, extra: Optional[str] = None) -> str:
        """Return system prompt content. Wrapper appends Instruction Vault `extra` here."""
        ...

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        """Produce next assistant message and parsed action. Returns (message_dict, action, cost)."""
        ...

    def get_tool_result_message(
        self, observation: str, next_message: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Return message(s) to append after a tool call.
        Tool-calling/FewShot: [{"role": "tool", "tool_call_id", "name", "content"}]
        ReAct: [{"role": "user", "content": "API output: " + observation}]
        """
        ...
