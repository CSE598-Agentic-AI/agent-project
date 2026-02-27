# Copyright Sierra
# Task verification: milestone/outcome checks to catch premature termination and agent hallucination.
# Optional layer that can block final respond until key outcomes are verified.

from typing import Any, Dict, List, Optional, Tuple

from tau_bench.types import Action, RESPOND_ACTION_NAME, Task


class TaskVerifier:
    """Checks that required milestones or outputs are present before accepting a final response."""

    def __init__(self, task: Optional[Task] = None) -> None:
        self.task = task

    def set_task(self, task: Task) -> None:
        self.task = task

    def should_allow_respond(
        self,
        action: Action,
        messages: List[Dict[str, Any]],
        env_actions_taken: List[Action],
    ) -> Tuple[bool, str]:
        """
        Before the agent sends a final respond, optionally verify that the task is actually done.
        Returns (True, "") to allow respond, (False, "Verification: ...") to ask for more work.
        """
        if action.name != RESPOND_ACTION_NAME:
            return True, ""

        if not self.task:
            return True, ""

        # Optional: require that if task has non-respond actions, we've taken at least some of them
        task_tool_actions = [
            a for a in self.task.actions
            if a.name != RESPOND_ACTION_NAME
        ]
        if task_tool_actions and not env_actions_taken:
            return (
                False,
                "Verification: The task requires at least one tool call before responding. Please use the available tools first.",
            )

        return True, ""
