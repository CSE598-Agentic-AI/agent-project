# Copyright Sierra
# Stateful Instruction Memory: stores the original user request and provides a memory block
# to inject into every prompt to prevent goal drift (User Instruction Hallucination).


class InstructionVault:
    """Stores the original user instruction and formats it for injection into prompts."""

    MEMORY_TAG_OPEN = "<memory>"
    MEMORY_TAG_CLOSE = "</memory>"
    LABEL = "Original user request (source of truth):"

    def __init__(self) -> None:
        self._instruction: str | None = None

    def store(self, initial_observation: str) -> None:
        """Store the initial user message / observation as the source of truth."""
        self._instruction = initial_observation.strip()

    @property
    def instruction(self) -> str:
        if self._instruction is None:
            return ""
        return self._instruction

    def get_memory_block(self) -> str:
        """Format the stored instruction for injection at top or bottom of context (primacy/recency)."""
        if not self._instruction:
            return ""
        return (
            f"{self.MEMORY_TAG_OPEN}\n"
            f"{self.LABEL}\n"
            f"{self._instruction}\n"
            f"{self.MEMORY_TAG_CLOSE}"
        )

    def clear(self) -> None:
        """Clear stored instruction (e.g. when starting a new task)."""
        self._instruction = None
