# Copyright Sierra
# Multi-agent wrapper: Instruction Vault, Policy Sentinel, and Task Verifier around an inner agent.

from typing import Any, Dict, List, Optional, Union

from tau_bench.agents.base import Agent
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs.base import Env
from tau_bench.multi_agent.instruction_vault import InstructionVault
from tau_bench.multi_agent.sentinel import PolicySentinel, LLMSentinel
from tau_bench.multi_agent.verifier import TaskVerifier
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


def _env_name(env: Env) -> str:
    """Infer domain name from env type for sentinel rules."""
    name = type(env).__name__.lower()
    if "retail" in name:
        return "retail"
    if "airline" in name:
        return "airline"
    return "retail"


class MultiAgentAgent(Agent):
    """
    Wraps a ToolCallingAgent with:
    - Instruction Vault: injects original user request into system prompt to reduce goal drift.
    - Policy Sentinel: validates tool calls against domain rules and env state before execution.
    - Task Verifier: optional check before final respond to reduce premature termination.
    """

    def __init__(
        self,
        inner_agent: ToolCallingAgent,
        use_sentinel: bool = True,
        use_verifier: bool = True,
        sentinel_api_base: Optional[str] = None,
        sentinel_model: Optional[str] = None,
        sentinel_provider: str = "openai",
    ) -> None:
        if not isinstance(inner_agent, ToolCallingAgent):
            raise TypeError("MultiAgentAgent requires a ToolCallingAgent as inner_agent")
        self.inner = inner_agent
        self.use_sentinel = use_sentinel
        self.use_verifier = use_verifier
        self.vault = InstructionVault()
        self.sentinel_api_base = sentinel_api_base
        self.sentinel_model = sentinel_model or "local"
        self.sentinel_provider = sentinel_provider
        self.sentinel: Optional[Union[PolicySentinel, LLMSentinel]] = None  # set in solve()
        self.verifier = TaskVerifier()

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        self.vault.clear()
        if self.use_sentinel:
            if self.sentinel_api_base:
                self.sentinel = LLMSentinel(
                    rules=env.rules,
                    api_base=self.sentinel_api_base,
                    model=self.sentinel_model,
                    provider=self.sentinel_provider,
                )
            else:
                self.sentinel = PolicySentinel(env.rules)
        else:
            self.sentinel = None
        self.verifier.set_task(env.task) if self.use_verifier else None

        env_reset = env.reset(task_index=task_index)
        obs = env_reset.observation
        info = env_reset.info.model_dump()
        self.vault.store(obs)

        memory_block = self.vault.get_memory_block()
        system_content = self.inner.get_system_content(extra=memory_block)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": obs},
        ]
        reward = 0.0
        total_cost = 0.0
        env_name = _env_name(env)
        actions_taken: List[Action] = []

        for _ in range(max_num_steps):
            next_message, action, cost = self.inner.generate_next_step(messages)
            total_cost += cost

            if action.name == RESPOND_ACTION_NAME:
                if self.use_verifier:
                    allow, msg = self.verifier.should_allow_respond(
                        action, messages, actions_taken
                    )
                    if not allow:
                        messages.append(
                            {"role": "user", "content": f"API output: {msg}"}
                        )
                        continue
                env_response = env.step(action)
            else:
                if self.use_sentinel and self.sentinel:
                    allowed, correction = self.sentinel.check(
                        action, env.data, env_name=env_name
                    )
                    if not allowed:
                        messages.extend(
                            [
                                next_message,
                                {
                                    "role": "user",
                                    "content": f"API output: {correction}",
                                },
                            ]
                        )
                        continue
                env_response = env.step(action)
                actions_taken.append(action)

            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = (next_message.get("tool_calls") or [])[:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )

            if env_response.done:
                break

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
