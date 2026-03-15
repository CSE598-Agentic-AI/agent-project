# Copyright Sierra
# Multi-agent wrapper: Instruction Vault, Policy Sentinel, and Task Verifier around an inner agent.

import os
from typing import Any, Dict, List, Optional, Union

from tau_bench.agents.base import Agent
from tau_bench.agents.inner_agent_protocol import InnerAgentForMultiAgent
from tau_bench.envs.base import Env
from tau_bench.multi_agent.fact_agent import FACTAgent
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
    Wraps any inner agent (tool-calling, act, react, few-shot) with:
    - Instruction Vault: injects original user request into system prompt to reduce goal drift.
    - Policy Sentinel: validates tool calls against domain rules and env state before execution.
    - FACT Agent: detects user instruction contradictions and generates clarification questions.
    - Task Verifier: optional check before final respond to reduce premature termination.
    """

    def __init__(
        self,
        inner_agent: InnerAgentForMultiAgent,
        use_sentinel: bool = True,
        use_verifier: bool = True,
        use_fact: bool = True,
        sentinel_model: Optional[str] = None,
        sentinel_provider: str = "openai",
        fact_model: Optional[str] = None,
        fact_provider: str = "openai",
    ) -> None:
        for method in ("get_system_content", "generate_next_step", "get_tool_result_message"):
            if not hasattr(inner_agent, method) or not callable(getattr(inner_agent, method)):
                raise TypeError(
                    f"MultiAgentAgent requires inner_agent to implement {method}"
                )
        self.inner = inner_agent
        self.use_sentinel = use_sentinel
        self.use_verifier = use_verifier
        self.use_fact = use_fact
        self.vault = InstructionVault()
        self.sentinel_model = sentinel_model or "local"
        self.sentinel_provider = sentinel_provider
        self.fact_model = fact_model or "local"
        self.fact_provider = fact_provider
        self.sentinel: Optional[Union[PolicySentinel, LLMSentinel]] = None  # set in solve()
        self.verifier = TaskVerifier()
        self.fact_agent: Optional[FACTAgent] = None  # set in solve() when use_fact

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        self.vault.clear()
        # Sentinel: LLM if SENTINEL_MODEL_API_BASE or OPENAI_API_BASE set (same pattern as user model)
        if self.use_sentinel:
            if os.getenv("SENTINEL_MODEL_API_BASE") or os.getenv("OPENAI_API_BASE"):
                self.sentinel = LLMSentinel(
                    rules=env.rules,
                    model=self.sentinel_model,
                    provider=self.sentinel_provider,
                )
            else:
                self.sentinel = PolicySentinel(env.rules)
        else:
            self.sentinel = None
        if self.use_fact:
            self.fact_agent = FACTAgent(
                model=self.fact_model,
                provider=self.fact_provider,
            )
        else:
            self.fact_agent = None
        self.verifier.set_task(env.task) if self.use_verifier else None

        if hasattr(self.inner, "prepare") and callable(getattr(self.inner, "prepare")):
            self.inner.prepare(env, task_index)

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
                if next_message.get("tool_calls"):
                    next_message["tool_calls"] = next_message["tool_calls"][:1]
                tool_msgs = self.inner.get_tool_result_message(
                    env_response.observation, next_message
                )
                messages.extend([next_message] + tool_msgs)
            else:
                new_user_msg = env_response.observation
                if self.fact_agent and self.vault.instruction:
                    has_conflict, clarification = self.fact_agent.check(
                        self.vault.instruction, new_user_msg
                    )
                    if has_conflict and clarification:
                        new_user_msg = (
                            f"{new_user_msg}\n\n"
                            "[FACT: Data conflict detected with the original request. "
                            "Before calling any tools, you MUST ask the user for clarification. "
                            "Do NOT proceed with tool calls. "
                            f"Suggested clarification: {clarification}]"
                        )
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": new_user_msg},
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
