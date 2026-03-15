# Multi-Agent Framework (tau-bench)

This package implements the **Multi-Agent System Framework** proposed in the project document to address the four error categories identified in MAST/tau-bench error analysis.

## Components

### 1. Instruction Vault (`instruction_vault.py`)
- **Problem**: User Instruction Hallucination / Goal Drift — the agent drifts from the original user request over long conversations.
- **Solution**: Stateful Instruction Memory. The original user observation is stored and injected into every prompt inside `<memory>...</memory>` (placed in the system prompt for primacy/recency).

### 2. Policy Sentinel (`sentinel.py`)
- **Problem**: Domain Policy Violations and Contextual Misinterpretation — e.g. calling `exchange_delivered_order_items` on a pending order or `modify_pending_order_items` on a delivered order.
- **Solution**: A Reviewer layer that runs **before** executing a tool call. It checks the proposed action against the current environment state (e.g. order status from `env.data`) and domain rules. If the action violates policy, the call is blocked and a correction message is sent back to the agent.
- **Modes**:
  - **Rule-based** (`PolicySentinel`): Hand-coded checks when `SENTINEL_MODEL_API_BASE` and `OPENAI_API_BASE` are unset.
  - **LLM-based** (`LLMSentinel`): Uses `SENTINEL_MODEL_API_BASE` or `OPENAI_API_BASE` from the environment (same pattern as user model). With `tau-experiment.sh`, the assistant server sets `OPENAI_API_BASE`, so no separate Sentinel server is needed.

### 3. FACT Agent (`fact_agent.py`)
- **Problem**: User Instruction Hallucination — the simulated user (or real user) sends a follow-up message that contradicts the original request (e.g., originally asked for "refund 3 items" but now says "refund 2").
- **Solution**: Before the assistant processes a new user message, the FACT (Follow-up Question ACTing) agent compares it against the Instruction Vault. If a contradiction is detected, it uses an LLM to generate a **context-aware clarification question** and injects it so the assistant must ask for clarification before calling any tools.
- **Enable**: Uses `FACT_API_BASE` or `OPENAI_API_BASE` from the environment (same pattern as user model). Use `--no-fact` to disable.

### 4. Task Verifier (`verifier.py`)
- **Problem**: Agent Hallucination / Premature Termination — the agent claims the task is done without having made required tool calls.
- **Solution**: Before accepting a final `respond` action, the verifier can require that at least one tool call was made when the task requires tool use, reducing “hallucinated progress.”

### 5. Multi-Agent Wrapper (`multi_agent_agent.py`)
- Wraps **all** base strategies (tool-calling, act, react, few-shot). The `agent_strategy` selects the inner agent; the multi-agent layers (Vault, Sentinel, FACT, Verifier) are always applied.
- Composes the inner agent with:
  - Instruction Vault: system prompt includes the memory block.
  - Policy Sentinel: every non-respond tool call is checked; if blocked, the correction is fed back as a synthetic user message and the loop continues without executing the call.
  - FACT Agent: when a new user message arrives (after a respond), compares it to the Instruction Vault; if contradiction detected, injects an LLM-generated clarification question so the assistant asks the user before proceeding.
  - Task Verifier: optional check before final respond.

## Inner-Agent Protocol

To be wrapped by `MultiAgentAgent`, an inner agent must implement:
- `get_system_content(extra)` — system prompt; wrapper appends Instruction Vault here
- `generate_next_step(messages)` — returns `(message_dict, action, cost)`
- `get_tool_result_message(observation, next_message)` — returns message(s) to append after a tool call (format varies by strategy: OpenAI tool-role vs ReAct "API output:" user message)
- `prepare(env, task_index)` *(optional)* — for agents that need setup (e.g. few-shot sampling)

## Usage

All strategies use the multi-agent framework. Select the inner strategy with `--agent-strategy`:

```bash
python run.py --agent-strategy tool-calling --env retail ...
python run.py --agent-strategy act --env retail ...
python run.py --agent-strategy react --env retail ...
python run.py --agent-strategy few-shot --env retail --few-shot-displays-path ... ...
python run.py --agent-strategy multi-agent --env retail ...   # alias for tool-calling
```

In code, `tau_bench.run.agent_factory` builds the inner agent (tool-calling, act, react, or few-shot) and wraps it with `MultiAgentAgent`.
