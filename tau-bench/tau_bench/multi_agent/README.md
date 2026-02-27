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
  - **Rule-based** (`PolicySentinel`): Hand-coded checks when `SENTINEL_MODEL_API_BASE` is unset.
  - **LLM-based** (`LLMSentinel`): Set `SENTINEL_MODEL_API_BASE` or `--sentinel-api-base` to enable. When using `tau-experiment.sh` with multi-agent, this defaults to the Assistant server (same model), so no separate Sentinel server is needed.

### 3. Task Verifier (`verifier.py`)
- **Problem**: Agent Hallucination / Premature Termination — the agent claims the task is done without having made required tool calls.
- **Solution**: Before accepting a final `respond` action, the verifier can require that at least one tool call was made when the task requires tool use, reducing “hallucinated progress.”

### 4. Multi-Agent Wrapper (`multi_agent_agent.py`)
- Composes the inner agent (e.g. `ToolCallingAgent`) with:
  - Instruction Vault: system prompt includes the memory block.
  - Policy Sentinel: every non-respond tool call is checked; if blocked, the correction is fed back as a synthetic user message and the loop continues without executing the call.
  - Task Verifier: optional check before final respond.

## Usage

Use the `multi-agent` agent strategy when running the benchmark:

```bash
# Example: run with multi-agent strategy (same CLI as other strategies)
python -m tau_bench.run --agent_strategy multi-agent --env retail ...
```

In code, the strategy is registered in `tau_bench.run.agent_factory` and builds a `ToolCallingAgent` wrapped with `MultiAgentAgent`.
