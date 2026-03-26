# Copyright Sierra

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)

MAX_CRITIC_RETRIES = 2
READ_ONLY_PREFIXES = ("get_", "list_", "calculate", "think")

# Environment-changing tools: run LLM critic after deterministic validation passes.
HIGH_STAKES_ACTION_NAMES = frozenset(
    {
        # Airline
        "book_reservation",
        "cancel_reservation",
        "update_reservation_flights",
        "update_reservation_passengers",
        "update_reservation_baggages",
        "send_certificate",
        "transfer_to_human_agents",
        # Retail
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "return_delivered_order_items",
        "modify_user_address",
    }
)

FLIGHT_SEARCH_TOOL_NAMES = frozenset(
    {"search_direct_flight", "search_onestop_flight"}
)
# How far back to look for a flight search before a user-facing respond (think() may sit in between).
CRITIC_FLIGHT_SEARCH_LOOKBACK = 6

# Phrases that indicate the executor is claiming failure after a search — worth critic review.
_IMPOSSIBILITY_PHRASES = (
    "no flights",
    "none of the",
    "couldn't find",
    "could not find",
    "no available",
    "no options",
    "no suitable",
    "no direct flight",
    "no one-stop",
    "unable to find",
    "unfortunately",
    "doesn't meet",
    "do not meet",
    "does not meet",
    "don't meet",
    "not possible",
    "cannot be handled",
)


# ---------------------------------------------------------------------------
# State models (per-run, never stored on self to stay thread-safe)
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    id: str
    description: str
    status: str  # pending | in_progress | done

@dataclass
class Plan:
    goal: str
    steps: List[PlanStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [
                {"id": s.id, "description": s.description, "status": s.status}
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Plan":
        return cls(
            goal=d.get("goal", ""),
            steps=[
                PlanStep(
                    id=s["id"],
                    description=s["description"],
                    status=s.get("status", "pending"),
                )
                for s in d.get("steps", [])
            ],
        )


@dataclass
class CompletedToolCall:
    name: str
    kwargs: Dict[str, Any]
    result_summary: str
    full_result: str = ""


@dataclass
class ConversationState:
    executor_messages: List[Dict[str, Any]] = field(default_factory=list)
    approved_plan: Optional[Plan] = None
    pending_plan_update: Optional[Dict[str, Any]] = None
    active_step_id: Optional[str] = None
    total_cost: float = 0.0
    internal_trace: List[Dict[str, Any]] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    completed_tool_calls: List[CompletedToolCall] = field(default_factory=list)
    original_user_request: str = ""


# ---------------------------------------------------------------------------
# Role prompts
# ---------------------------------------------------------------------------

PLANNER_INSTRUCTION = """You are a planning agent for a customer service system.

# Domain Policy
{wiki}

# Available Tools
{tools}

# Current Approved Plan
{current_plan}

# Pending Plan Update (awaiting user confirmation)
{pending_update}

# Conversation So Far
{conversation}

# Instructions
Analyze the conversation and decide how to proceed with the plan.

You MUST respond with ONLY valid JSON (no markdown, no extra text) in this exact format:
{{
  "decision": "continue_existing_plan" | "propose_plan_change" | "request_clarification",
  "reason": "Brief explanation of your decision",
  "plan": {{
    "goal": "The overall goal based on the user request",
    "steps": [
      {{"id": "s1", "description": "Step description", "status": "pending"}}
    ]
  }},
  "active_step_id": "s1",
  "confirmation_question": null
}}

Rules:
- If there is no existing plan, create one and use "continue_existing_plan".
- If the user's latest message materially changes the goal, constraints, or approach, \
use "propose_plan_change" and set "confirmation_question" to ask for confirmation.
- IMPORTANT: Reducing the scope of the plan (e.g. dropping items the user originally \
asked about, handling fewer requests than originally stated) is a MATERIAL plan change \
that requires "propose_plan_change" with user confirmation. Never silently drop user \
requests from the plan.
- If you just need to progress through existing steps, use "continue_existing_plan" and \
update step statuses accordingly.
- Step status updates (pending -> in_progress -> done) do NOT require user confirmation.
- If a pending plan update exists and the user confirmed it, apply the change and return \
"continue_existing_plan" with the updated plan.
- If a pending plan update exists and the user rejected it, return "continue_existing_plan" \
with the original plan unchanged.
- If you need more information to proceed, use "request_clarification" and set \
"confirmation_question" to your question.
- The "plan" field must ALWAYS be present.
- When the user mentions a fallback preference (e.g. "if X isn't available, I'll take Y"), \
include that fallback in the plan steps so it is not lost.
- MULTIPLE DISTINCT OPERATIONS: Use "request_clarification" only when the user clearly \
asks for two or more unrelated transaction types in one message (e.g. cancel reservation A \
and book an unrelated trip B with no priority). A single booking, change, or cancellation \
with many constraints (times, bags, payment, cabin, certificates) is still ONE coherent \
request — build one plan and execute it; do not ask the user to split "which request first" \
for normal constraint-rich instructions.
"""

EXECUTOR_SYSTEM_TEMPLATE = """{wiki}

# Current Plan
{plan_summary}

# Active Step
{active_step}

# Tool Calls Already Completed
{completed_tools}

CRITICAL RULES:
- NEVER fabricate or guess user data (emails, names, IDs, addresses, order numbers). \
Only use information the user has explicitly provided in the conversation or that was \
returned by a previous tool call. For example, NEVER call find_user_id_by_email with \
an email you made up — ask the user for their email first.
- If you need information the user hasn't provided, ASK them for it instead of guessing.
- Do not repeat tool calls that have already been completed (see above).
- Execute the active step of the plan using the available tools.
- Follow the domain policy strictly.
- If you have enough information to respond to the user, respond directly.
- If you need to gather information or perform an action, use the appropriate tool.
- SCOPE AWARENESS: Pay close attention to EXACTLY which items the user wants to act on. \
If the user says "only exchange item X" or "just modify the lamp", do NOT include other \
items in the tool call. Read the user's most recent message carefully for scope qualifiers \
like "only", "just", "specifically".
- RELATIVE TERMS: When the user asks for something "less X" or "more Y", compare against \
the CURRENT item's attributes. "Less bright" means a brightness level LOWER than what they \
currently have, not the same level.
- MANDATORY AUTHENTICATION: User authentication is REQUIRED by policy for any order-viewing \
or order-modifying action. If the user refuses to provide identifying information (name, \
email, etc.), clearly state that authentication is a system requirement and you cannot \
proceed without it. Be direct — do not keep politely re-asking indefinitely. After 2 clear \
explanations, tell the user you cannot help with this request without authentication.
- RESPECT USER FALLBACK PREFERENCES: When the user states a preference chain like \
"I want X, but if X is unavailable I'll take Y", and you discover X is unavailable, \
apply the fallback Y directly. Do NOT present other alternatives that differ from what \
the user explicitly stated as their fallback.
- CONSTRAINT CHECK BEFORE OFFERS OR BOOKING: When the user states hard limits (e.g. \
"not before 11am", "after 3pm", "same day only", "economy only"), filter flight or itinerary \
options so you NEVER present or book an option that violates those limits. Among options \
that satisfy ALL hard constraints, apply the user's tie-breakers (e.g. lowest price, \
shortest total time). If no option satisfies the constraints, say so and search again or \
ask a minimal clarifying question.
- EXECUTE IMMEDIATELY: When you have decided on an action and have all required information, \
call the tool in THIS response. NEVER write a message that says "I will now…", "I'll proceed \
with…", or "I'm going to…" without calling the tool in the same turn. Describing a future \
action without executing it is a failure — the user's response to that description will be \
"thank you" and the task will end without the action being performed.
- Be decisive and efficient. When you have all information needed, proceed to the action \
rather than asking for one more round of confirmation. ONE confirmation round is sufficient.
- COMBINED CONFIRMATION: When the user in one message both (a) confirms a proposed action \
(e.g. \"Yes, that's correct\", \"Just the desk lamp exchange\") and (b) adds another request \
(e.g. \"And also, I'd like to return the water bottle\"), treat the entire message as \
confirmation for BOTH. Proceed to execute — do NOT ask again \"Would you like me to proceed \
with X and Y?\". The user has already confirmed and stated the full scope.
"""

CRITIC_AGENT_SYSTEM = """You are the **Critic Agent** — an independent reviewer in a \
multi-agent customer-service system. The Executor proposes an action; your job is to \
catch mistakes **before** it reaches the user or the environment.

You do NOT chat with the user. You only output a structured verdict.

## Your mindset
- Treat **Actual Tool Output Data** as ground truth. If the Executor's proposed \
`respond` text contradicts that data, **reject**.
- Be adversarial to lazy claims: e.g. "no flights", "none meet the criteria", \
"couldn't find any options" **must** be checked against the raw search results.
- For **one-stop itineraries**, the user's "departure after/before X" usually applies to \
the **first segment's** `scheduled_departure_time_est` (outbound from the origin airport). \
Parse times as HH:MM:SS in the tool JSON; 19:00:00 is after 11:00:00.
- If **any** itinerary in the tool output satisfies the user's stated hard constraints, \
a `respond` that denies that is **wrong** — reject and tell the Executor to list those \
options (and apply tie-breakers like lowest price if the user asked).

## update_* actions (update_reservation_flights, update_reservation_passengers, etc.)
These actions specify the DESIRED NEW STATE — they will DIFFER from current tool output \
by design. Do NOT reject because "the passenger in the reservation is X but the action \
says Y" or "the flight in the reservation is A but the action proposes B". The entire \
point of an update is to change something. Only reject if: (a) the reservation_id is \
wrong, (b) a hard policy rule is violated (e.g. basic economy can't change flights), or \
(c) the data in the action was never mentioned by the user or in any tool result.

## transfer_to_human_agents
Reject if the only obstacle is that the Executor misread tool output, searched the \
wrong dates, or failed to match an existing reservation that is already in \
**Actual Tool Output Data**. Transfer only when the request is truly out of scope or \
impossible after **correctly** using the data.

## cancel / book / modify
- Reject if IDs or payment methods were never seen in conversation or tool results.
- Reject flight bookings that violate user-stated time windows when compared to the \
proposed flight list in arguments.

## Domain policy
{wiki}

You MUST respond with ONLY valid JSON (no markdown, no extra text):
{{
  "approved": true,
  "reason": "Short justification",
  "feedback_for_executor": null,
  "risk_level": "low",
  "evidence": null
}}

Use "evidence" for a brief quote or pointer into tool output when rejecting (else null). \
If rejecting, "feedback_for_executor" must be concrete: what to list, what to re-read, \
or which tool result contradicts the proposal.
"""


def _parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction: raw → fenced block → first brace pair."""
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = content.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(content[start : i + 1])
                except json.JSONDecodeError:
                    break
    return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MultiAgentV1(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        planner_model: Optional[str] = None,
        planner_provider: Optional[str] = None,
        critic_model: Optional[str] = None,
        critic_provider: Optional[str] = None,
        max_critic_retries: int = MAX_CRITIC_RETRIES,
    ) -> None:
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.planner_model = planner_model or model
        self.planner_provider = planner_provider or provider
        self.critic_model = critic_model or model
        self.critic_provider = critic_provider or provider
        self.max_critic_retries = max_critic_retries
        self._tools_str = json.dumps(tools_info, indent=2)
        self._critic_agent: Optional["CriticAgent"] = None

    def _get_critic(self) -> "CriticAgent":
        if self._critic_agent is None:
            self._critic_agent = CriticAgent(
                wiki=self.wiki,
                model=self.critic_model,
                provider=self.critic_provider,
                temperature=self.temperature,
            )
        return self._critic_agent

    @staticmethod
    def should_run_critic_review(
        action: Action, state: ConversationState
    ) -> bool:
        if action.name in HIGH_STAKES_ACTION_NAMES:
            return True
        # Only run critic on respond if it makes an impossibility claim after a flight search.
        if action.name == RESPOND_ACTION_NAME and state.completed_tool_calls:
            tail = state.completed_tool_calls[-CRITIC_FLIGHT_SEARCH_LOOKBACK:]
            if any(tc.name in FLIGHT_SEARCH_TOOL_NAMES for tc in tail):
                text = (action.kwargs.get(RESPOND_ACTION_FIELD_NAME) or "").lower()
                if any(phrase in text for phrase in _IMPOSSIBILITY_PHRASES):
                    return True
        return False

    # ---- Formatting helpers ----

    @staticmethod
    def _format_plan(plan: Optional[Plan]) -> str:
        if plan is None:
            return "No plan yet."
        markers = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}
        lines = [f"Goal: {plan.goal}"]
        for s in plan.steps:
            lines.append(f"  {markers.get(s.status, '[ ]')} {s.id}: {s.description}")
        return "\n".join(lines)

    @staticmethod
    def _get_active_step_description(
        plan: Optional[Plan], step_id: Optional[str]
    ) -> str:
        if plan is None or step_id is None:
            return "No active step."
        for s in plan.steps:
            if s.id == step_id:
                return f"{s.id}: {s.description} (status: {s.status})"
        return f"Step {step_id} not found in plan."

    @staticmethod
    def _format_conversation_for_planner(
        messages: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            if role == "system":
                continue
            elif role == "tool":
                name = msg.get("name", "tool")
                lines.append(f"[Tool:{name}] {(msg.get('content') or '')[:200]}")
            elif role == "assistant":
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        lines.append(
                            f"[Agent called {fn.get('name', '?')}({fn.get('arguments', '')})]"
                        )
                elif msg.get("content"):
                    lines.append(f"[Agent] {msg['content'][:300]}")
            elif role == "user":
                lines.append(f"[User] {(msg.get('content') or '')[:300]}")
        return "\n".join(lines) if lines else "No conversation yet."

    @staticmethod
    def _format_completed_tools(
        completed: List[CompletedToolCall],
    ) -> str:
        if not completed:
            return "None yet."
        lines: List[str] = []
        for tc in completed:
            args_str = json.dumps(tc.kwargs)[:150]
            lines.append(f"- {tc.name}({args_str}) → {tc.result_summary}")
        return "\n".join(lines)

    @staticmethod
    def _format_tool_output_data(
        completed: List[CompletedToolCall],
        max_per_tool: int = 1500,
        max_total: int = 6000,
    ) -> str:
        """Full tool results for the critic to make factual decisions."""
        if not completed:
            return "No tool data available yet."
        lines: List[str] = []
        total = 0
        for tc in completed:
            result_text = tc.full_result[:max_per_tool]
            entry = f"## {tc.name}({json.dumps(tc.kwargs)[:200]})\n{result_text}"
            if total + len(entry) > max_total:
                lines.append("... (earlier tool outputs truncated for space)")
                break
            lines.append(entry)
            total += len(entry)
        return "\n\n".join(lines)

    # ---- LLM role callers ----

    def _call_planner(self, state: ConversationState) -> Dict[str, Any]:
        pending_str = (
            json.dumps(state.pending_plan_update, indent=2)
            if state.pending_plan_update
            else "None"
        )
        prompt = PLANNER_INSTRUCTION.format(
            wiki=self.wiki,
            tools=self._tools_str,
            current_plan=self._format_plan(state.approved_plan),
            pending_update=pending_str,
            conversation=self._format_conversation_for_planner(
                state.executor_messages
            ),
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Analyze the conversation and return your planning decision as JSON.",
            },
        ]
        api_base = os.getenv("AGENT_MODEL_API_BASE") or os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("AGENT_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY")
        res = completion(
            model=self.planner_model,
            custom_llm_provider=self.planner_provider,
            messages=messages,
            temperature=self.temperature,
            api_base=api_base,
            api_key=api_key,
        )
        cost = res._hidden_params.get("response_cost", 0) or 0
        state.total_cost += cost

        content = res.choices[0].message.content or ""
        parsed = _parse_json_response(content)
        state.internal_trace.append(
            {"role": "planner", "raw_output": content, "parsed": parsed, "cost": cost}
        )

        if parsed is None:
            return {
                "decision": "continue_existing_plan",
                "reason": "Failed to parse planner output, continuing.",
                "plan": (
                    state.approved_plan.to_dict()
                    if state.approved_plan
                    else {"goal": "Help the user", "steps": []}
                ),
                "active_step_id": state.active_step_id,
                "confirmation_question": None,
            }
        return parsed

    def _call_executor(
        self,
        state: ConversationState,
        extra_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Action, float]:
        system_prompt = EXECUTOR_SYSTEM_TEMPLATE.format(
            wiki=self.wiki,
            plan_summary=self._format_plan(state.approved_plan),
            active_step=self._get_active_step_description(
                state.approved_plan, state.active_step_id
            ),
            completed_tools=self._format_completed_tools(state.completed_tool_calls),
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(state.executor_messages[1:])
        if extra_messages:
            messages.extend(extra_messages)

        api_base = os.getenv("AGENT_MODEL_API_BASE") or os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("AGENT_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY")
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            tools=self.tools_info,
            temperature=self.temperature,
            api_base=api_base,
            api_key=api_key,
        )
        cost = res._hidden_params.get("response_cost", 0) or 0
        msg = res.choices[0].message.model_dump()
        action = self._message_to_action(msg)

        state.internal_trace.append(
            {
                "role": "executor",
                "action": {"name": action.name, "kwargs": action.kwargs},
                "cost": cost,
            }
        )
        return msg, action, cost

    # ---- Helpers ----

    @staticmethod
    def _is_rejection(text: str) -> bool:
        """Heuristic: does the user's response reject the proposed plan?"""
        low = text.strip().lower()
        negations = [
            "no ", "no,", "no.", "nope", "nah", "don't", "do not",
            "cancel", "never mind", "nevermind", "not what i",
            "that's not", "that is not", "i didn't", "i did not",
        ]
        return any(n in low for n in negations) or low in ("no", "nope", "nah")

    @staticmethod
    def _extract_user_details(state: "ConversationState") -> Optional[Dict[str, Any]]:
        """Return parsed get_user_details result from the most recent call."""
        for tc in reversed(state.completed_tool_calls):
            if tc.name == "get_user_details" and tc.full_result:
                try:
                    return json.loads(tc.full_result)
                except Exception:
                    pass
        return None

    @staticmethod
    def _compute_payment_split(
        profile: Dict[str, Any],
        total_amount: float,
    ) -> Optional[List[Dict[str, Any]]]:
        """Policy-compliant split: ≤1 certificate, ≤3 gift cards, ≤1 credit card.
        Certificates used first (largest), then gift cards (largest first), then credit card."""
        pms = profile.get("payment_methods", {})
        certificates, gift_cards, credit_cards = [], [], []
        for pid, pm in pms.items():
            src = pm.get("source", "")
            if src == "certificate":
                certificates.append((pm.get("amount", 0), pid))
            elif src == "gift_card":
                gift_cards.append((pm.get("amount", 0), pid))
            elif src == "credit_card":
                credit_cards.append(pid)

        certificates.sort(reverse=True)
        gift_cards.sort(reverse=True)

        result: List[Dict[str, Any]] = []
        remaining = round(float(total_amount), 2)

        if certificates and remaining > 0:
            amt, pid = certificates[0]
            use = round(min(amt, remaining), 2)
            result.append({"payment_id": pid, "amount": use})
            remaining = round(remaining - use, 2)

        for amt, pid in gift_cards[:3]:
            if remaining <= 0:
                break
            use = round(min(amt, remaining), 2)
            if use > 0:
                result.append({"payment_id": pid, "amount": use})
                remaining = round(remaining - use, 2)

        if remaining > 0.01 and credit_cards:
            result.append({"payment_id": credit_cards[0], "amount": round(remaining, 2)})
            remaining = 0.0

        return result if remaining <= 0.01 else None

    @staticmethod
    def _validate_payment_split(
        action: Action, state: "ConversationState"
    ) -> Optional[str]:
        """Returns a correction message if the proposed payment split violates policy,
        else None (no correction needed)."""
        if action.name != "book_reservation":
            return None
        proposed_pms = action.kwargs.get("payment_methods")
        if not proposed_pms or not isinstance(proposed_pms, list):
            return None

        profile = MultiAgentV1._extract_user_details(state)
        if not profile:
            return None

        total = round(sum(pm.get("amount", 0) for pm in proposed_pms), 2)
        if total <= 0:
            return None

        correct = MultiAgentV1._compute_payment_split(profile, total)
        if not correct:
            return None

        def _sig(pms):
            return sorted((pm["payment_id"], pm["amount"]) for pm in pms)

        if _sig(proposed_pms) == _sig(correct):
            return None  # already correct

        correct_lines = "\n".join(
            f"  - {pm['payment_id']}: ${pm['amount']}" for pm in correct
        )
        return (
            f"[PaymentSplitter] The proposed payment split for ${total} is wrong. "
            f"Policy: ≤1 certificate (largest first), ≤3 gift cards (largest first), "
            f"≤1 credit card for remainder.\n"
            f"Correct split:\n{correct_lines}\n"
            "Use EXACTLY these payment_methods in your next booking call."
        )

    @staticmethod
    def _ensure_active_step_in_progress(state: "ConversationState") -> None:
        """Planner often leaves the active step as pending; executor needs in_progress."""
        if not state.approved_plan or not state.active_step_id:
            return
        for s in state.approved_plan.steps:
            if s.id == state.active_step_id and s.status == "pending":
                s.status = "in_progress"
                return

    @staticmethod
    def _validate_action(action: Action, state: "ConversationState") -> Dict[str, Any]:
        """Deterministic validation replacing the LLM critic.
        Checks that referenced IDs actually exist in conversation or tool results."""
        if action.name == RESPOND_ACTION_NAME:
            return {"approved": True, "reason": "Response auto-approved."}

        known_data = ""
        for tc in state.completed_tool_calls:
            known_data += tc.full_result + "\n"
        for msg in state.executor_messages:
            if msg.get("role") in ("user", "tool"):
                known_data += (msg.get("content") or "") + "\n"

        for key in ("user_id", "order_id"):
            if key in action.kwargs:
                val = str(action.kwargs[key])
                if val and val not in known_data:
                    return {
                        "approved": False,
                        "reason": f"{key} '{val}' not in conversation or tool results.",
                        "feedback_for_executor": (
                            f"The {key} '{val}' hasn't appeared in the conversation "
                            f"or any tool result. Please verify it first."
                        ),
                    }

        return {"approved": True, "reason": "Validation passed."}

    @staticmethod
    def _message_to_action(message: Dict[str, Any]) -> Action:
        if (
            message.get("tool_calls")
            and len(message["tool_calls"]) > 0
            and message["tool_calls"][0].get("function") is not None
        ):
            tc = message["tool_calls"][0]
            return Action(
                name=tc["function"]["name"],
                kwargs=json.loads(tc["function"]["arguments"]),
            )
        return Action(
            name=RESPOND_ACTION_NAME,
            kwargs={RESPOND_ACTION_FIELD_NAME: message.get("content", "")},
        )

    # ---- Orchestrated solve loop ----

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 40,
    ) -> SolveResult:
        reset_resp = env.reset(task_index=task_index)
        state = ConversationState()
        state.info = (
            reset_resp.info.model_dump()
            if hasattr(reset_resp.info, "model_dump")
            else {}
        )
        state.executor_messages = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": reset_resp.observation},
        ]
        state.original_user_request = reset_resp.observation

        last_source = "user"
        plan_change_proposals = 0

        for _ in range(max_num_steps):
            # ---------- AUTO-APPLY PENDING PLAN CONFIRMATION ----------
            if last_source == "user" and state.pending_plan_update:
                user_response = (
                    state.executor_messages[-1].get("content", "")
                    if state.executor_messages
                    else ""
                )
                proposed = state.pending_plan_update.get("proposed_plan")
                if not self._is_rejection(user_response) and proposed and isinstance(proposed, dict):
                    state.approved_plan = Plan.from_dict(proposed)
                state.pending_plan_update = None

            # ---------- PLANNER PHASE ----------
            # Run after user messages and after tool observations so the plan/active step
            # stay aligned during multi-tool stretches (no silent drift).
            if last_source in ("user", "tool"):
                planner_result = self._call_planner(state)
                decision = planner_result.get("decision", "continue_existing_plan")
                plan_data = planner_result.get("plan")

                if last_source == "tool" and decision in (
                    "request_clarification",
                    "propose_plan_change",
                ):
                    decision = "continue_existing_plan"

                if decision == "propose_plan_change" and last_source == "user":
                    plan_change_proposals += 1
                    if plan_change_proposals > 2:
                        if plan_data and isinstance(plan_data, dict):
                            state.approved_plan = Plan.from_dict(plan_data)
                        state.pending_plan_update = None
                        state.active_step_id = planner_result.get(
                            "active_step_id", state.active_step_id
                        )
                    else:
                        state.pending_plan_update = {
                            "proposed_plan": plan_data,
                            "reason": planner_result.get("reason", ""),
                        }
                        question = planner_result.get(
                            "confirmation_question",
                            "Could you confirm you'd like me to proceed with this updated plan?",
                        )
                        env_resp = env.step(
                            Action(
                                name=RESPOND_ACTION_NAME,
                                kwargs={RESPOND_ACTION_FIELD_NAME: question},
                            )
                        )
                        state.reward = env_resp.reward
                        state.info = {**state.info, **env_resp.info.model_dump()}
                        state.executor_messages.extend(
                            [
                                {"role": "assistant", "content": question},
                                {"role": "user", "content": env_resp.observation},
                            ]
                        )
                        last_source = "user"
                        if env_resp.done:
                            break
                        continue
                else:
                    plan_change_proposals = 0

                if decision == "request_clarification" and last_source == "user":
                    question = planner_result.get(
                        "confirmation_question",
                        "Could you provide more details about your request?",
                    )
                    env_resp = env.step(
                        Action(
                            name=RESPOND_ACTION_NAME,
                            kwargs={RESPOND_ACTION_FIELD_NAME: question},
                        )
                    )
                    state.reward = env_resp.reward
                    state.info = {**state.info, **env_resp.info.model_dump()}
                    state.executor_messages.extend(
                        [
                            {"role": "assistant", "content": question},
                            {"role": "user", "content": env_resp.observation},
                        ]
                    )
                    last_source = "user"
                    if env_resp.done:
                        break
                    continue

                # continue_existing_plan
                if plan_data and isinstance(plan_data, dict):
                    state.approved_plan = Plan.from_dict(plan_data)
                state.pending_plan_update = None
                state.active_step_id = planner_result.get(
                    "active_step_id", state.active_step_id
                )

                self._ensure_active_step_in_progress(state)

            # ---------- EXECUTOR + VALIDATION PHASE ----------
            action: Optional[Action] = None
            executor_msg: Optional[Dict[str, Any]] = None
            # retry_context accumulates ALL prior feedback so the critic/executor see their
            # own history and cannot contradict themselves or loop on the same objection.
            retry_context: List[Dict[str, Any]] = []
            critic_rejection_history: List[str] = []

            for attempt in range(self.max_critic_retries + 1):
                executor_msg, action, cost = self._call_executor(
                    state, extra_messages=retry_context or None
                )
                state.total_cost += cost

                validation = self._validate_action(action, state)
                if not validation.get("approved", True):
                    feedback = validation.get(
                        "feedback_for_executor",
                        "Please reconsider your action.",
                    )
                    retry_context.append(
                        {
                            "role": "user",
                            "content": (
                                f"[Validator attempt {attempt + 1}] Rejected: "
                                f"{feedback}. Try a different approach."
                            ),
                        }
                    )
                    continue

                # Deterministic payment correction (no LLM needed)
                payment_correction = self._validate_payment_split(action, state)
                if payment_correction:
                    retry_context.append(
                        {"role": "user", "content": payment_correction}
                    )
                    continue

                if self.should_run_critic_review(action, state):
                    crit = self._get_critic().review(
                        self, state, action, critic_rejection_history
                    )
                    if not crit.get("approved", True):
                        fb = crit.get("feedback_for_executor") or crit.get(
                            "reason", "Action rejected."
                        )
                        critic_rejection_history.append(
                            f"[Attempt {attempt + 1}] Proposed {action.name} — "
                            f"Rejected: {fb}"
                        )
                        retry_context.append(
                            {
                                "role": "user",
                                "content": (
                                    f"[Critic attempt {attempt + 1}] {fb}\n"
                                    "Prior critic rejections on this turn:\n"
                                    + "\n".join(
                                        f"  • {r}" for r in critic_rejection_history[:-1]
                                    )
                                    + "\nDo NOT propose the same action again. "
                                    "Fix the specific issue above, or ask the user "
                                    "for the single piece of information you are missing."
                                ),
                            }
                        )
                        continue
                break
            else:
                # All retries exhausted without approval. Build a targeted question from
                # the critic's most recent concrete rejection rather than a blank ask.
                if critic_rejection_history:
                    last_fb = critic_rejection_history[-1]
                    fallback_msg = (
                        f"I'm having trouble completing your request and need a bit of "
                        f"clarification. Here's where I got stuck: {last_fb.split('Rejected:')[-1].strip()[:300]} "
                        "Could you confirm or correct that information so I can proceed?"
                    )
                else:
                    fallback_msg = (
                        "I want to make sure I handle your request correctly. "
                        "Could you confirm the key details — such as the reservation, "
                        "flights, or payment — so I can proceed?"
                    )
                action = Action(
                    name=RESPOND_ACTION_NAME,
                    kwargs={RESPOND_ACTION_FIELD_NAME: fallback_msg},
                )
                executor_msg = {
                    "role": "assistant",
                    "content": fallback_msg,
                }

            # ---------- EXECUTE ACTION ----------
            assert action is not None and executor_msg is not None
            env_resp = env.step(action)
            state.reward = env_resp.reward
            state.info = {**state.info, **env_resp.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                state.completed_tool_calls.append(
                    CompletedToolCall(
                        name=action.name,
                        kwargs=action.kwargs,
                        result_summary=(env_resp.observation or "")[:200],
                        full_result=env_resp.observation or "",
                    )
                )
                if executor_msg.get("tool_calls"):
                    executor_msg["tool_calls"] = executor_msg["tool_calls"][:1]
                    state.executor_messages.extend(
                        [
                            executor_msg,
                            {
                                "role": "tool",
                                "tool_call_id": executor_msg["tool_calls"][0]["id"],
                                "name": executor_msg["tool_calls"][0]["function"][
                                    "name"
                                ],
                                "content": env_resp.observation,
                            },
                        ]
                    )
                else:
                    state.executor_messages.append(
                        {
                            "role": "user",
                            "content": f"API output: {env_resp.observation}",
                        }
                    )
                last_source = "tool"
            else:
                content = action.kwargs.get(RESPOND_ACTION_FIELD_NAME, "")
                state.executor_messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": env_resp.observation},
                    ]
                )
                last_source = "user"

                # Mark active step done after responding to user
                if state.approved_plan and state.active_step_id:
                    for s in state.approved_plan.steps:
                        if s.id == state.active_step_id and s.status in (
                            "in_progress",
                            "pending",
                        ):
                            s.status = "done"
                            break

            if env_resp.done:
                break

        return SolveResult(
            reward=state.reward,
            messages=state.executor_messages,
            info={**state.info, "multi_agent_trace": state.internal_trace},
            total_cost=state.total_cost,
        )


class CriticAgent:
    """Independent LLM agent that reviews the Executor's proposed action before env step.

    Invoked by ``MultiAgentV1`` for high-stakes tools, transfers to human, and for
    user-facing ``respond`` immediately after flight search tool results (where false
    negatives like "no flights" are common).
    """

    def __init__(
        self,
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def review(
        self,
        orchestrator: "MultiAgentV1",
        state: ConversationState,
        action: Action,
        prior_rejections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        recent = state.executor_messages[-20:]
        tool_blob = orchestrator._format_tool_output_data(
            state.completed_tool_calls,
            max_per_tool=4500,
            max_total=18000,
        )
        prior_str = (
            "\n".join(f"  {i+1}. {r}" for i, r in enumerate(prior_rejections))
            if prior_rejections
            else "None — this is the first review on this turn."
        )
        system_prompt = CRITIC_AGENT_SYSTEM.format(wiki=self.wiki)
        user_payload = f"""# Original User Request (task opening user message)
{state.original_user_request[:4000]}

# Current Plan
{orchestrator._format_plan(state.approved_plan)}

# Active Step
{orchestrator._get_active_step_description(state.approved_plan, state.active_step_id)}

# Tool Calls Already Completed (summaries)
{orchestrator._format_completed_tools(state.completed_tool_calls)}

# Actual Tool Output Data (raw JSON from tools — authoritative)
{tool_blob}

# YOUR OWN PRIOR REJECTIONS ON THIS TURN (do not contradict these)
{prior_str}

# Proposed action
Tool name: {action.name}
Arguments (JSON):
{json.dumps(action.kwargs, indent=2)}

# Recent conversation (abbreviated)
{orchestrator._format_conversation_for_planner(recent)}

IMPORTANT: If you already rejected a similar action on this turn (see prior rejections above),
only reject again if the executor made a DIFFERENT concrete mistake. Do NOT generate a new
objection to the same action — that creates an infinite loop. If the executor fixed the
prior issue and you have no new concrete evidence of a problem, APPROVE.

Return your verdict as a single JSON object only."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
        api_base = os.getenv("AGENT_MODEL_API_BASE") or os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("AGENT_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY")
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
            api_base=api_base,
            api_key=api_key,
        )
        cost = res._hidden_params.get("response_cost", 0) or 0
        state.total_cost += cost

        content = res.choices[0].message.content or ""
        parsed = _parse_json_response(content)
        state.internal_trace.append(
            {
                "role": "critic_agent",
                "agent": "CriticAgent",
                "action_reviewed": {"name": action.name, "kwargs": action.kwargs},
                "raw_output": content,
                "parsed": parsed,
                "cost": cost,
            }
        )
        if parsed is None:
            return {
                "approved": True,
                "reason": "Critic agent output unparseable; approving.",
                "feedback_for_executor": None,
                "risk_level": "medium",
                "evidence": None,
            }
        return parsed
