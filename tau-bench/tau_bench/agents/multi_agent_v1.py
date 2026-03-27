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

# Short user messages that constitute confirmation of a previously-proposed action.
CONFIRMATION_TOKENS = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright",
    "go ahead", "proceed", "please do", "please proceed", "confirm",
    "confirmed", "do it", "sounds good", "correct", "that's correct",
    "that's right", "please", "go for it", "make it happen", "that works",
    "i confirm", "i agree", "absolutely", "of course", "fine",
})

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
    tool_name: str
    tool_args: Dict[str, Any]
    status: str  # pending | in_progress | done | failed
    needs_review_before_call: bool = False
    is_state_changing: bool = False

@dataclass
class Plan:
    goal: str
    plan_text: str
    steps: List[PlanStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "plan_text": self.plan_text,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "tool_name": s.tool_name,
                    "tool_args": s.tool_args,
                    "status": s.status,
                    "needs_review_before_call": s.needs_review_before_call,
                    "is_state_changing": s.is_state_changing,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Plan":
        return cls(
            goal=d.get("goal", ""),
            plan_text=d.get("plan_text", ""),
            steps=[
                PlanStep(
                    id=s["id"],
                    description=s["description"],
                    tool_name=s["tool_name"],
                    tool_args=s.get("tool_args", {}),
                    status=s.get("status", "pending"),
                    needs_review_before_call=bool(
                        s.get("needs_review_before_call", False)
                    ),
                    is_state_changing=bool(s.get("is_state_changing", False)),
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
    planner_repair_hints: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Role prompts
# ---------------------------------------------------------------------------

PLANNER_INSTRUCTION = """You are a planning agent for a customer-service workflow.

# Domain Policy
{wiki}

# Available Tools (authoritative)
{tools}

# Current Approved Plan
{current_plan}

# Pending Plan Update (awaiting user confirmation)
{pending_update}

# Planner Repair Hints (from critic/tool failures)
{repair_hints}

# Conversation So Far
{conversation}

# Required JSON output (no markdown, no extra text)
{{
  "decision": "continue_existing_plan" | "propose_plan_change" | "request_clarification",
  "reason": "brief reason",
  "plan": {{
    "goal": "overall goal",
    "plan_text": "human readable plan summary",
    "steps": [
      {{
        "id": "s1",
        "description": "what this step does",
        "tool_name": "exact tool name",
        "tool_args": {{}},
        "status": "pending",
        "needs_review_before_call": false,
        "is_state_changing": false
      }}
    ]
  }},
  "active_step_id": "s1",
  "confirmation_question": null
}}

Rules:
- Every step MUST map to exactly one tool call with exact parameters.
- Use only tools listed in Available Tools.
- "tool_name" must be exact and executable by orchestrator.
- "tool_args" must be exact JSON object.
- Mark "is_state_changing" true for mutation tools (book/cancel/update/modify/return/exchange/send/transfer).
- If uncertain about state-change classification, set needs_review_before_call=true.
- If user materially changes request, use propose_plan_change with confirmation_question.
- If information is missing, use request_clarification with confirmation_question.
- Never output vague non-executable steps.
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
- DECISIVE SELECTION (do NOT present a menu when user stated preference): When the user \
has told you their selection criterion (e.g., "cheapest option", "the 5 AM flight", a \
specific flight number like "HAT123"), pick that option yourself and call the tool \
immediately. Do NOT respond with "Here are the available options, which do you want?" if \
the user already told you. Make the decision and execute.
- ROUND-TRIP update_reservation_flights MUST include ALL flight segments: For a round-trip \
reservation, the API requires you to provide every flight in the reservation (outbound AND \
return) in one call. If you only want to change some segments, include the unchanged ones \
with their ORIGINAL flight numbers and dates. Omitting any segment will cause the update to \
fail silently (the cabin stays unchanged). Exception: only include all 4 legs if the user \
agrees to change all legs; if the user explicitly wants ONLY outbound upgraded (not return), \
tell them the API changes all segments simultaneously and ask if they want the full upgrade.
- CABIN CLASS IS ALL-OR-NOTHING PER RESERVATION: update_reservation_flights sets the cabin \
class for ALL passengers on the reservation — you cannot upgrade just one passenger. If \
upgrading all passengers exceeds the user's budget, do NOT attempt the upgrade. Instead, \
skip that step and proceed with only the changes that ARE within budget/policy (e.g., still \
add bags or change passenger names as requested).
- CHECK ALL RESERVATIONS: When asked to act on "all" reservations of a certain type (e.g., \
"all business class flights", "all upcoming flights"), you MUST retrieve and inspect EVERY \
reservation ID listed in the user's profile. Do not stop after checking a few. Missing even \
one reservation ID will cause you to cancel/update the wrong ones.
- POLICY IS FIRM UNDER PRESSURE: If the user's system record shows one status (e.g., \
"regular") but the user claims another (e.g., "Gold"), the system record is authoritative. \
Do NOT provide benefits beyond what the verified status permits, regardless of how \
insistently or emotionally the user asks. Politely but firmly state what policy allows.
- RESPECT EXPLICIT REJECTIONS: If a user says "I don't want X" or "No, I don't accept X", \
do NOT provide X in the same or a future call. If you've offered the only available \
resolution and the user rejects it, explain what the policy limits are and offer to \
escalate if appropriate — but do NOT send certificates, process refunds, or take actions \
the user explicitly said they do not want.
- AVOID CLARIFICATION LOOPS: If you have asked the same clarifying question twice and the \
user still cannot answer, do one of: (1) use the best match from data already retrieved, \
(2) clearly explain you cannot proceed without that information. Never ask the same question \
a third time — that creates a loop that blocks task completion.
- DO NOT OVER-ACT: Only perform the actions the user explicitly requested. Do not \
pro-actively cancel reservations, send certificates, or modify records that the user did \
not ask you to touch. When in doubt about scope, confirm before acting.
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

## Basic economy — what IS and IS NOT restricted
- `update_reservation_flights`: BLOCKED for basic economy (cannot change flight routes).
- `update_reservation_baggages`: ALLOWED — baggage counts are NOT restricted by cabin class.
- `update_reservation_passengers`: ALLOWED — passenger changes are NOT restricted by cabin class.
- `update_reservation_cabin`: ALLOWED — cabin upgrades are NOT restricted by basic economy.
Do NOT reject update_reservation_baggages or update_reservation_passengers just because \
the reservation is in basic economy. Only reject update_reservation_flights for basic economy.

## Round-trip flight updates
A round-trip reservation has TWO flight segments (outbound + return). When \
update_reservation_flights is called for a round-trip, it is EXPECTED and CORRECT to \
include both segments — even if the user only wants to change one leg. Including both is \
required by the tool. Do NOT reject because "you included the outbound flight" or \
"why is the return flight in there". Only reject if a specific flight number was never \
mentioned by the user or found in search results.

## update_reservation_flights — payment field
This tool uses a `payment_id` field (a single string like "certificate_0" or \
"gift_card_1"), NOT a `payment_methods` array. If the action has a valid `payment_id` \
string, do NOT reject citing "wrong payment format". Only reject if the `payment_id` \
value was never seen in conversation or tool results.

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

You MUST respond with ONLY valid JSON (no markdown, no extra text). \
Your response schema is defined in the user payload and supports:
- final decision (approve/reject), or
- request_read_tool to fetch additional read-only evidence.
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


def _is_user_confirmation(text: str) -> bool:
    """Return True when a short user message is clearly an affirmative confirmation."""
    stripped = text.strip().lower()
    # Long messages almost always contain new information/instructions, not pure confirmation.
    if len(stripped.split()) > 15:
        return False
    return any(tok in stripped for tok in CONFIRMATION_TOKENS)


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
        lines = [f"Goal: {plan.goal}", f"Plan: {plan.plan_text}"]
        for s in plan.steps:
            lines.append(
                f"  {markers.get(s.status, '[ ]')} {s.id}: {s.description} :: "
                f"{s.tool_name}({json.dumps(s.tool_args)[:120]})"
            )
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
        repair_hints = (
            "\n".join(f"- {h}" for h in state.planner_repair_hints[-8:])
            if state.planner_repair_hints
            else "None"
        )
        prompt = PLANNER_INSTRUCTION.format(
            wiki=self.wiki,
            tools=self._tools_str,
            current_plan=self._format_plan(state.approved_plan),
            pending_update=pending_str,
            repair_hints=repair_hints,
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
        parsed: Optional[Dict[str, Any]] = None
        raw_output = ""
        total_cost = 0.0
        for _ in range(2):
            res = completion(
                model=self.planner_model,
                custom_llm_provider=self.planner_provider,
                messages=messages,
                temperature=self.temperature,
                api_base=api_base,
                api_key=api_key,
            )
            cost = res._hidden_params.get("response_cost", 0) or 0
            total_cost += cost
            content = res.choices[0].message.content or ""
            raw_output = content
            parsed = _parse_json_response(content)
            if parsed is not None and self._is_valid_planner_result(parsed):
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid or non-executable. "
                        "Return valid JSON with executable step objects exactly as specified."
                    ),
                }
            )
            parsed = None

        state.total_cost += total_cost
        state.internal_trace.append(
            {
                "role": "planner",
                "raw_output": raw_output,
                "parsed": parsed,
                "cost": total_cost,
            }
        )

        if parsed is None or not self._is_valid_planner_result(parsed):
            return {
                "decision": "continue_existing_plan",
                "reason": "Failed to parse planner output, continuing.",
                "plan": (
                    state.approved_plan.to_dict()
                    if state.approved_plan
                    else {"goal": "Help the user", "plan_text": "", "steps": []}
                ),
                "active_step_id": state.active_step_id,
                "confirmation_question": None,
            }
        return parsed

    @staticmethod
    def _is_valid_planner_result(result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        if "decision" not in result or "plan" not in result:
            return False
        plan = result.get("plan")
        if not isinstance(plan, dict):
            return False
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return False
        for s in steps:
            if not isinstance(s, dict):
                return False
            required = ("id", "description", "tool_name", "tool_args", "status")
            if any(k not in s for k in required):
                return False
            if not isinstance(s.get("tool_args"), dict):
                return False
        return True

    def _call_executor(
        self,
        state: ConversationState,
        extra_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Action, float]:
        mode_instruction = ""
        if state.executor_mode == "execute_action":
            mode_instruction = (
                "\n\n**⚡ EXECUTE-ACTION MODE**: The user confirmed or a prior step just "
                "completed. You MUST call a tool in this response. Do NOT write a text "
                "response, acknowledgment, or description of what you will do. Identify "
                "the tool required by the active plan step and call it NOW."
            )
        system_prompt = EXECUTOR_SYSTEM_TEMPLATE.format(
            wiki=self.wiki,
            plan_summary=self._format_plan(state.approved_plan),
            active_step=self._get_active_step_description(
                state.approved_plan, state.active_step_id
            ),
            completed_tools=self._format_completed_tools(state.completed_tool_calls),
        ) + mode_instruction
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
    def _is_state_changing_tool(tool_name: str) -> bool:
        return tool_name in HIGH_STAKES_ACTION_NAMES

    @staticmethod
    def _is_read_only_tool(tool_name: str) -> bool:
        return tool_name.startswith(READ_ONLY_PREFIXES)

    @staticmethod
    def _get_active_step(plan: Optional[Plan], step_id: Optional[str]) -> Optional[PlanStep]:
        if plan is None:
            return None
        if step_id:
            for s in plan.steps:
                if s.id == step_id and s.status in ("pending", "in_progress"):
                    return s
        for s in plan.steps:
            if s.status in ("pending", "in_progress"):
                return s
        return None

    @staticmethod
    def _has_open_steps(plan: Optional[Plan]) -> bool:
        return bool(plan and any(s.status in ("pending", "in_progress") for s in plan.steps))

    @staticmethod
    def _next_step(plan: Optional[Plan], current_id: Optional[str]) -> Optional[PlanStep]:
        if plan is None:
            return None
        seen = current_id is None
        for s in plan.steps:
            if not seen and s.id == current_id:
                seen = True
                continue
            if seen and s.status == "pending":
                return s
        return None

    @staticmethod
    def _is_tool_failure(observation: str) -> bool:
        low = (observation or "").strip().lower()
        return (
            low.startswith("error")
            or "failed" in low
            or "not found" in low
            or "invalid" in low
            or "unable to" in low
        )

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
    def _advance_active_step(state: "ConversationState") -> bool:
        """If the current active step is 'done', advance active_step_id to the next
        pending/in_progress step. Returns True if advanced."""
        if not state.approved_plan or not state.active_step_id:
            return False
        current = next(
            (s for s in state.approved_plan.steps if s.id == state.active_step_id),
            None,
        )
        if current is None or current.status != "done":
            return False
        for s in state.approved_plan.steps:
            if s.status == "pending":
                s.status = "in_progress"
                state.active_step_id = s.id
                return True
        return False

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
        need_replan = True

        for _ in range(max_num_steps):
            if last_source == "user" and state.pending_plan_update:
                user_response = state.executor_messages[-1].get("content", "")
                proposed = state.pending_plan_update.get("proposed_plan")
                if (
                    not self._is_rejection(user_response)
                    and proposed
                    and isinstance(proposed, dict)
                ):
                    state.approved_plan = Plan.from_dict(proposed)
                    state.active_step_id = state.pending_plan_update.get("active_step_id")
                state.pending_plan_update = None

            if need_replan or last_source == "user" or not self._has_open_steps(state.approved_plan):
                planner_result = self._call_planner(state)
                decision = planner_result.get("decision", "continue_existing_plan")
                plan_data = planner_result.get("plan")

                if decision == "request_clarification":
                    question = planner_result.get(
                        "confirmation_question",
                        "Could you provide more details to continue?",
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
                    need_replan = True
                    if env_resp.done:
                        break
                    continue

                if decision == "propose_plan_change":
                    state.pending_plan_update = {
                        "proposed_plan": plan_data,
                        "active_step_id": planner_result.get("active_step_id"),
                        "reason": planner_result.get("reason", ""),
                    }
                    question = planner_result.get(
                        "confirmation_question",
                        "Please confirm the updated plan.",
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
                    need_replan = False
                    if env_resp.done:
                        break
                    continue

                if plan_data and isinstance(plan_data, dict):
                    state.approved_plan = Plan.from_dict(plan_data)
                state.active_step_id = planner_result.get("active_step_id", state.active_step_id)
                need_replan = False

            step = self._get_active_step(state.approved_plan, state.active_step_id)
            if step is None:
                final_msg = "I have completed all planned actions."
                env_resp = env.step(
                    Action(
                        name=RESPOND_ACTION_NAME,
                        kwargs={RESPOND_ACTION_FIELD_NAME: final_msg},
                    )
                )
                state.reward = env_resp.reward
                state.info = {**state.info, **env_resp.info.model_dump()}
                state.executor_messages.extend(
                    [
                        {"role": "assistant", "content": final_msg},
                        {"role": "user", "content": env_resp.observation},
                    ]
                )
                last_source = "user"
                if env_resp.done:
                    break
                need_replan = True
                continue

            step.status = "in_progress"
            state.active_step_id = step.id
            action = Action(name=step.tool_name, kwargs=step.tool_args)

            validation = self._validate_action(action, state)
            if not validation.get("approved", True):
                step.status = "failed"
                state.planner_repair_hints.append(
                    validation.get("feedback_for_executor", "Validator rejected planned action.")
                )
                state.internal_trace.append(
                    {
                        "role": "executor_strict",
                        "step_id": step.id,
                        "planned_action": {"name": action.name, "kwargs": action.kwargs},
                        "result": "validation_reject",
                        "feedback": validation.get("feedback_for_executor"),
                    }
                )
                need_replan = True
                continue

            if action.name == "book_reservation":
                payment_correction = self._validate_payment_split(action, state)
                if payment_correction:
                    step.status = "failed"
                    state.planner_repair_hints.append(payment_correction)
                    state.internal_trace.append(
                        {
                            "role": "executor_strict",
                            "step_id": step.id,
                            "planned_action": {"name": action.name, "kwargs": action.kwargs},
                            "result": "payment_validation_reject",
                            "feedback": payment_correction,
                        }
                    )
                    need_replan = True
                    continue

            requires_review = bool(step.needs_review_before_call) or self._is_state_changing_tool(action.name) or bool(step.is_state_changing)
            if requires_review:
                critic_result = self._get_critic().review_with_verification(
                    orchestrator=self,
                    state=state,
                    action=action,
                    env=env,
                )
                if not critic_result.get("approved", False):
                    step.status = "failed"
                    fb = critic_result.get("feedback_for_executor") or critic_result.get("reason", "Critic rejected action.")
                    state.planner_repair_hints.append(fb)
                    state.internal_trace.append(
                        {
                            "role": "critic_gate",
                            "step_id": step.id,
                            "planned_action": {"name": action.name, "kwargs": action.kwargs},
                            "critic_result": critic_result,
                            "result": "blocked",
                        }
                    )
                    need_replan = True
                    continue
                state.internal_trace.append(
                    {
                        "role": "critic_gate",
                        "step_id": step.id,
                        "planned_action": {"name": action.name, "kwargs": action.kwargs},
                        "critic_result": critic_result,
                        "result": "approved",
                    }
                )

            env_resp = env.step(action)
            state.reward = env_resp.reward
            state.info = {**state.info, **env_resp.info.model_dump()}
            state.completed_tool_calls.append(
                CompletedToolCall(
                    name=action.name,
                    kwargs=action.kwargs,
                    result_summary=(env_resp.observation or "")[:200],
                    full_result=env_resp.observation or "",
                )
            )
            state.executor_messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"strict-{step.id}",
                                "function": {
                                    "name": action.name,
                                    "arguments": json.dumps(action.kwargs),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"strict-{step.id}",
                        "name": action.name,
                        "content": env_resp.observation,
                    },
                ]
            )
            state.internal_trace.append(
                {
                    "role": "executor_strict",
                    "step_id": step.id,
                    "planned_action": {"name": action.name, "kwargs": action.kwargs},
                    "result": "executed",
                    "tool_observation": (env_resp.observation or "")[:800],
                }
            )

            if self._is_tool_failure(env_resp.observation or ""):
                step.status = "failed"
                state.planner_repair_hints.append(
                    f"Tool {action.name} failed for step {step.id}: {(env_resp.observation or '')[:300]}"
                )
                need_replan = True
            else:
                step.status = "done"
                nxt = self._next_step(state.approved_plan, step.id)
                if nxt is not None:
                    nxt.status = "pending" if nxt.status == "failed" else nxt.status
                    state.active_step_id = nxt.id
                need_replan = False

            last_source = "tool"
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

    def _review_once(
        self,
        orchestrator: "MultiAgentV1",
        state: ConversationState,
        action: Action,
        verification_observations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        recent = state.executor_messages[-20:]
        tool_blob = orchestrator._format_tool_output_data(
            state.completed_tool_calls,
            max_per_tool=4500,
            max_total=18000,
        )
        verification_blob = json.dumps(verification_observations or [], indent=2)[:8000]
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

# Additional verification reads performed during this critic pass
{verification_blob}

# Proposed action
Tool name: {action.name}
Arguments (JSON):
{json.dumps(action.kwargs, indent=2)}

# Recent conversation (abbreviated)
{orchestrator._format_conversation_for_planner(recent)}

You may return one of:
1) Final verdict:
{{
  "decision": "approve" | "reject",
  "reason": "...",
  "feedback_for_executor": null | "...",
  "evidence": null | "..."
}}
2) Verification request:
{{
  "decision": "request_read_tool",
  "reason": "...",
  "read_tool": {{"name": "get_.../list_.../search_.../calculate/think", "kwargs": {{...}}}}
}}

Only request read-only tools. Do not request mutation tools.

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
                "decision": "approve",
                "reason": "Critic agent output unparseable; approving.",
                "feedback_for_executor": None,
                "evidence": None,
            }
        return parsed

    def review_with_verification(
        self,
        orchestrator: "MultiAgentV1",
        state: ConversationState,
        action: Action,
        env: Env,
        max_iterations: int = 8,
    ) -> Dict[str, Any]:
        verification_observations: List[Dict[str, Any]] = []
        for i in range(max_iterations):
            verdict = self._review_once(
                orchestrator=orchestrator,
                state=state,
                action=action,
                verification_observations=verification_observations,
            )
            decision = (verdict.get("decision") or "").lower()
            if decision in ("approve", "reject"):
                return {
                    "approved": decision == "approve",
                    "reason": verdict.get("reason", ""),
                    "feedback_for_executor": verdict.get("feedback_for_executor"),
                    "evidence": verdict.get("evidence"),
                }
            if decision != "request_read_tool":
                return {
                    "approved": False,
                    "reason": "Critic returned invalid decision.",
                    "feedback_for_executor": "Critic output invalid. Replan required.",
                    "evidence": None,
                }
            read_tool = verdict.get("read_tool") or {}
            name = read_tool.get("name")
            kwargs = read_tool.get("kwargs") or {}
            if not name or not isinstance(kwargs, dict):
                return {
                    "approved": False,
                    "reason": "Critic requested malformed read tool call.",
                    "feedback_for_executor": "Critic verification malformed. Replan required.",
                    "evidence": None,
                }
            if not orchestrator._is_read_only_tool(name):
                return {
                    "approved": False,
                    "reason": f"Critic requested non-read tool '{name}'.",
                    "feedback_for_executor": "Critic attempted mutation during verification.",
                    "evidence": None,
                }

            env_resp = env.step(Action(name=name, kwargs=kwargs))
            state.reward = env_resp.reward
            state.info = {**state.info, **env_resp.info.model_dump()}
            state.completed_tool_calls.append(
                CompletedToolCall(
                    name=name,
                    kwargs=kwargs,
                    result_summary=(env_resp.observation or "")[:200],
                    full_result=env_resp.observation or "",
                )
            )
            state.executor_messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"critic-read-{i}",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(kwargs),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"critic-read-{i}",
                        "name": name,
                        "content": env_resp.observation,
                    },
                ]
            )
            verification_observations.append(
                {
                    "tool_name": name,
                    "kwargs": kwargs,
                    "observation": (env_resp.observation or "")[:5000],
                }
            )
            state.internal_trace.append(
                {
                    "role": "critic_verification_tool",
                    "requested_tool": {"name": name, "kwargs": kwargs},
                    "observation": (env_resp.observation or "")[:1200],
                }
            )
            if env_resp.done:
                return {
                    "approved": False,
                    "reason": "Environment ended during critic verification.",
                    "feedback_for_executor": "Unable to continue; environment terminated.",
                    "evidence": None,
                }

        return {
            "approved": False,
            "reason": "Critic verification exceeded max iterations.",
            "feedback_for_executor": "Critic unable to reach decision; replan required.",
            "evidence": None,
        }
