# Copyright Sierra
# Policy-Sentinel: validates a proposed tool call against domain rules and current env state
# before execution (addresses Domain Policy Violations and Contextual Misinterpretation).

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from tau_bench.types import Action, RESPOND_ACTION_NAME


class PolicySentinel:
    """Reviews proposed tool calls against domain rules and environment state. Returns (allowed, correction_message)."""

    def __init__(self, rules: List[str]) -> None:
        self.rules = rules

    def check(
        self, action: Action, env_data: Dict[str, Any], env_name: str = "retail"
    ) -> Tuple[bool, str]:
        """
        Check if the proposed action is allowed given current env state and rules.
        Returns (True, "") if allowed, (False, "Correction: ...") if blocked.
        """
        if action.name == RESPOND_ACTION_NAME:
            return True, ""

        if env_name == "retail":
            return self._check_retail(action, env_data)
        if env_name == "airline":
            return self._check_airline(action, env_data)
        return True, ""

    def _check_retail(self, action: Action, data: Dict[str, Any]) -> Tuple[bool, str]:
        orders = data.get("orders") or {}
        kwargs = action.kwargs or {}
        order_id = kwargs.get("order_id")

        if order_id and order_id in orders:
            status = orders[order_id].get("status", "").lower()
        else:
            status = None

        # exchange_delivered_order_items: order must be delivered
        if action.name == "exchange_delivered_order_items":
            if not order_id:
                return False, "Correction needed: exchange_delivered_order_items requires order_id."
            if order_id not in orders:
                return True, ""  # let tool return "order not found"
            if status != "delivered":
                return (
                    False,
                    f"Correction needed: Order {order_id} is in status '{status}'. "
                    "Only orders with status 'delivered' can be exchanged. Use modify_pending_order_items for pending orders.",
                )

        # modify_pending_order_items: order must be pending
        if action.name == "modify_pending_order_items":
            if not order_id:
                return False, "Correction needed: modify_pending_order_items requires order_id."
            if order_id not in orders:
                return True, ""
            if status != "pending" and "pending" not in (status or ""):
                return (
                    False,
                    f"Correction needed: Order {order_id} is in status '{status}'. "
                    "Only orders with status 'pending' can be modified with this tool. Use exchange_delivered_order_items for delivered orders.",
                )

        # return_delivered_order_items: order must be delivered
        if action.name == "return_delivered_order_items":
            if order_id and order_id in orders and status and status != "delivered":
                return (
                    False,
                    f"Correction needed: Order {order_id} status is '{status}'. return_delivered_order_items applies only to delivered orders.",
                )

        # cancel_pending_order: order must be pending
        if action.name == "cancel_pending_order":
            if order_id and order_id in orders and status and status != "pending" and "pending" not in (status or ""):
                return (
                    False,
                    f"Correction needed: Order {order_id} status is '{status}'. Only pending orders can be cancelled with cancel_pending_order.",
                )

        return True, ""

    def _check_airline(self, action: Action, data: Dict[str, Any]) -> Tuple[bool, str]:
        reservations = data.get("reservations") or {}
        kwargs = action.kwargs or {}
        reservation_id = kwargs.get("reservation_id")
        reservation = reservations.get(reservation_id) if reservation_id else None

        # cancel_reservation: cannot cancel already cancelled
        if action.name == "cancel_reservation":
            if reservation and reservation.get("status") == "cancelled":
                return (
                    False,
                    f"Correction needed: Reservation {reservation_id} is already cancelled.",
                )
            return True, ""

        # update_reservation_flights: basic economy cannot change flight segments
        if action.name == "update_reservation_flights":
            if reservation and reservation.get("cabin") == "basic_economy":
                new_flights = kwargs.get("flights") or []
                cur_flights = reservation.get("flights") or []

                def _flight_key(f: Dict[str, Any]) -> tuple:
                    return (f.get("flight_number"), f.get("date"))

                cur_keys = sorted(_flight_key(f) for f in cur_flights)
                new_keys = sorted(_flight_key(f) for f in new_flights)
                if cur_keys != new_keys:
                    return (
                        False,
                        "Correction needed: Basic economy flights cannot be modified. "
                        "Only cabin class can be changed without changing flight segments.",
                    )
            return True, ""

        # update_reservation_baggages: can add but not remove bags
        if action.name == "update_reservation_baggages":
            if reservation:
                total = kwargs.get("total_baggages", 0)
                nonfree = kwargs.get("nonfree_baggages", 0)
                cur_total = reservation.get("total_baggages", 0)
                cur_nonfree = reservation.get("nonfree_baggages", 0)
                if total < cur_total or nonfree < cur_nonfree:
                    return (
                        False,
                        "Correction needed: You can add but not remove checked bags. "
                        "total_baggages and nonfree_baggages cannot be decreased.",
                    )
            return True, ""

        # update_reservation_passengers: cannot change number of passengers
        if action.name == "update_reservation_passengers":
            if reservation:
                new_passengers = kwargs.get("passengers") or []
                cur_count = len(reservation.get("passengers") or [])
                if len(new_passengers) != cur_count:
                    return (
                        False,
                        "Correction needed: You cannot modify the number of passengers. "
                        "Only passenger details (name, dob) can be updated.",
                    )
            return True, ""

        return True, ""


class LLMSentinel:
    """
    LLM-based Sentinel: uses an LLM API to validate proposed tool calls against
    domain rules and environment state. Requires a separate vLLM server (e.g. via
    sentinel-vllm-job.sh) exposing SENTINEL_MODEL_API_BASE.
    """

    SENTINEL_SYSTEM = """You are a Policy Sentinel. Your job is to review proposed tool calls and decide if they violate domain rules or are invalid given the current environment state.

You must respond with exactly one of:
- GO - if the action is allowed
- NO_GO - if the action violates a policy or is invalid

If NO_GO, you must also provide a short "Correction needed: ..." message explaining why the action is blocked. Keep corrections concise and actionable."""

    def __init__(
        self,
        rules: List[str],
        model: str = "local",
        provider: str = "openai",
        temperature: float = 0.0,
    ) -> None:
        self.rules = rules
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def check(
        self, action: Action, env_data: Dict[str, Any], env_name: str = "retail"
    ) -> Tuple[bool, str]:
        """
        Check if the proposed action is allowed. Calls the Sentinel LLM API.
        API base is read from SENTINEL_MODEL_API_BASE or OPENAI_API_BASE (assistant base) at call time.
        Returns (True, "") if allowed, (False, "Correction: ...") if blocked.
        """
        if action.name == RESPOND_ACTION_NAME:
            return True, ""

        api_base = os.getenv("SENTINEL_MODEL_API_BASE") or os.getenv("OPENAI_API_BASE")

        user_content = self._build_prompt(action, env_data, env_name)
        messages = [
            {"role": "system", "content": self.SENTINEL_SYSTEM},
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
            reply = (res.choices[0].message.content or "").strip().upper()
        except Exception as e:
            # On API failure, allow the action (fail open) to avoid blocking runs
            return True, ""

        if "NO_GO" in reply or "NO GO" in reply:
            # Extract correction message if present
            correction = "Correction needed: Action violates domain policy or current state."
            if "Correction needed:" in (res.choices[0].message.content or ""):
                raw = res.choices[0].message.content
                idx = raw.find("Correction needed:")
                if idx >= 0:
                    correction = raw[idx:].split("\n")[0].strip()
            return False, correction
        return True, ""

    def _build_prompt(
        self, action: Action, env_data: Dict[str, Any], env_name: str
    ) -> str:
        rules_text = "\n".join(f"- {r}" for r in self.rules) if self.rules else "None provided."
        state_preview = json.dumps(env_data, indent=2, default=str)[:4000]
        return f"""Domain: {env_name}

Domain rules:
{rules_text}

Current environment state (relevant data):
{state_preview}

Proposed action:
- Tool: {action.name}
- Arguments: {json.dumps(action.kwargs or {}, indent=2)}

Does this action violate any domain policy or is it invalid given the current state? Respond with GO or NO_GO, and if NO_GO provide a "Correction needed: ..." message."""

