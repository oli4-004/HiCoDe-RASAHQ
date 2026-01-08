from __future__ import annotations
from typing import Any, Dict, List, Optional, Text

import logging
import json

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, EventType, FollowupAction
from rasa_sdk.executor import CollectingDispatcher
from CampusCompass.app.api.llmcontroller import LLMController
from CampusCompass.app.api.mapscontroller import MapsController
from rasa.core.actions.action_trigger_search import ActionTriggerSearch
from rasa.shared.core.events import BotUttered, Event
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker

logger = logging.getLogger("campuscompass.actions")

# ---------------------------
# Lazy controller
# ---------------------------
_llm_controller: Optional[LLMController] = None
_maps_controller: Optional[MapsController] = None


def get_llm_controller() -> LLMController:
    """
    Create controller only when needed.
    Prevents import-time crashes and keeps Rasa core isolated.
    """
    global _llm_controller
    if _llm_controller is None:
        _llm_controller = LLMController()
    return _llm_controller


def get_maps_controller() -> MapsController:
    """
    Lazy-initialised shared MapsController instance.
    """
    global _maps_controller
    if _maps_controller is None:
        _maps_controller = MapsController()
    return _maps_controller


def build_history_snippet(tracker: Tracker, max_messages: int = 10) -> str:
    """
    Build a short conversation snippet from the last few user+assistant
    messages, in chronological order, with explicit prefixes.

    Example:
        User: Oh no! I lost my student card...
        Assistant: You can request a new student card at the Central Student Desk in the Berchmanianum.
        User: How can I get there as quickly as possible?
    """
    # If we encounter a corrected bot message, we skip earlier bot messages
    # until we hit the previous user message (same turn masking).
    lines: List[str] = []
    skip_bots_until_user = False

    for e in reversed(tracker.events):
        ev_type = e.get("event")
        if ev_type not in ("user", "bot"):
            continue

        # If we already saw a corrected bot message, ignore older bot messages
        # from the same turn until we reach a user message.
        if skip_bots_until_user and ev_type == "bot":
            continue

        text = (e.get("text") or "").strip()
        if not text:
            continue

        # Detect "this bot message overrides earlier bot messages in this turn"
        if ev_type == "bot":
            md = e.get("metadata") or {}
            if isinstance(md, dict) and md.get("fact_check_override") is True:
                # From now on (while walking back), ignore earlier bots until we hit a user event.
                skip_bots_until_user = True

        if ev_type == "user":
            # Once we hit the previous user message, stop skipping bots.
            skip_bots_until_user = False

        prefix = "User" if ev_type == "user" else "Assistant"
        lines.append(f"{prefix}: {text}")

        if len(lines) >= max_messages:
            break

    if not lines:
        return ""

    return "\n".join(reversed(lines))


# ---------------------------------------------------------------------------
# SMALLTALK
# ---------------------------------------------------------------------------

class ActionSmalltalkLLM(Action):
    """
    Generate ONE short, friendly sentence that reacts to off-topic smalltalk,
    without actually answering campus questions.
    """

    def name(self) -> Text:
        return "action_smalltalk_llm"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        user_text = tracker.latest_message.get("text") or ""
        controller = get_llm_controller()

        try:
            reply = controller.smalltalk_reply(user_text)
        except Exception as e:
            logger.error(f"[ActionSmalltalkLLM] smalltalk_reply failed: {e}")
            reply = "Got it 😄"

        dispatcher.utter_message(text=reply)
        return []


# ---------------------------------------------------------------------------
# ROUTES & TRAVEL MODE
# ---------------------------------------------------------------------------

class ActionGetRouteDescription(Action):
    """
    Orchestrates:
    - read raw source/target building slots
    - normalize them via LLMController (using building docs)
    - ask MapsController for walking directions
    - let LLMController turn the route into a nice message
    """

    def name(self) -> Text:
        return "action_get_route_description"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        mode_hint = (tracker.get_slot("travel_mode_hint") or "").strip().lower()
        if mode_hint == "public_transport":
            dispatcher.utter_message(
                text="For bus/train routes I recommend a live planner (9292 or Google Maps). "
                     "If you want a walking, cycling or driving route, tell me: walk / bike / car."
            )
            return []

        source_raw = (tracker.get_slot("source_building_raw") or "").strip()
        target_raw = (tracker.get_slot("target_building_raw") or "").strip()

        logger.info(
            "[ActionGetRouteDescription] slots=%s",
            tracker.current_slot_values(),
        )

        if not source_raw:
            dispatcher.utter_message(
                text="I am still missing your starting point."
            )
            return []

        if not target_raw:
            dispatcher.utter_message(
                text="I am still missing your destination point."
            )
            return []

        llm = get_llm_controller()
        maps = get_maps_controller()

        try:
            # 1) normalize building names based on your docs
            source_normalized = llm.normalize_building_name(source_raw)
            target_normalized = llm.normalize_building_name(target_raw)

            logger.info(
                "[ActionGetRouteDescription] normalized source=%r target=%r",
                source_normalized,
                target_normalized,
            )

            # 2) call MapsController to get structured route data
            travel_mode = llm.normalize_travel_mode(mode_hint)
            logger.info("[ROUTE] normalized_mode=%s", travel_mode)
            route_data = maps.get_walking_directions(
                origin_name=source_normalized,
                destination_name=target_normalized,
                mode=travel_mode,
            )

            # 3) let the LLM turn route_data into a user-friendly message
            answer_text = llm.format_route_description(
                origin_name=source_normalized,
                destination_name=target_normalized,
                route=route_data,
                mode=travel_mode
            )

        except Exception as e:
            logger.error(f"[ActionGetRouteDescription] failed: {e}")
            dispatcher.utter_message(response="utter_route_api_error")
            return []

        # 4) Stuur de route terug als meerdere bubbels + eventueel een kaartje
        map_url: Optional[str] = None
        if isinstance(route_data, dict):
            map_url = route_data.get("static_map_url") or None

        # Split de gegenereerde tekst op regels en stuur elke niet-lege regel als aparte bubble
        lines = [line.strip() for line in (answer_text or "").split("\n") if line.strip()]
        if not lines:
            # fallback: toch één bericht sturen als er iets misging met de formatting
            dispatcher.utter_message(text=answer_text or "Here is your route.")
        else:
            for line in lines:
                dispatcher.utter_message(text=line)

        if map_url:
            dispatcher.utter_message(
                image=map_url,
            )
        return []


class ActionClearRouteContext(Action):
    def name(self) -> Text:
        return "action_clear_route_context"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        events: List[EventType] = []

        prev_source = tracker.get_slot("source_building_raw")
        prev_target = tracker.get_slot("target_building_raw")

        if prev_source:
            events.append(SlotSet("prev_source_building_raw", prev_source))
        if prev_target:
            events.append(SlotSet("prev_target_building_raw", prev_target))

        events.extend(
            [
                SlotSet("source_building_raw", None),
                SlotSet("target_building_raw", None),
                SlotSet("travel_mode_hint", "unknown"),
            ]
        )
        return events


class ActionLLMPrefillRouteSlots(Action):
    """
    Use the LLM to prefill source_building_raw and target_building_raw
    from the latest user message.

    - Does NOT ask the user anything.
    - Only runs on the latest text; no extra context.
    - Respects existing slot values (does not overwrite them).
    """

    def name(self) -> Text:
        return "action_llm_prefill_route_slots"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        events: List[EventType] = []

        # Respect existing slots: if both are already set, do nothing.
        current_source = (tracker.get_slot("source_building_raw") or "").strip()
        current_target = (tracker.get_slot("target_building_raw") or "").strip()

        if current_source and current_target:
            return events

        prev_source = (tracker.get_slot("prev_source_building_raw") or "").strip()
        prev_target = (tracker.get_slot("prev_target_building_raw") or "").strip()

        history_snippet = build_history_snippet(tracker, max_messages=10)
        if not history_snippet:
            return events

        llm = get_llm_controller()

        try:
            result = llm.extract_route_entities(
                history_snippet,
                prev_source=prev_source or None,
                prev_destination=prev_target or None,
            )
        except Exception as e:
            logger.error(f"[ActionLLMPrefillRouteSlots] extract_route_entities failed: {e}")
            return events

        src = (result.get("source") or "").strip() if result else ""
        tgt = (result.get("destination") or "").strip() if result else ""

        logger.info(
            "[ActionLLMPrefillRouteSlots] from text=%r -> source=%r, destination=%r",
            history_snippet,
            src,
            tgt,
        )

        if not current_source and src:
            events.append(SlotSet("source_building_raw", src))
        if not current_target and tgt:
            events.append(SlotSet("target_building_raw", tgt))

        return events


class ActionLLMPrefillTravelMode(Action):
    """
    Use the LLM to prefill the travel_mode_hint slot from the latest
    user message.

    - Does NOT ask the user anything.
    - Only looks at the latest text; no extra context.
    - Respects existing slot values (does not overwrite a non-unknown value).
    """

    def name(self) -> Text:
        return "action_llm_prefill_travel_mode"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        events: List[EventType] = []

        VALID = {"walk", "bike", "car", "public_transport", "unknown"}

        current_hint = (tracker.get_slot("travel_mode_hint") or "").strip().lower()

        if current_hint in VALID and current_hint != "unknown":
            return events

        history_snippet = build_history_snippet(tracker, max_messages=10)
        if not history_snippet:
            return events

        llm = get_llm_controller()

        try:
            classified = llm.classify_travel_mode_hint(history_snippet)
        except Exception as e:
            logger.error(f"[ActionLLMPrefillTravelMode] classify_travel_mode_hint failed: {e}")
            return events

        mode_hint = (classified or "").strip().lower()
        logger.info(
            "[ActionLLMPrefillTravelMode] from text=%r -> travel_mode_hint=%r",
            history_snippet,
            mode_hint,
        )

        if mode_hint and mode_hint != "unknown":
            events.append(SlotSet("travel_mode_hint", mode_hint))

        return events


# ---------------------------------------------------------------------------
# ABBREVIATIONS
# ---------------------------------------------------------------------------

class ActionLLMPrefillAbbreviation(Action):
    """
    Use the LLM to prefill the abbreviation_raw slot from the latest
    user message.

    - Does NOT ask the user anything.
    - Only looks at the latest text; no extra context.
    - Respects an existing abbreviation_raw (does not overwrite it).
    """

    def name(self) -> Text:
        return "action_llm_prefill_abbreviation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        events: List[EventType] = []

        current_abbr = (tracker.get_slot("abbreviation_raw") or "").strip()
        if current_abbr:
            # Already set → do not overwrite.
            return events

        history_snippet = build_history_snippet(tracker, max_messages=10)
        if not history_snippet:
            return events

        llm = get_llm_controller()

        try:
            extracted = llm.extract_abbreviation(history_snippet)
        except Exception as e:
            logger.error(f"[ActionLLMPrefillAbbreviation] extract_abbreviation failed: {e}")
            return events

        abbr = (extracted or "").strip()
        logger.info(
            "[ActionLLMPrefillAbbreviation] from text=%r -> abbreviation_raw=%r",
            history_snippet,
            abbr,
        )

        if abbr:
            events.append(SlotSet("abbreviation_raw", abbr))

        return events


class ActionClearAbbreviationContext(Action):
    def name(self) -> Text:
        return "action_clear_abbreviation_context"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[EventType]:
        events: List[EventType] = []
        events.append(SlotSet("abbreviation_raw", None))

        return events


# ---------------------------------------------------------------------------
# FALLBACK
# ---------------------------------------------------------------------------
class ActionCannotHandle(Action):
    """
    Stuurt een fallback met de link naar de campusplattegrond.
    """

    def name(self) -> Text:
        return "action_cannot_handle"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        campus_map_url = "https://www.ru.nl/sites/default/files/2025-05/campusplattegrond-2025.pdf"

        dispatcher.utter_message(
            text="I'm sorry, I cannot help you with this issue. If you are in a hurry, please refer to Google Maps or the campus map."
        )

        dispatcher.utter_message(
            text="Campus Map",
            buttons=[
                {
                    "title": "Click here",
                    "url": campus_map_url,
                }
            ],
        )

        return []

class ActionIncrementFallbackAttempts(Action):
    """
        Telt hoevaak de chatbot niet begrijpt wat de user bedoelt
    """
    def name(self) -> str:
        return "action_increment_fallback_attempts"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        cur = tracker.get_slot("fallback_attempts")
        try:
            n = int(float(cur)) if cur is not None else 0
        except Exception:
            n = 0
        return [SlotSet("fallback_attempts", n + 1)]

class ActionTriggerSearchLLM(Action):
    """
    Replacement for enterprise-search style responses:
    - Select relevant kb files via lookup_kb.json (LLMController._load_building_docs)
    - LLM returns JSON with status: ANSWER / NEED_INFO / NOT_IN_DOCS
    - For ANSWER: utter answer
    - For NEED_INFO / NOT_IN_DOCS: set slots only; flows decide what to do
    """

    def name(self) -> Text:
        # Keep this name if your Rasa setup calls "action_trigger_search"
        return "action_trigger_search"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[EventType]:

        user_text = (tracker.latest_message.get("text") or "").strip()
        if not user_text:
            dispatcher.utter_message(text="I didn’t catch your question—could you rephrase it?")
            return [
                SlotSet("docs_status", None),
                SlotSet("docs_need_info_question", None),
            ]

        llm = get_llm_controller()

        # Use recent conversation context to resolve pronouns like "there", "it", "that building", etc.
        history_snippet = build_history_snippet(tracker, max_messages=10)

        try:
            raw_json = llm.answer_question_from_docs(
                question_text=user_text,
                history_snippet=history_snippet,
            )
        except Exception as e:
            logger.error(f"[ActionTriggerSearchLLM] answer_question_from_docs failed: {e}")
            dispatcher.utter_message(text="Sorry — I couldn’t answer that reliably right now.")
            return [
                SlotSet("docs_status", None),
                SlotSet("docs_need_info_question", None),
            ]

        obj: Dict[str, Any] = {}
        try:
            obj = json.loads((raw_json or "").strip())
        except Exception:
            logger.error(f"[ActionTriggerSearchLLM] invalid JSON from LLM: {raw_json!r}")
            obj = {"status": "NOT_IN_DOCS"}

        status = (obj.get("status") or "").strip()

        # Always clear question unless we explicitly set it
        events: List[EventType] = [
            SlotSet("docs_need_info_question", None),
        ]

        if status == "ANSWER":
            answer = (obj.get("answer") or "").strip()
            if not answer:
                # Treat as not found
                events.append(SlotSet("docs_status", "NOT_IN_DOCS"))
                return events

            dispatcher.utter_message(
                text=answer,
                metadata={"answered_by": "llm_docs_selector", "docs_status": "ANSWER"},
            )
            events.append(SlotSet("docs_status", None))  # clear so we don't get stuck
            return events

        if status == "NEED_INFO":
            question = (obj.get("question") or "").strip()
            if not question:
                events.append(SlotSet("docs_status", "NOT_IN_DOCS"))
                return events

            # Do NOT utter here; flows will ask/collect.
            events.append(SlotSet("docs_status", "NEED_INFO"))
            events.append(SlotSet("docs_need_info_question", question))
            return events

        # default: NOT_IN_DOCS
        events.append(SlotSet("docs_status", "NOT_IN_DOCS"))
        return events
