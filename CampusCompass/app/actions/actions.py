from __future__ import annotations
from typing import Any, Dict, List, Optional, Text

import logging

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, EventType
from CampusCompass.app.api.llmcontroller import LLMController
from CampusCompass.app.api.mapscontroller import MapsController

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
            route_data = maps.get_walking_directions(
                origin_name=source_normalized,
                destination_name=target_normalized,
            )

            # 3) let the LLM turn route_data into a user-friendly message
            answer_text = llm.format_route_description(
                origin_name=source_normalized,
                destination_name=target_normalized,
                route=route_data,
            )

        except Exception as e:
            logger.error(f"[ActionGetRouteDescription] failed: {e}")
            dispatcher.utter_message(response="utter_route_api_error")
            return []

        dispatcher.utter_message(text=answer_text)
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

