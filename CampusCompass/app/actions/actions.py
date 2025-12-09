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
            mode_hint = (tracker.get_slot("travel_mode_hint") or "").strip()
            travel_mode = llm.normalize_travel_mode(mode_hint)
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

        # Kaartje als aparte bubble erachteraan
        if map_url:
            dispatcher.utter_message(
                text="Route map",
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
                    "title": "Open Campus Map",
                    "payload": f"/{campus_map_url}",
                    "url": campus_map_url,
                }
            ],
        )

        return []

