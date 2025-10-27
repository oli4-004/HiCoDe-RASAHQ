from typing import Text, Any, Dict, List
from rasa_sdk import Tracker, FormValidationAction, Action
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from CampusCompass.llm.llmcontroller import LLMController

CONFIDENCE_THRESHOLD = 0.8

class ValidateRouteForm(FormValidationAction):
    def name(self) -> Text:
        """
        :return: the name of the action
        """
        return "validate_route_form"

    def normalize(self, raw:Text, restrict_to: List[str] | None = None):
        """
        Uses the LLM to map 'raw' (possibly vague) to a valid form entry
        Returns a single normalized name or None to re-ask.

        :param raw: the raw text of the form
        """
        llm = LLMController()
        result = llm.normalize_building(raw = raw, restrict_to = restrict_to)
        return result

    def clear_disambiguation_states(self):
        """
        Clears the disambiguation states in domain.yml
        """
        return [SlotSet("followup_question", None),
                SlotSet("disambiguation_candidates", []),
                SlotSet("disambiguation_target_slot", None)]

    async def validate_origin_building(self, slot_value: Any, dispatcher:CollectingDispatcher, tracker: Tracker, domain: Dict) -> Dict[Text, Any]:
        """
        Normalizes and checks whether we can identify where the user currently is.

        :param slot_value: the raw text of the form
        :param dispatcher: is able to communicate with the user
        :param tracker: is able to retrieve the conversation
        :param domain: the entire domain file in a dictionary
        :return: tries to fill the slot or asks another question
        """

        target = tracker.get_slot("disambiguation_target_slot")
        restrict_to = tracker.get_slot("disambiguation_candidates") or []
        in_disambiguation_for_origin = (target == "origin_building" and len(restrict_to) > 0)

        result = self.normalize(
            str(slot_value) if slot_value is not None else None,
            restrict_to=restrict_to if in_disambiguation_for_origin else None
        )

        normalized, confidence = result.get("normalized"), result.get("confidence", 0.0)
        if normalized and confidence >= CONFIDENCE_THRESHOLD:
            return {"origin_building": normalized}

        candidates = [candidate.get("name") for candidate in (result.get("candidates") or []) if candidate.get("name") is not None]

        events = [
            SlotSet("origin_building", None),
            SlotSet("followup_question", result.get("followup_question") or ""),
            SlotSet("disambiguation_candidates", candidates),
            SlotSet("disambiguation_target_slot", "origin_building"),
        ]

        return {event.key: event.value for event in events if isinstance(event, SlotSet)}

    async def validate_destination_building(self, slot_value: Any, dispatcher:CollectingDispatcher, tracker: Tracker, domain: Dict) -> Dict[Text, Any]:
        """
        Normalizes and checks whether we can identify where the user currently wants to go.

        :param slot_value: the raw text of the form
        :param dispatcher: is able to communicate with the user
        :param tracker: is able to retrieve the conversation
        :param domain: the entire domain file in a dictionary
        :return: tries to fill the slot or asks another question
        """

        target = tracker.get_slot("disambiguation_target_slot")
        restrict_to = tracker.get_slot("disambiguation_candidates") or []
        in_disambiguation_for_destination = (target == "destination_building" and len(restrict_to > 0))

        result = self.normalize(
            str(slot_value) if slot_value is not None else None,
            restrict_to=restrict_to if in_disambiguation_for_destination else None
        )

        normalized, confidence = result.get("normalized"), result.get("confidence", 0.0)
        if normalized and confidence >= CONFIDENCE_THRESHOLD:
            return {"destination_building": normalized}

        candidates = [candidate.get("name") for candidate in (result.get("candidates") or []) if
                      candidate.get("name") is not None]

        events = [
            SlotSet("destination_building", None),
            SlotSet("followup_question", result.get("followup_question") or ""),
            SlotSet("disambiguation_candidates", candidates),
            SlotSet("disambiguation_target_slot", "destination_building"),
        ]

        return {event.key: event.value for event in events if isinstance(event, SlotSet)}

class ActionRouteSummary(Action):
    def name(self) -> Text:
        return "action_route_summary"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict):
        origin = tracker.get_slot("origin_building")
        dest = tracker.get_slot("destination_building")
        dispatcher.utter_message(text=f"(debug) computing route from '{origin}' to '{dest}' …")
        return []




