from typing import Text, Any, Dict, List
from rasa_sdk import Tracker, FormValidationAction, Action
from rasa_sdk.executor import CollectingDispatcher
import logging

from CampusCompass.app.config import CC_ROOT
from CampusCompass.app.llm.llmcontroller import LLMController
from rasa_sdk.events import SlotSet
import re

DEBUG_LOG_PATH = CC_ROOT / "debug.log"

logger = logging.getLogger("campuscompass")
logger.setLevel(logging.INFO)

if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(DEBUG_LOG_PATH)
    for h in logger.handlers
):
    fh = logging.FileHandler(DEBUG_LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

CONFIDENCE_THRESHOLD = 0.8

ABBREVIATION_MAP: Dict[str, str] = {
    "EOS": "Elinor Ostrom Building",
    "E": "Erasmus Building",
    "MM": "Maria Montessori Building",
    "SP": "Spinoza Building",
    "GR": "Grotius Building",
    "LIN": "Linnaeus Building",
    "CC": "Lecture Hall Complex",
    "TVA": "Thomas van Aquino Building",
    "GYM": "Gymnasion (Sports Centre)",
    "GN": "Gymnasion (Sports Centre)",
    "RSC": "Radboud Sports Centre (Gymnasion / Sports Halls)",
    "ELN": "Aula / Erasmuslaan 9 Building",
    "UB": "University Library",
    "HG": "Huygens Building",
    "MERC": "Mercator Buildings (I/II/III)",
    "HAL": "Radboud Sports Centre (Gymnasion / Sports Halls)",
}

ROOM_PREFIX_MAP: Dict[str, str] = {
    "HG": "Huygens Building",
    "MM": "Maria Montessori Building",
    "SP": "Spinoza Building",
    "GR": "Grotius Building",
    "LIN": "Linnaeus Building",
    "CC": "Lecture Hall Complex",
    "GN": "Gymnasion (Sports Centre)",
    "GYM": "Gymnasion (Sports Centre)",
    "E": "Erasmus Building",
    "EOS": "Elinor Ostrom Building",
    "TVA": "Thomas van Aquino Building",
    "ELN": "Aula / Erasmuslaan 9 Building",
    "UB": "University Library",
    "RSC": "Radboud Sports Centre (Gymnasion / Sports Halls)",
    "MERC": "Mercator Buildings (I/II/III)",
    "HAL": "Radboud Sports Centre (Gymnasion / Sports Halls)",
}


class ValidateRouteForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_route_form"

    def _normalize(self, raw: Text, restrict_to: List[str] | None, context: Dict[str, Any]) -> Dict[str, Any]:
        llm = LLMController()
        logger.info(f"[normalize] raw={raw!r} restrict_to={restrict_to} context={context}")
        result = llm.normalize_building(raw=raw, restrict_to=restrict_to, context=context)
        logger.info(f"[normalize] result={result}")
        return result

    def _clear_disambiguation(self) -> Dict[Text, Any]:
        return {
            "followup_question": None,
            "disambiguation_candidates": [],
            "disambiguation_target_slot": None,
        }

    # ---- ORIGIN ----
    async def validate_origin_building(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict,
    ) -> Dict[Text, Any]:

        target = tracker.get_slot("disambiguation_target_slot")
        restrict_to = tracker.get_slot("disambiguation_candidates") or []
        in_disambig = (target == "origin_building" and len(restrict_to) > 0)

        context = {
            "target_slot": "origin_building",
            "other_slot": "destination_building",
            "other_value": tracker.get_slot("destination_building"),
            "previous_candidates": restrict_to if in_disambig else [],
            "last_followup": tracker.get_slot("followup_question"),
            "sender_id": tracker.sender_id,
        }

        result = self._normalize(
            str(slot_value) if slot_value is not None else None,
            restrict_to=restrict_to if in_disambig else None,
            context=context,
        )

        normalized = result.get("normalized")
        confidence = float(result.get("confidence", 0.0) or 0.0)

        if normalized and confidence >= CONFIDENCE_THRESHOLD:
            return {
                "origin_building": normalized,
                **self._clear_disambiguation(),
            }

        candidates = [c.get("name") for c in (result.get("candidates") or []) if c.get("name")]

        return {
            "origin_building": "",
            "followup_question": result.get("followup_question") or "",
            "disambiguation_candidates": candidates,
            "disambiguation_target_slot": "origin_building",
        }

    # ---- DESTINATION ----
    async def validate_destination_building(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict,
    ) -> Dict[Text, Any]:

        target = tracker.get_slot("disambiguation_target_slot")
        restrict_to = tracker.get_slot("disambiguation_candidates") or []
        in_disambig = (target == "destination_building" and len(restrict_to) > 0)

        context = {
            "target_slot": "destination_building",
            "other_slot": "origin_building",
            "other_value": tracker.get_slot("origin_building"),
            "previous_candidates": restrict_to if in_disambig else [],
            "last_followup": tracker.get_slot("followup_question"),
            "sender_id": tracker.sender_id,
        }

        result = self._normalize(
            str(slot_value) if slot_value is not None else None,
            restrict_to=restrict_to if in_disambig else None,
            context=context,
        )

        normalized = result.get("normalized")
        confidence = float(result.get("confidence", 0.0) or 0.0)

        if normalized and confidence >= CONFIDENCE_THRESHOLD:
            return {
                "destination_building": normalized,
                **self._clear_disambiguation(),
            }

        candidates = [c.get("name") for c in (result.get("candidates") or []) if c.get("name")]

        return {
            "destination_building": "",
            "followup_question": result.get("followup_question") or "",
            "disambiguation_candidates": candidates,
            "disambiguation_target_slot": "destination_building",
        }


class ActionRouteSummary(Action):
    def name(self) -> Text:
        return "action_route_summary"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        origin = tracker.get_slot("origin_building") or ""
        dest = tracker.get_slot("destination_building") or ""

        logger.info(f"[route_summary] origin={origin} dest={dest} sender={tracker.sender_id}")

        llm = LLMController()
        route_text = llm.generate_route(origin, dest)

        # De rule doet al: utter_route_summary -> "Got it: from 'X' to 'Y'."
        # Hier alleen de daadwerkelijke route + checkvraag.
        dispatcher.utter_message(text=route_text)
        dispatcher.utter_message(text="Did you manage to find it?")

        return []


class ActionExplainAbbreviation(Action):
    def name(self) -> Text:
        return "action_explain_abbreviation"

    def _extract_abbreviations(self, text: str) -> List[str]:
        upper = text.upper()

        # 1) Losse hoofdletter-woorden (EOS, MM, HG, ...)
        tokens = re.findall(r"\b[A-Z]{2,4}\b", upper)

        # 2) Roomcodes zoals HG00.616, MM 01.029, GR -1.175 -> pak prefix
        room_prefixes = []
        for match in re.findall(r"\b([A-Z]{1,4})\s*[-]?\d", upper):
            room_prefixes.append(match)

        candidates = set(tokens + room_prefixes)

        # 3) Filter op bekende afkortingen/prefixes
        found: List[str] = []
        for abbr in candidates:
            if abbr in ABBREVIATION_MAP or abbr in ROOM_PREFIX_MAP:
                found.append(abbr)

        return sorted(set(found))

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        text = (tracker.latest_message.get("text") or "").strip()
        logger.info(f"[abbr] text={text!r} sender={tracker.sender_id}")

        abbrs = self._extract_abbreviations(text)
        logger.info(f"[abbr] detected={abbrs}")

        explanations: List[str] = []

        for abbr in abbrs:

            if abbr in ROOM_PREFIX_MAP:
                explanations.append(
                    f"Abbreviations starting with {abbr} refer to rooms in the {ROOM_PREFIX_MAP[abbr]}."
                )
            elif abbr in ABBREVIATION_MAP:
                explanations.append(
                    f"{abbr} stands for the {ABBREVIATION_MAP[abbr]}."
                )

        if not explanations:
            msg = (
                "I'm not fully sure about that abbreviation yet. "
                "Share the full room code (like HG00.616 or MM 01.029) or abbreviation, "
                "and I'll find the correct building."
            )
        else:
            msg = " ".join(explanations)

        dispatcher.utter_message(text=msg)
        return []
