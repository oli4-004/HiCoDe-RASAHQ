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

    def _generate_route_text(self, origin: str, dest: str) -> str:
        """
        Generate a short walking route between two campus locations.
        Uses OpenAI if API key is available; otherwise falls back to simple heuristics.
        """
        if not origin or not dest:
            return (
                "I couldn't detect both your starting point and destination. "
                "Tell me where you are now and where you want to go, and I'll guide you."
            )

        # Simple special-case example (nice for your demo)
        if "Maria Montessori" in origin and "Elinor Ostrom" in dest:
            return (
                "From Maria Montessori, exit on the side facing the main bike path. "
                "Turn left and follow the path past Spinoza and Grotius towards the central lecture halls (CC). "
                "Keep going straight: the Elinor Ostrom Building (EOS) is the modern building directly next to the Lecture Hall Complex."
            )

        # If no OpenAI key: generic but usable fallback
        if not OPENAI_API_KEY:
            return (
                f"Walk from {origin} following the campus wayfinding signs in the direction of {dest}. "
                f"Stay on the main pedestrian/bike paths and follow the building signage once you're close."
            )

        # LLM-based routing
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)

            system = (
                "You are CampusCompass, a concise navigation assistant for Radboud University Nijmegen. "
                "Given an origin and destination building on this campus, respond in English with a clear, short "
                "walking route of 3–6 numbered steps. Use recognizable outdoor landmarks and building names. "
                "Stay under 120 words. Do not add extra apologies or disclaimers."
            )

            user = (
                f"User needs to walk from '{origin}' to '{dest}' on the Radboud University Nijmegen campus. "
                f"Give only the walking directions."
            )

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )

            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("Empty route from LLM")

            return text

        except Exception as e:
            logger.error(f"[route_summary] LLM route generation failed: {e}")
            return (
                f"Walk from {origin} towards {dest} following the main campus paths and official signage. "
                f"Once nearby, follow the building signs to the entrance of {dest}."
            )

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        origin = tracker.get_slot("origin_building")
        dest = tracker.get_slot("destination_building")

        logger.info(f"[route_summary] origin={origin} dest={dest} sender={tracker.sender_id}")

        route_text = self._generate_route_text(origin or "", dest or "")

        # "Got it: from 'X' to 'Y'." komt uit utter_route_summary
        # Hier doen we: route uitleg + checkvraag
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
