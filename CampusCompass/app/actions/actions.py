from __future__ import annotations
from typing import Any, Dict, List, Optional, Text

import logging

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from CampusCompass.app.llm.llmcontroller import LLMController

logger = logging.getLogger("campuscompass.actions")

# ---------------------------
# Lazy controller
# ---------------------------
_controller: Optional[LLMController] = None


def get_controller() -> LLMController:
    """
    Create controller only when needed.
    Prevents import-time crashes and keeps Rasa core isolated.
    """
    global _controller
    if _controller is None:
        _controller = LLMController()
    return _controller


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
        controller = get_controller()

        try:
            reply = controller.smalltalk_reply(user_text)
        except Exception as e:
            logger.error(f"[ActionSmalltalkLLM] smalltalk_reply failed: {e}")
            reply = "Got it 😄"

        dispatcher.utter_message(text=reply)
        return []
