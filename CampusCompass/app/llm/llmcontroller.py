from typing import Text, Optional, List, Dict, Any
import json
import re
import logging
import time
import httpx
from pathlib import Path
from openai import OpenAI
from CampusCompass.app.config import OPENAI_API_KEY

logger = logging.getLogger("campuscompass")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


class LLMController:
    """
    Lichte controller voor smalltalk-replies, met OpenAI logging.
    Bevat een generieke _chat_completion helper voor latere functionaliteiten.
    """

    def __init__(self):
        self.api_key = OPENAI_API_KEY

        # ---------- OpenAI call logging to logs/openai_calls.log ----------
        Path("logs").mkdir(exist_ok=True)
        openai_logger = logging.getLogger("campuscompass.openai")
        if not openai_logger.handlers:
            fh = logging.FileHandler("logs/openai_calls.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            openai_logger.addHandler(fh)
            openai_logger.setLevel(logging.INFO)
        self._openai_logger = openai_logger

        def log_request(request: httpx.Request):
            request.extensions["start_time"] = time.time()
            self._openai_logger.info(
                f"REQUEST {request.method} {request.url} headers={dict(request.headers)}"
            )

        def log_response(response: httpx.Response):
            start = response.request.extensions.get("start_time")
            dur = (time.time() - start) if start else None
            rid = response.headers.get("x-request-id")
            self._openai_logger.info(
                f"RESPONSE {response.status_code} duration={dur:.3f}s request_id={rid}"
            )

        http_client = httpx.Client(
            timeout=httpx.Timeout(8.0, connect=3.0),
            event_hooks={"request": [log_request], "response": [log_response]},
        )

        # Fail-fast: korte timeouts + max 1 retry om Rasa REST timeouts te voorkomen
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=http_client.timeout,
                max_retries=1,
                http_client=http_client,
            )
        else:
            self.client = None
            logger.warning("OPENAI_API_KEY missing: running in smalltalk-fallback mode.")

    # ---------------------------------------------------------------------
    # GENERIC CHAT COMPLETION HELPER
    # ---------------------------------------------------------------------

    def _chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.4,
        max_completion_tokens: int = 60,
    ) -> Optional[str]:
        """
        Dunne wrapper rond self.client.chat.completions.create.

        Retourneert de gegenereerde tekst (str) of None bij fouten.
        """
        if not self.client:
            return None

        try:
            resp = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                messages=messages,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text or None
        except Exception as e:
            logger.error(f"[LLMController._chat_completion] failed: {e}")
            return None

    # ---------------------------------------------------------------------
    # SMALLTALK REPLY
    # ---------------------------------------------------------------------

    def smalltalk_reply(self, user_text: str) -> str:
        """
        Generate ONE short, friendly sentence that reacts to off-topic smalltalk,
        without answering campus questions.
        """

        user_text = _clean(user_text)
        fallback_reply = "Got it 😄"

        if not self.client:
            return fallback_reply

        system_msg = (
            "You are CampusCompass, a friendly but focused campus assistant at "
            "Radboud University Nijmegen.\n"
            "The user is making smalltalk or a light remark that is not a campus task.\n"
            "Reply with ONE short, friendly sentence that acknowledges what the user said.\n"
            "Rules:\n"
            "- Do NOT answer campus questions (no directions, buildings, or opening hours).\n"
            "- Do NOT give factual information about the campus.\n"
            "- Do NOT ask follow-up questions.\n"
            "- You may use at most one emoji.\n"
            "- Keep it casual and kind, max ~20 words."
        )

        text = self._chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_text},
            ],
            model="gpt-4o-mini",
            temperature=0.4,
            max_completion_tokens=60,
        )

        if not text:
            return fallback_reply
        return text
