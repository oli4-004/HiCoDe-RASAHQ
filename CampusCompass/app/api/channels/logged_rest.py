from __future__ import annotations

from typing import Any, Dict, Optional, Text
from datetime import datetime
from pathlib import Path
import re
import uuid

import httpx

from CampusCompass.app.config import ROUTES_DIR

from sanic import Blueprint, response
from sanic.request import Request

from rasa.core.channels.channel import InputChannel, CollectingOutputChannel, UserMessage

from CampusCompass.app.api.chatlogger import LOGGER


class LoggedRestInput(InputChannel):
    """
    Drop-in vervanger voor de standaard 'rest' channel, maar dan met logging.
    """

    _STATICMAP_HOST = "https://maps.googleapis.com/maps/api/staticmap"

    def name(self) -> Text:
        return "rest"

    @staticmethod
    def _flatten_bot_message(msg: Dict[str, Any]) -> str:
        """
        Rasa REST antwoord kan text/image/buttons/custom bevatten.
        We maken er 1 log-string van (best effort).
        """
        parts = []

        text = (msg.get("text") or "").strip()
        if text:
            parts.append(text)

        image = (msg.get("image") or "").strip()
        if image:
            parts.append(f"[image] {image}")

        buttons = msg.get("buttons")
        if isinstance(buttons, list) and buttons:
            btns = []
            for b in buttons:
                if not isinstance(b, dict):
                    continue
                title = (b.get("title") or "").strip()
                url = (b.get("url") or "").strip()
                payload = (b.get("payload") or "").strip()
                if url:
                    btns.append(f"{title} -> {url}".strip())
                elif payload:
                    btns.append(f"{title} -> {payload}".strip())
                else:
                    btns.append(title)
            if btns:
                parts.append("[buttons] " + " | ".join([x for x in btns if x]))

        return "\n".join([p for p in parts if p]).strip()

    def blueprint(self, on_new_message):
        bp = Blueprint("logged_rest_webhook", __name__)

        @bp.get("/health")
        async def health(_: Request):
            return response.json({"status": "ok"})

        @bp.get("/routes/<filename:str>")
        async def serve_route(_: Request, filename: str):
            # simpele path traversal guard
            if ".." in filename or "/" in filename or "\\" in filename:
                return response.text("not found", status=404)

            path = ROUTES_DIR / filename
            if not path.exists() or not path.is_file():
                return response.text("not found", status=404)

            return await response.file(str(path))

        @bp.post("/webhook")
        async def webhook(request: Request):
            payload = request.json or {}

            sender_id = (
                payload.get("sender")
                or payload.get("sender_id")
                or payload.get("conversation_id")
                or "anonymous"
            )

            user_text = (
                payload.get("message")
                or payload.get("text")
                or ""
            )

            LOGGER.append(sender_id, "User", user_text)

            out = CollectingOutputChannel()
            metadata: Optional[Dict[str, Any]] = payload.get("metadata")

            msg = UserMessage(
                text=user_text,
                output_channel=out,
                sender_id=sender_id,
                input_channel=self.name(),
                metadata=metadata,
            )

            await on_new_message(msg)

            for m in out.messages:
                if not isinstance(m, dict):
                    continue
                img = m.get("image")
                if isinstance(img, str) and self._is_google_static_map(img):
                    fname = await self._download_static_map(img)
                    if fname:
                        m["image"] = f"/webhooks/rest/routes/{fname}"

            for m in out.messages:
                flat = self._flatten_bot_message(m if isinstance(m, dict) else {})
                if flat:
                    LOGGER.append(sender_id, "CampusCompass", flat)

            return response.json(out.messages)

        return bp

    def _is_google_static_map(self, url: str) -> bool:
        u = (url or "").strip()
        return u.startswith(self._STATICMAP_HOST)

    def _redact_key(self, url: str) -> str:
        return re.sub(r"(key=)[^&]+", r"\1[REDACTED]", url)

    async def _download_static_map(self, url: str) -> str | None:
        """
        Download static map image to ROUTES_DIR and return the filename.
        """
        ROUTES_DIR.mkdir(parents=True, exist_ok=True)

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(url)
                r.raise_for_status()

            ctype = (r.headers.get("content-type") or "").lower()
            ext = ".png"
            if "jpeg" in ctype or "jpg" in ctype:
                ext = ".jpg"

            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            fname = f"route_{stamp}_{uuid.uuid4().hex}{ext}"
            (ROUTES_DIR / fname).write_bytes(r.content)
            return fname

        except Exception as e:
            safe = self._redact_key(url)
            LOGGER.append("system", "CampusCompass", f"[routes] download failed for {safe}: {e}")
            return None

