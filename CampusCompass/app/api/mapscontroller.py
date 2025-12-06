from typing import Optional, List, Dict, Any
import logging
import time
import html
import re
import json
from pathlib import Path

import httpx

from CampusCompass.app.config import GOOGLE_MAPS_API_KEY

logger = logging.getLogger("campuscompass.maps")


class MapsController:
    """
    Small controller for talking to the Google Maps Directions API.
    Responsible ONLY for:
    - Building the HTTP request
    - Calling the API
    - Returning structured route data (duration + steps)

    It does NOT know anything about Rasa, slots, or LLMs.
    """

    def __init__(self) -> None:
        self.api_key = GOOGLE_MAPS_API_KEY

        if not self.api_key:
            logger.warning(
                "GOOGLE_MAPS_API_KEY missing – MapsController will not be able to call Google Directions API."
            )

        # ---------- HTTP + payload logging to logs/maps_calls.log ----------
        Path("logs").mkdir(exist_ok=True)
        maps_logger = logging.getLogger("campuscompass.maps.http")
        if not maps_logger.handlers:
            fh = logging.FileHandler("logs/maps_calls.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            maps_logger.addHandler(fh)
            maps_logger.setLevel(logging.INFO)
        self._maps_logger = maps_logger

        def log_request(request: httpx.Request):
            request.extensions["start_time"] = time.time()

            # ⚠️ Optioneel: API-key uit de URL strippen voordat je logt
            url_str = str(request.url)
            url_str = re.sub(r"(key=)[^&]+", r"\1[REDACTED]", url_str)

            self._maps_logger.info(
                "REQUEST %s %s headers=%s",
                request.method,
                url_str,
                dict(request.headers),
            )

        def log_response(response: httpx.Response):
            start = response.request.extensions.get("start_time")
            dur = (time.time() - start) if start else None
            self._maps_logger.info(
                "RESPONSE %s duration=%.3fs",
                response.status_code,
                dur if dur is not None else -1.0,
            )

        self._client: Optional[httpx.Client] = httpx.Client(
            timeout=httpx.Timeout(8.0, connect=3.0),
            event_hooks={"request": [log_request], "response": [log_response]},
        )

    # ------------------------------------------------------------------
    # PUBLIC: walking directions
    # ------------------------------------------------------------------

    def get_walking_directions(
        self, origin_name: str, destination_name: str
    ) -> Dict[str, Any]:
        """
        Get walking directions between two building names.

        Returns a dict like:
        {
          "origin": "Huygens building",
          "destination": "Comenius building B",
          "origin_query": "Huygens building, Radboud University Nijmegen",
          "destination_query": "Comenius building B, Radboud University Nijmegen",
          "duration_text": "9 mins",
          "distance_text": "700 m",
          "steps": ["Head north on ...", "Turn left onto ...", ...]
        }
        """
        if not self.api_key or not self._client:
            raise RuntimeError("MapsController not properly initialised or API key missing.")

        origin_query = self._to_maps_query(origin_name)
        destination_query = self._to_maps_query(destination_name)

        params = {
            "origin": origin_query,
            "destination": destination_query,
            "mode": "walking",
            "language": "en",
            "region": "nl",
            "key": self.api_key,
        }

        url = "https://maps.googleapis.com/maps/api/directions/json"

        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        # -------- extra logging: raw JSON van Google (geknipt) --------
        try:
            # alleen eerste route loggen en max ~4000 chars om het leesbaar te houden
            trimmed = {
                "status": data.get("status"),
                "geocoded_waypoints": data.get("geocoded_waypoints"),
                "routes": data.get("routes", [])[:1],
            }
            self._maps_logger.info(
                "DIRECTIONS_RAW origin=%r destination=%r payload=%s",
                origin_query,
                destination_query,
                json.dumps(trimmed, ensure_ascii=False)[:4000],
            )
        except Exception as e:
            self._maps_logger.warning("Failed to log raw directions JSON: %s", e)

        if data.get("status") != "OK":
            raise RuntimeError(f"Directions API status {data.get('status')}")

        leg = data["routes"][0]["legs"][0]
        duration_text = leg.get("duration", {}).get("text", "")
        distance_text = leg.get("distance", {}).get("text", "")
        steps = leg.get("steps", [])

        instruction_lines: List[str] = []
        for s in steps:
            html_instr = s.get("html_instructions") or ""
            instruction_lines.append(self._clean_html(html_instr))

        # Limit steps so the prompt stays reasonable
        instruction_lines = [s for s in instruction_lines if s][:12]

        # -------- extra logging: wat we naar de LLM sturen als 'steps' --------
        try:
            self._maps_logger.info(
                "DIRECTIONS_PARSED origin=%r destination=%r duration=%r distance=%r steps=%s",
                origin_query,
                destination_query,
                duration_text,
                distance_text,
                instruction_lines,
            )
        except Exception as e:
            self._maps_logger.warning("Failed to log parsed directions: %s", e)

        return {
            "origin": origin_name,
            "destination": destination_name,
            "origin_query": origin_query,
            "destination_query": destination_query,
            "duration_text": duration_text,
            "distance_text": distance_text,
            "steps": instruction_lines,
        }

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _to_maps_query(self, building_name: str) -> str:
        """
        Convert a (normalized) building name into a query string for Google Maps.

        - Als de naam al een volledig adres met 'Nijmegen' bevat, stuur die
          1-op-1 door.
        - Alleen bij vage namen voeg je 'Radboud University Nijmegen' toe.
        """
        cleaned = (building_name or "").strip()
        if not cleaned:
            return "Radboud University Nijmegen"

        lower = cleaned.lower()

        # Als het er al uitziet als adres in Nijmegen → niet meer aanklooien.
        if "nijmegen" in lower:
            return cleaned

        # Anders wat extra context zodat Maps snapt dat het om de campus gaat.
        return f"{cleaned}, Radboud University Nijmegen, Nijmegen, Netherlands"

    def _clean_html(self, text: str) -> str:
        """
        Strip HTML tags and entities from Google Maps instructions.
        """
        text = html.unescape(text or "")
        return re.sub("<.*?>", "", text).strip()
