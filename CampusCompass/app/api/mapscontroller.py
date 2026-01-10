from typing import Optional, List, Dict, Any
import logging
import time
import html
import re
import json
import math
from pathlib import Path

import httpx

from CampusCompass.app.config import GOOGLE_MAPS_API_KEY, BUILDING_COORDS

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
        self.max_distance = 40

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

            # API-key uit de URL strippen voordat je logt
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
            self,
            origin_name: str,
            destination_name: str,
            mode: str = "walking"
    ) -> Dict[str, Any]:
        """
        Get walking directions between two building names.

        Returns a dict like:
        {
          "origin": "Huygens building",
          "destination": "Comenius building B",
          "origin_query": "...",
          "destination_query": "...",
          "duration_text": "9 mins",
          "distance_text": "700 m",
          "steps": ["Head north on ...", "Turn left onto ...", ...],
          "segments": [ {start_lat, start_lng, end_lat, end_lng}, ... ],
          "landmarks": [ {name, address, latitude, longitude, distance_m}, ... ],
          "step_landmarks": [ { ... } or None, aligned with steps ]
        }
        """
        if not self.api_key or not self._client:
            raise RuntimeError("MapsController not properly initialised or API key missing.")

        origin_query = self._to_maps_query(origin_name)
        destination_query = self._to_maps_query(destination_name)

        params = {
            "origin": origin_query,
            "destination": destination_query,
            "mode": mode,
            "language": "en",
            "region": "nl",
            "key": self.api_key,
        }

        if mode == "transit":
            params["departure_time"] = int(time.time())

        url = "https://maps.googleapis.com/maps/api/directions/json"

        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        # -------- extra logging: raw JSON van Google (geknipt) --------
        try:
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

        # Begin- en eindcoördinaten van de route + overview polyline voor kaartje
        start_loc = leg.get("start_location") or {}
        end_loc = leg.get("end_location") or {}
        overview_poly = (data["routes"][0].get("overview_polyline") or {}).get("points")

        static_map_url = self._build_static_map_url(
            start_lat=start_loc.get("lat"),
            start_lng=start_loc.get("lng"),
            end_lat=end_loc.get("lat"),
            end_lng=end_loc.get("lng"),
            encoded_path=overview_poly,
        )

        instruction_lines: List[str] = []
        segments: List[Dict[str, float]] = []

        step_modes: List[str] = []
        segments_per_step: List[Optional[Dict[str, float]]] = []

        for s in steps:
            step_mode = (s.get("travel_mode") or "").upper()
            step_modes.append(step_mode)

            html_instr = s.get("html_instructions") or ""
            dist_text = (s.get("distance") or {}).get("text") or ""
            plain = self._clean_html(html_instr)

            if dist_text:
                instruction_lines.append(f"{plain} ({dist_text})")
            else:
                instruction_lines.append(plain)

            start = s.get("start_location") or {}
            end = s.get("end_location") or {}

            try:
                seg = {
                    "start_lat": float(start.get("lat")),
                    "start_lng": float(start.get("lng")),
                    "end_lat": float(end.get("lat")),
                    "end_lng": float(end.get("lng")),
                }
                segments.append(seg)
                segments_per_step.append(seg)
            except (TypeError, ValueError):
                # keep alignment with steps
                segments_per_step.append(None)

        instruction_lines = [s for s in instruction_lines if s]

        buildings = BUILDING_COORDS or []

        segments_for_landmarks = segments
        if mode == "transit":
            segments_for_landmarks = [
                seg for seg, sm in zip(segments_per_step, step_modes)
                if seg is not None and sm == "WALKING"
            ]

        landmarks = self._find_buildings_along_route(
            segments=segments_for_landmarks,
            buildings=buildings,
            max_distance_m=self.max_distance,
        )

        raw_step_buildings: List[List[Dict[str, Any]]] = []

        for seg, sm in zip(segments_per_step, step_modes):
            # No segment => no buildings
            if seg is None:
                raw_step_buildings.append([])
                continue

            if mode == "transit" and sm != "WALKING":
                raw_step_buildings.append([])
                continue

            per_step = self._find_buildings_for_segment(
                segment=seg,
                buildings=buildings,
                max_distance_m=self.max_distance,
            )
            raw_step_buildings.append(per_step)

        origin_base = (origin_name.split(",")[0] or "").strip()
        dest_base = (destination_name.split(",")[0] or "").strip()

        def _find_building_record(base_name: str) -> Optional[Dict[str, Any]]:
            base_lower = (base_name or "").strip().lower()
            for b in buildings:
                b_name = (b.get("name") or "").split(",")[0].strip().lower()
                if b_name == base_lower:
                    return b
            return None

        origin_building = _find_building_record(origin_base)
        dest_building = _find_building_record(dest_base)

        # Verzamel alle namen die op dezelfde “site” liggen als origin/destination
        # (zelfde deur/gebouw, veel strenger dan de 40 m landmark-drempel)
        site_eps_m = 5.0

        def _names_same_site(ref_b: Optional[Dict[str, Any]]) -> set[str]:
            if not ref_b:
                return set()

            try:
                ref_lat = float(ref_b.get("latitude"))
                ref_lng = float(ref_b.get("longitude"))
            except (TypeError, ValueError):
                return set()

            names: set[str] = set()
            for b in buildings:
                try:
                    blat = float(b.get("latitude"))
                    blng = float(b.get("longitude"))
                except (TypeError, ValueError):
                    continue

                d = self._distance_between_points_m(ref_lat, ref_lng, blat, blng)
                if d <= site_eps_m:
                    base = (b.get("name") or "").split(",")[0].strip()
                    if base:
                        names.add(base)
            return names

        origin_site_names = _names_same_site(origin_building)
        dest_site_names = _names_same_site(dest_building)

        # Selectieregels:
        # - origin/destination én alles op dezelfde site nooit als landmark
        # - elk gebouw max 1x over de hele route
        # - per stap max 2 landmarks (eerste 2 na filtering)
        already_mentioned: set[str] = set()
        already_mentioned.update(origin_site_names)
        already_mentioned.update(dest_site_names)

        step_landmarks: List[List[str]] = []

        for per_step in raw_step_buildings:
            names_in_step: List[str] = []

            for b in per_step:
                name = (b.get("name") or "").strip()
                if not name:
                    continue
                base = name.split(",")[0].strip()

                # skip origin/destination-site én dingen die al genoemd zijn
                if base in already_mentioned:
                    continue
                if base in names_in_step:
                    continue

                names_in_step.append(base)

            if not names_in_step:
                step_landmarks.append([])
                continue

            chosen = names_in_step[:2]  # max 2 per stap
            step_landmarks.append(chosen)
            already_mentioned.update(chosen)

        # -------- extra logging --------
        try:
            self._maps_logger.info(
                "DIRECTIONS_PARSED origin=%r destination=%r duration=%r distance=%r steps=%s landmarks=%s step_landmarks=%s",
                origin_query,
                destination_query,
                duration_text,
                distance_text,
                instruction_lines,
                [
                    {
                        "name": lm.get("name"),
                        "distance_m": round(lm.get("distance_m", 0.0), 1),
                    }
                    for lm in landmarks
                ],
                [
                    per_step
                    for per_step in step_landmarks
                ],
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
            "segments": segments,
            "landmarks": landmarks,
            "step_landmarks": step_landmarks,
            "static_map_url": static_map_url,
            "travel_mode": mode,
        }

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------
    def _build_static_map_url(
        self,
        *,
        start_lat: Optional[float],
        start_lng: Optional[float],
        end_lat: Optional[float],
        end_lng: Optional[float],
        encoded_path: Optional[str],
    ) -> Optional[str]:
        """
        Bouw een Google Static Maps URL voor de route:
        - gebruikt de overview_polyline als path (als beschikbaar),
        - markeert start (A) en eind (B).
        """
        if not self.api_key:
            return None

        base = "https://maps.googleapis.com/maps/api/staticmap"
        parts: List[str] = [
            "size=640x400",
            "scale=2",
            "maptype=roadmap",
        ]

        if encoded_path:
            parts.append(f"path=enc:{encoded_path}")

        if start_lat is not None and start_lng is not None:
            parts.append(f"markers=color:green|label:A|{start_lat},{start_lng}")

        if end_lat is not None and end_lng is not None:
            parts.append(f"markers=color:red|label:B|{end_lat},{end_lng}")

        parts.append(f"key={self.api_key}")
        return base + "?" + "&".join(parts)

    def _to_maps_query(self, building_name: str) -> str:
        """
        Convert a (normalized) location name into a query string for Google Maps.

        Rules:
        - If it's already an address or contains 'Nijmegen' -> pass through.
        - If it contains 'station' -> NEVER append Radboud context (it's not a campus building).
        - If it's a known campus building (BUILDING_COORDS) -> append Radboud context.
        - Otherwise -> keep generic country context to help geocoding.
        """
        cleaned = (building_name or "").strip()
        if not cleaned:
            return "Radboud University Nijmegen"

        lower = cleaned.lower()

        if "nijmegen" in lower:
            return cleaned
        if "," in cleaned or re.search(r"\d", cleaned):
            return cleaned

        if "station" in lower:
            # Add minimal context; don't bias towards RU campus.
            return f"{cleaned}, Netherlands"

        base = cleaned.split(",")[0].strip().lower()
        is_known = False
        for b in (BUILDING_COORDS or []):
            bname = (b.get("name") or "").split(",")[0].strip().lower()
            if bname == base:
                is_known = True
                break

        if is_known:
            return f"{cleaned}, Radboud University Nijmegen, Nijmegen, Netherlands"

        return f"{cleaned}, Netherlands"

    def _clean_html(self, text: str) -> str:
        """
        Strip HTML tags and entities from Google Maps instructions.
        """
        text = html.unescape(text or "")
        return re.sub("<.*?>", "", text).strip()

    # ---------- geometrie helpers voor route-landmarks ----------

    def _point_to_segment_distance_m(
        self,
        lat: float,
        lng: float,
        s_lat: float,
        s_lng: float,
        e_lat: float,
        e_lng: float,
    ) -> float:
        """
        Benaderde kortste afstand tussen een punt (lat/lng) en een lijnsegment in meters.
        Voor campus-schaal is dit vlakke model prima.
        """
        # eenvoudige projectie naar meters
        lat0 = math.radians((s_lat + e_lat) / 2.0)
        kx = 111320 * math.cos(lat0)   # meter per graad longitude
        ky = 111132                    # meter per graad latitude

        px, py = ((lng - s_lng) * kx, (lat - s_lat) * ky)
        sx, sy = (0.0, 0.0)
        ex, ey = ((e_lng - s_lng) * kx, (e_lat - s_lat) * ky)

        vx, vy = (ex - sx, ey - sy)
        seg_len2 = vx * vx + vy * vy
        if seg_len2 == 0.0:
            # start == end
            return math.hypot(px - sx, py - sy)

        t = ((px - sx) * vx + (py - sy) * vy) / seg_len2
        t = max(0.0, min(1.0, t))  # clamp binnen segment
        projx, projy = (sx + t * vx, sy + t * vy)
        return math.hypot(px - projx, py - projy)

    def _distance_between_points_m(
        self,
        lat1: float,
        lng1: float,
        lat2: float,
        lng2: float,
    ) -> float:
        """
        Benaderde afstand tussen twee punten (lat/lng) in meters.
        Gebruikt dezelfde vlakke benadering als _point_to_segment_distance_m.
        """
        lat0 = math.radians((lat1 + lat2) / 2.0)
        kx = 111320 * math.cos(lat0)
        ky = 111132

        dx = (lng2 - lng1) * kx
        dy = (lat2 - lat1) * ky
        return math.hypot(dx, dy)


    def _find_buildings_along_route(
            self,
            segments: List[Dict[str, float]],
            buildings: List[Dict[str, Any]],
            max_distance_m: float,
    ) -> List[Dict[str, Any]]:
        """
        Kies gebouwen die echt in de buurt van de totale route liggen.
        Retourneert een kleine lijst, gesorteerd op minimale afstand tot een van de segmenten.
        """
        if not segments or not buildings:
            return []

        candidates: List[Dict[str, Any]] = []

        for b in buildings:
            blat = b.get("latitude")
            blng = b.get("longitude")
            if blat is None or blng is None:
                continue

            try:
                blat_f = float(blat)
                blng_f = float(blng)
            except (TypeError, ValueError):
                continue

            min_d: Optional[float] = None
            for seg in segments:
                d = self._point_to_segment_distance_m(
                    blat_f,
                    blng_f,
                    seg["start_lat"],
                    seg["start_lng"],
                    seg["end_lat"],
                    seg["end_lng"],
                )
                if min_d is None or d < min_d:
                    min_d = d

            if min_d is not None and min_d <= max_distance_m:
                c = dict(b)
                c["distance_m"] = float(min_d)
                candidates.append(c)

        candidates.sort(key=lambda x: x.get("distance_m", 1e9))
        # max 6 landmarks zodat de prompt niet explodeert
        return candidates[:6]

    def _find_buildings_for_segment(
            self,
            segment: Dict[str, float],
            buildings: List[Dict[str, Any]],
            max_distance_m: float,
    ) -> List[Dict[str, Any]]:
        """
        Kies ALLE gebouwen die in de buurt liggen van één segment.
        Retourneert een lijst gesorteerd op afstand tot dit segment.
        """
        if not segment or not buildings:
            return []

        candidates: List[Dict[str, Any]] = []

        for b in buildings:
            blat = b.get("latitude")
            blng = b.get("longitude")
            if blat is None or blng is None:
                continue

            try:
                blat_f = float(blat)
                blng_f = float(blng)
            except (TypeError, ValueError):
                continue

            d = self._point_to_segment_distance_m(
                blat_f,
                blng_f,
                segment["start_lat"],
                segment["start_lng"],
                segment["end_lat"],
                segment["end_lng"],
            )
            if d <= max_distance_m:
                c = dict(b)
                c["distance_m"] = float(d)
                candidates.append(c)

        candidates.sort(key=lambda x: x.get("distance_m", 1e9))
        return candidates

