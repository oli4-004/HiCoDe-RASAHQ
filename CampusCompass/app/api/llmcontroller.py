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
        self._building_docs_text: Optional[str] = None
        logger.info("[LLMController] initialised, api_key_present=%s", bool(self.api_key))

        # ---------- OpenAI call logging to logs/openai_calls.log ----------
        Path("logs").mkdir(exist_ok=True)
        openai_logger = logging.getLogger("campuscompass.openai")
        if not openai_logger.handlers:
            fh = logging.FileHandler("logs/llm_debug.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            openai_logger.addHandler(fh)
            openai_logger.setLevel(logging.INFO)
        self._openai_logger = openai_logger

        def _redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
            safe = dict(headers)
            for k in list(safe.keys()):
                if k.lower() in (
                        "authorization",
                        "proxy-authorization",
                        "cookie",
                        "set-cookie",
                        "x-api-key",
                        "x-openai-api-key",
                ):
                    safe[k] = "***REDACTED***"
            return safe

        def log_request(request: httpx.Request):
            request.extensions["start_time"] = time.time()

            # geen query-params loggen, voor de zekerheid
            safe_url = request.url.copy_with(query=None)
            safe_headers = _redact_headers(request.headers)

            self._openai_logger.info(
                "REQUEST %s %s headers=%s",
                request.method,
                safe_url,
                dict(safe_headers),
            )

        def log_response(response: httpx.Response):
            start = response.request.extensions.get("start_time")
            dur = (time.time() - start) if start else None

            rid = response.headers.get("x-request-id")
            remaining = response.headers.get("x-ratelimit-remaining-requests")

            self._openai_logger.info(
                "RESPONSE %s duration=%.3fs request_id=%s remaining=%s",
                response.status_code,
                dur or -1.0,
                rid,
                remaining,
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

            try:
                self._openai_logger.info("OUTPUT: %s", text)
            except Exception as log_err:
                logger.warning(f"[LLMController._chat_completion] logging failed: {log_err}")

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

    # ---------------------------------------------------------------------
    # LOADING DOCS AS KNOWLEDGE BASE
    # ---------------------------------------------------------------------
    def _load_building_docs(self) -> str:
        """
        Load building documentation from the docs folder once, and cache it.

        Assumes files like docs/kb_building_*.txt, but you can adjust the pattern
        if your filenames differ.
        """
        if self._building_docs_text is not None:
            return self._building_docs_text

        docs_dir = Path("docs")
        parts: List[str] = []

        if docs_dir.exists():
            for path in docs_dir.glob("kb_*.txt"):
                try:
                    parts.append(path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"[LLMController._load_building_docs] Failed to read {path}: {e}")
        else:
            logger.warning("[LLMController._load_building_docs] docs directory does not exist")

        # Join and truncate so we do not send an enormous prompt
        combined = "\n\n".join(parts)
        self._building_docs_text = combined[:12000]  # keep it reasonably small
        return self._building_docs_text

    # ---------------------------------------------------------------------
    # ROUTE: NORMALIZE TRAVEL MODE HINT → GOOGLE MAPS MODE
    # ---------------------------------------------------------------------
    def normalize_travel_mode(self, raw: str) -> str:
        """
        Map a free-form user phrase about how they want to travel
        to exactly one of the Google Maps modes:

            - 'walking'
            - 'bicycling'
            - 'driving'

        If the hint is empty or ambiguous, default to 'walking'.
        """

        raw_clean = _clean(raw)

        # If there is no LLM client, always fall back to walking.
        if not self.client:
            return "walking"

        # Empty or whitespace-only → walking
        if not raw_clean:
            return "walking"

        system_msg = (
            "You are CampusCompass, a campus assistant at Radboud University Nijmegen.\n"
            "Your task is to map a free-form user phrase about how they want to travel\n"
            "to exactly ONE of these three strings:\n"
            "  - walking\n"
            "  - bicycling\n"
            "  - driving\n"
            "\n"
            "Rules:\n"
            "- If the user clearly wants to go on foot, answer 'walking'.\n"
            "- If the user clearly wants to go by bike, answer 'bicycling'.\n"
            "- If the user clearly wants to go by car, answer 'driving'.\n"
            "- If the user mentions bus, train, tram, metro or public transport in general,\n"
            "  or if their preference is unclear or mixed, default to 'walking'.\n"
            "- Answer with exactly one word: 'walking', 'bicycling', or 'driving'.\n"
            "- Do NOT add any explanations, punctuation, or extra words."
        )

        user_msg = f"User travel mode phrase: '{raw_clean}'"

        text = self._chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model="gpt-4o-mini",
            temperature=0.0,
            max_completion_tokens=4,
        )

        if not text:
            return "walking"

        mode = text.strip().split()[0].lower()

        if mode in {"walking", "bicycling", "driving"}:
            return mode

        # Safety fallback
        return "walking"


    # ---------------------------------------------------------------------
    # ROUTE: NORMALIZE BUILDING / VENUE NAME (BUILDINGS + VENUES FROM DOCS)
    # ---------------------------------------------------------------------

    def normalize_building_name(self, raw: str) -> str:
        """
        Use the LLM + CAMPUS_DOCS to normalize a raw reference
        (e.g. 'HG', 'MM', 'EOS', 'bar van tandheelkunde', 'Aesculaaf')
        into a single line that Google Maps can geocode:

            <Name>, <street> <house number>, <postcode> Nijmegen

        The LLM must ONLY use addresses that actually appear in CAMPUS_DOCS.
        If nothing matches, return the raw input unchanged.
        """
        raw = _clean(raw)
        if not raw:
            return raw

        if not self.client:
            return raw

        # Volledige campusdocs (gebouwen, venues, abbreviaties, etc.)
        docs_text = self._load_building_docs()

        system_msg = (
            "You are CampusCompass, a campus assistant for Radboud University Nijmegen.\n"
            "You receive a raw location reference (often an abbreviation or nickname)\n"
            "and CAMPUS_DOCS containing building and venue information.\n"
            "\n"
            "CAMPUS_DOCS includes:\n"
            "- Building names with their exact postal addresses.\n"
            "- Venues inside buildings (bars, cafés, canteens, study spaces, prayer rooms, etc.).\n"
            "- Abbreviations such as MM, EOS, HG, CC, UB, BM that refer to buildings.\n"
            "- Descriptions that link venues to buildings (e.g. a bar inside Dentistry).\n"
            "\n"
            "YOUR TASK\n"
            "- Interpret the raw reference as EITHER:\n"
            "  (a) a building, OR\n"
            "  (b) a venue inside a building,\n"
            "  based ONLY on CAMPUS_DOCS.\n"
            "- Then output ONE line in a format that Google Maps can geocode.\n"
            "\n"
            "OUTPUT FORMAT (VERY STRICT)\n"
            "- If the best match is a BUILDING:\n"
            "    <Canonical building name>, <street name> <house number>, <postcode> Nijmegen\n"
            "  Example:\n"
            "    'Maria Montessori building, Thomas van Aquinostraat 4, 6525 GD Nijmegen'\n"
            "\n"
            "- If the best match is a VENUE INSIDE a building:\n"
            "    <Venue name>, <street name> <house number>, <postcode> Nijmegen\n"
            "  Examples (ONLY if CAMPUS_DOCS actually contains them):\n"
            "    'Café de Aesculaaf, Geert Grooteplein Noord 21, 6525 EZ Nijmegen'\n"
            "    ''t Dappenglaasje, Philips van Leydenlaan 25, 6525 EX Nijmegen'\n"
            "    'Grand Café de Iris, Thomas van Aquinostraat 4, 6525 GD Nijmegen'\n"
            "\n"
            "RULES\n"
            "- Use the exact name and address as they appear in CAMPUS_DOCS.\n"
            "- Do NOT add 'Netherlands' after 'Nijmegen'. Stop after 'Nijmegen'.\n"
            "- Do NOT add any extra words, labels, or explanations before or after the line.\n"
            "- Do NOT invent addresses: every street, house number and postcode you use must\n"
            "  be explicitly present somewhere in CAMPUS_DOCS.\n"
            "\n"
            "MATCHING HINTS (ONLY if consistent with CAMPUS_DOCS)\n"
            "- 'MM' usually refers to the Maria Montessori building.\n"
            "- 'HG' usually refers to the Huygens building.\n"
            "- 'EOS' usually refers to the Elinor Ostrom building.\n"
            "- 'CC' usually refers to the Collegezalencomplex.\n"
            "- 'UB' usually refers to the University Library.\n"
            "- 'Aesculaaf' usually refers to the café/venue 'Café de Aesculaaf'.\n"
            "- Phrases like 'bar van tandheelkunde' usually refer to the bar inside Dentistry,\n"
            "  e.g. ''t Dappenglaasje' if CAMPUS_DOCS describes it.\n"
            "\n"
            "SOURCE-OF-TRUTH RULE\n"
            "- CAMPUS_DOCS is the ONLY source of truth.\n"
            "- Ignore any external knowledge about Nijmegen or Radboud University.\n"
            "- If you cannot find a reliable match in CAMPUS_DOCS, do NOT guess.\n"
            "\n"
            "FALLBACK\n"
            "- If you truly cannot find ANY matching building or venue in CAMPUS_DOCS,\n"
            "  respond with EXACTLY the original raw input, unchanged, with no extra words.\n"
        )

        user_msg = (
            f"Raw location reference: '{raw}'.\n\n"
            "CAMPUS_DOCS (buildings + streets + abbreviations + venues/spaces):\n"
            f"{docs_text}"
        )

        text = self._chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model="gpt-4o-mini",
            temperature=0.0,
            max_completion_tokens=80,
        )

        if not text:
            return raw

        # Neem alleen de eerste regel van de LLM-output
        normalized = (text or "").strip().splitlines()[0].strip()
        lower_norm = normalized.lower()

        # Sanity-checks: meta / lege antwoorden → fallback
        if (
            not normalized
            or "campus_docs" in lower_norm
            or "raw input unchanged" in lower_norm
            or "not listed" in lower_norm
        ):
            return raw

        return normalized

    # ---------------------------------------------------------------------
    # ROUTE: TURN STRUCTURED DATA INTO A NICE MESSAGE
    # ---------------------------------------------------------------------
    def format_route_description(
            self, origin_name: str, destination_name: str, route: Dict[str, Any], mode: str = "walking"
    ) -> str:
        """
        Take structured route data from MapsController and turn it into
        a short, friendly explanation for the user.

        The LLM may ONLY:
        - summarise the existing steps, in the same order,
        - use buildings that the backend has already marked as step_landmarks.
        """

        route = route or {}
        duration = route.get("duration_text") or ""
        steps = route.get("steps") or []
        step_landmarks = route.get("step_landmarks") or []

        # -------- Fallback if OpenAI is not available --------
        fallback_lines: List[str] = []
        if duration:
            fallback_lines.append(
                f"The route from {origin_name} to {destination_name} takes about {duration}."
            )
        else:
            fallback_lines.append(
                f"Here is a route from {origin_name} to {destination_name}."
            )

        if steps:
            fallback_lines.append("Rough directions:")
            for i, step in enumerate(steps[:8], start=1):
                fallback_lines.append(f"{i}. {step}")

        fallback_lines.append("This route is based on Google Maps.")
        fallback = "\n".join(fallback_lines)

        if not self.client:
            return fallback

        system_msg = (
            "You are CampusCompass, a campus navigation assistant at Radboud University Nijmegen.\n"
            "You receive ROUTE_DATA with:\n"
            "  • origin: ORIGIN building name,\n"
            "  • destination: DESTINATION building name,\n"
            "  • duration_text: total travel time,\n"
            "  • steps: an ordered list of text instructions from Google Maps. Each step string may end\n"
            "    with a distance in parentheses such as \"(70 m)\" or \"(0.3 km)\".\n"
            "  • step_landmarks: a list aligned with steps; step_landmarks[i] is a LIST of\n"
            "    0–2 building names that SHOULD be mentioned for step i.\n"
            "  • travel_mode: either walking, bicycling or driving. The mode of transport. Use this when talking about duration (on foot, by bike, by car)"
            "\n"
            "YOUR TASK\n"
            "Turn ROUTE_DATA into a clear, easy-to-follow explanation for a student on campus.\n"
            "You MUST stay faithful to ROUTE_DATA.\n"
            "\n"
            "OUTPUT FORMAT\n"
            "  1. First line: \"It is about X minutes from ORIGIN to DESTINATION.\" "
            "(use ROUTE_DATA.duration_text; if missing, say \"a few minutes\").\n"
            "  2. Then write the walking instructions as numbered lines:\n"
            "       1. ...\n"
            "       2. ...\n"
            "       3. ...\n"
            "     Rules for these lines:\n"
            "       - Each numbered step MUST be on its own line.\n"
            "       - You MUST NOT put more than one numbered step on the same line.\n"
            "       - Keep each line short and readable.\n"
            "  3. Keep the whole answer concise (roughly up to 100 words total).\n"
            "\n"
            "ROUTE FIDELITY (VERY IMPORTANT)\n"
            "  - Treat ROUTE_DATA.steps as ground truth for the path.\n"
            "  - Preserve ALL steps in the same order. You may combine two small consecutive turns\n"
            "    into one line, but you MUST NOT skip, reorder or invent steps.\n"
            "  - Only use street names that actually appear in ROUTE_DATA.steps.\n"
            "  - Do NOT invent extra streets or squares.\n"
            "\n"
            "DISTANCE RULES (PER STEP)\n"
            "  - Many step strings end with a distance in parentheses, such as \"(70 m)\" or \"(0.3 km)\".\n"
            "  - Whenever a step contains such a distance, you MUST express it explicitly in your text, e.g.:\n"
            "      • \"Walk about 70 m along Heyendaalseweg …\"\n"
            "      • \"Follow Willem Nuyenslaan for about 0.3 km, then …\"\n"
            "  - Do NOT drop these distances; they are important for orientation.\n"
            "\n"
            "LANDMARK RULES (PER STEP)\n"
            "  - For each index i, step_landmarks[i] is ALREADY a final list of building names to mention\n"
            "    for that step (0, 1 or 2 names, already filtered).\n"
            "  - If step_landmarks[i] is empty, you MUST NOT mention any building for that step.\n"
            "  - If step_landmarks[i] contains one name, mention that building naturally in the same line\n"
            "    as step i.\n"
            "  - If step_landmarks[i] contains two names, mention both buildings naturally for that step.\n"
            "  - Do NOT add, remove or swap landmarks: you may only use building names that appear inside\n"
            "    step_landmarks, plus ORIGIN and DESTINATION.\n"
            "  - Use landmarks as something the user walks past or along, for example:\n"
            "      • \"Walk about 70 m on Thomas van Aquinostraat, passing [LANDMARK], then …\"\n"
            "      • \"Turn left onto Willem Nuyenslaan and follow it for 0.3 km, with [LANDMARK 1]\n"
            "         and [LANDMARK 2] along the way.\"\n"
            "  - Mention ORIGIN only in the first line (\"from ORIGIN to DESTINATION\").\n"
            "  - Mention DESTINATION only in the first or final line as the endpoint; do NOT claim that\n"
            "    the user \"passes\" ORIGIN or DESTINATION mid-route unless the step text itself says so.\n"
            "  - You MUST NOT introduce any other buildings than ORIGIN, DESTINATION, or those listed\n"
            "    in step_landmarks.\n"
            "  - You may shorten a landmark to its building name (e.g. \"Grotius building\"), but do not\n"
            "    invent new names.\n"
            "  - You MUST NOT say that a building is \"on your left\" or \"on your right\" unless the\n"
            "    corresponding step text explicitly mentions that same building and its side.\n"
            "  - You MUST NOT mention \"crossing the road\", bridges, tunnels or any other actions that are\n"
            "    not explicitly present in ROUTE_DATA.steps.\n"
            "\n"
            "STYLE RESTRICTIONS\n"
            "  - Answer in English.\n"
            "  - Be purely practical: describe what the user should do, how far they walk for each step,\n"
            "    and which streets/buildings they pass.\n"
            "  - Do NOT add emotional or touristic commentary.\n"
            "  - Do NOT mention JSON, APIs or internal tools.\n"
        )

        payload = {
            "origin": origin_name,
            "destination": destination_name,
            "duration_text": duration,
            "distance_text": route.get("distance_text"),
            "steps": steps,
            "step_landmarks": step_landmarks,
            "travel_mode": mode
        }

        user_msg = (
            "Here is the route data.\n\n"
            "ROUTE_DATA:\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        text = self._chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model="gpt-4o-mini",
            temperature=0.0,
            max_completion_tokens=360,
        )

        if not text:
            return fallback
        return text.strip()
