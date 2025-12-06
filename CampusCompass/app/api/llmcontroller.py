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
    # ROUTE: NORMALIZE BUILDING NAME  (STRICT: NAME + ADDRESS FROM DOCS ONLY)
    # ---------------------------------------------------------------------
    def normalize_building_name(self, raw: str) -> str:
        """
        Use the LLM + campus docs to normalize a raw building reference
        (e.g. 'HG', 'MM', 'EOS', 'CC') into a canonical building name + postal
        address that Google Maps can geocode.

        IMPORTANT:
        - The ONLY allowed source is CAMPUS_DOCS (buildings, streets, abbreviations).
        - If nothing in CAMPUS_DOCS matches, return the raw input unchanged.
        """
        raw = _clean(raw)
        if not raw:
            return raw

        if not self.client:
            return raw

        docs_text = self._load_building_docs()

        system_msg = (
            "You are CampusCompass, a campus assistant for Radboud University Nijmegen.\n"
            "You receive a raw building reference (often an abbreviation) and CAMPUS_DOCS.\n"
            "CAMPUS_DOCS contains:\n"
            "- Building names with their exact postal addresses.\n"
            "- Street descriptions with building lists.\n"
            "- Abbreviations like MM, EOS, HG, CC, UB, BM that refer to buildings.\n"
            "\n"
            "YOUR JOB\n"
            "- Find the single best matching *building* for the raw reference using ONLY CAMPUS_DOCS.\n"
            "- Then output its canonical building name plus the exact address as written in CAMPUS_DOCS.\n"
            "\n"
            "ALLOWED BUILDING EXAMPLES (PATTERN, NOT TO GUESS FROM MEMORY)\n"
            "- 'EOS'  -> 'Elinor Ostrom building, Heyendaalseweg 141, 6525 AJ Nijmegen'\n"
            "- 'MM'   -> 'Maria Montessori building, Thomas van Aquinostraat 4, 6525 GD Nijmegen'\n"
            "- 'HG'   -> 'Huygens building, Heyendaalseweg 135, 6525 AJ Nijmegen'\n"
            "- 'UB'   -> 'University Library, Erasmuslaan 36, 6525 GG Nijmegen'\n"
            "- 'CC'   -> 'Collegezalencomplex, Mercatorpad 1, 6525 HS Nijmegen'\n"
            "- 'BM'   -> 'Berchmanianum, Houtlaan 4, 6525 XZ Nijmegen'\n"
            "These are only correct IF and BECAUSE CAMPUS_DOCS contains those exact lines.\n"
            "\n"
            "SOURCE-OF-TRUTH RULE (VERY STRICT)\n"
            "- CAMPUS_DOCS is the ONLY truth. Ignore your own knowledge of Nijmegen completely.\n"
            "- You are NOT allowed to invent an address or modify one, even if you think you know it.\n"
            "- The building name and address you output MUST be assembled entirely from text that occurs in CAMPUS_DOCS.\n"
            "- If CAMPUS_DOCS shows a building with a specific street, house number and postcode,\n"
            "  you MUST use that exact combination, character for character.\n"
            "\n"
            "MATCHING RULES\n"
            "- First, check the abbreviations section: if it defines 'MM', 'EOS', 'HG', 'CC', 'UB', 'BM', etc.,\n"
            "  link the abbreviation to the corresponding building name in CAMPUS_DOCS.\n"
            "- Then, find that building name in the street/building lists to obtain the full address.\n"
            "- If multiple entries in CAMPUS_DOCS mention the same building, prefer the one with the most complete address\n"
            "  (street, house number, postcode).\n"
            "\n"
            "OUTPUT FORMAT (STRICT)\n"
            "- If you find a match: output EXACTLY one line of the form:\n"
            "  <Canonical building name>, <street name> <house number>, <postcode> Nijmegen\n"
            "  Example: 'Elinor Ostrom building, Heyendaalseweg 141, 6525 AJ Nijmegen'.\n"
            "- Do NOT add 'Netherlands' or anything after 'Nijmegen'.\n"
            "- Do NOT add explanations, labels, or extra text.\n"
            "\n"
            "FALLBACK\n"
            "- ONLY if you truly cannot find any matching building in CAMPUS_DOCS,\n"
            "  output the raw input unchanged, as a single line.\n"
        )

        user_msg = (
            f"Raw building reference: '{raw}'.\n\n"
            "CAMPUS_DOCS (buildings + streets + addresses + abbreviations):\n"
            f"{docs_text}"
        )

        text = self._chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model="gpt-4o-mini",
            temperature=0.0,
            max_completion_tokens=60,
        )

        if not text:
            return raw

        normalized = text.strip().splitlines()[0].strip()

        # Fallback: als het model meldt dat het gebouw niet in CAMPUS_DOCS staat,
        # kies een zinnige default voor veelvoorkomende namen.
        lower_norm = normalized.lower()
        lower_raw = raw.lower()

        if "not listed in campus_docs" in lower_norm:
            if lower_raw.startswith("comenius"):
                # gebruik één concrete Comenius-variant i.p.v. de foutmelding
                return "Comenius building B"
            if lower_raw.startswith("mercator"):
                return "Mercator I"
            # anders: val gewoon terug op de ruwe input
            return raw

        return normalized or raw

    # ---------------------------------------------------------------------
    # ROUTE: TURN STRUCTURED DATA INTO A NICE MESSAGE
    # ---------------------------------------------------------------------
    def format_route_description(
            self, origin_name: str, destination_name: str, route: Dict[str, Any]
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
                f"On foot, the route from {origin_name} to {destination_name} takes about {duration}."
            )
        else:
            fallback_lines.append(
                f"Here is a walking route from {origin_name} to {destination_name}."
            )

        if steps:
            fallback_lines.append("Rough walking directions:")
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
            "  • duration_text: total walking time,\n"
            "  • steps: an ordered list of text instructions from Google Maps,\n"
            "  • step_landmarks: a list aligned with steps; step_landmarks[i] is a LIST of\n"
            "    0–2 building names that SHOULD be mentioned for step i.\n"
            "\n"
            "YOUR TASK\n"
            "Turn ROUTE_DATA into ONE clear answer for a student on campus.\n"
            "You MUST stay faithful to ROUTE_DATA.\n"
            "\n"
            "OUTPUT FORMAT\n"
            "  1. First sentence: \"It is about X minutes on foot from ORIGIN to DESTINATION.\" "
            "(use ROUTE_DATA.duration_text; if missing, say \"a few minutes\").\n"
            "  2. Then 1–3 short sentences that describe the route.\n"
            "  3. No bullet points or numbering. Maximum ~80 words in total.\n"
            "\n"
            "ROUTE FIDELITY (VERY IMPORTANT)\n"
            "  - Treat ROUTE_DATA.steps as ground truth for the path.\n"
            "  - Preserve ALL steps in the same order. You may combine two small consecutive turns "
            "    into one sentence, but you MUST NOT skip, reorder or invent steps.\n"
            "  - Only use street names that actually appear in ROUTE_DATA.steps.\n"
            "  - Do NOT invent extra streets or squares.\n"
            "\n"
            "LANDMARK RULES (PER STEP)\n"
            "  - For each index i, step_landmarks[i] is ALREADY a final list of building names to mention\n"
            "    for that step (0, 1 or 2 names, already filtered).\n"
            "  - If step_landmarks[i] is empty, you MUST NOT mention any building for that step.\n"
            "  - If step_landmarks[i] contains one name, mention that building naturally in the same sentence\n"
            "    as step i.\n"
            "  - If step_landmarks[i] contains two names, mention both buildings naturally for that step.\n"
            "  - Do NOT add, remove or swap landmarks: you may only use building names that appear inside\n"
            "    step_landmarks, plus ORIGIN and DESTINATION.\n"
            "  - Use landmarks as something the user walks past or along, for example:\n"
            "      • \"Head west on Thomas van Aquinostraat, passing [LANDMARK], then ...\"\n"
            "      • \"Turn left onto Willem Nuyenslaan, with [LANDMARK 1] and [LANDMARK 2] along the way, then ...\"\n"
            "  - Mention ORIGIN only in the first sentence (\"from ORIGIN to DESTINATION\").\n"
            "  - Mention DESTINATION only in the first or final sentence as the endpoint; do NOT claim that\n"
            "    the user \"passes\" ORIGIN or DESTINATION mid-route unless the step text itself says so.\n"
            "  - You MUST NOT introduce any other buildings than ORIGIN, DESTINATION, or those listed\n"
            "    in step_landmarks.\n"
            "  - You may shorten a landmark to its building name (e.g. \"Grotius building\"), but do not invent new names.\n"
            "  - Do NOT guess whether a building is on the left or right unless the step text itself contains that side.\n"
            "\n"
            "STYLE RESTRICTIONS\n"
            "  - Answer in English.\n"
            "  - Be purely practical: describe what the user should do and which streets/buildings they pass.\n"
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
            temperature=0.0,  # zo min mogelijk creatief / random
            max_completion_tokens=180,
        )

        if not text:
            return fallback
        return text.strip()







