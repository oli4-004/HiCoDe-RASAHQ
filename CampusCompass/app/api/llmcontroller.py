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
        self._kb_lookup_cache: Optional[List[Dict[str, Any]]] = None
        self._kb_lookup_path = Path("docs") / "lookup_kb.json"
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
    def _load_building_docs(self, query_text: Optional[str] = None) -> str:
        """
        Load building documentation from the docs folder.

        - If query_text is provided: use docs/lookup_kb.json + a small LLM call to select
          the 5 most relevant kb files, then concatenate ONLY those file contents.
        - If anything fails: fall back to the old behavior (load all kb_*.txt once and cache it).
        """

        # Fallback: all docs
        def _fallback_load_all() -> str:
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

            combined = "\n\n".join(parts)
            self._building_docs_text = combined
            return self._building_docs_text

        query_text = _clean(query_text or "")
        if not query_text:
            return _fallback_load_all()

        docs_dir = Path("docs")
        if not docs_dir.exists():
            logger.warning("[LLMController._load_building_docs] docs directory does not exist")
            return _fallback_load_all()

        # If OpenAI client is missing, no point in LLM-based selection → fallback
        if not self.client:
            return _fallback_load_all()

        # 1) Load lookup_kb.json once
        try:
            if self._kb_lookup_cache is None:
                if not self._kb_lookup_path.exists():
                    logger.warning(f"[LLMController._load_building_docs] lookup not found: {self._kb_lookup_path}")
                    return _fallback_load_all()

                raw_lookup = self._kb_lookup_path.read_text(encoding="utf-8")
                data = json.loads(raw_lookup)

                # Accept either {"docs":[...]} or direct list [...]
                if isinstance(data, dict) and isinstance(data.get("docs"), list):
                    docs_list = data["docs"]
                elif isinstance(data, list):
                    docs_list = data
                else:
                    logger.warning("[LLMController._load_building_docs] lookup_kb.json has unexpected shape")
                    return _fallback_load_all()

                # Minimal validation
                cleaned: List[Dict[str, Any]] = []
                for item in docs_list:
                    if not isinstance(item, dict):
                        continue
                    doc_id = item.get("doc_id")
                    filename = item.get("filename")
                    if not isinstance(doc_id, int) or not isinstance(filename, str) or not filename.strip():
                        continue
                    cleaned.append(item)

                if not cleaned:
                    logger.warning("[LLMController._load_building_docs] lookup has no valid entries")
                    return _fallback_load_all()

                self._kb_lookup_cache = cleaned

            lookup_docs = self._kb_lookup_cache or []
        except Exception as e:
            logger.warning(f"[LLMController._load_building_docs] lookup load failed: {e}")
            return _fallback_load_all()

        # 2) Ask LLM for top 5 doc_ids (hallucination-resistant via validation)
        # Keep payload small: sample entities/keywords
        candidates = []
        allowed_ids = []
        for item in lookup_docs:
            doc_id = item.get("doc_id")
            filename = item.get("filename")
            if not isinstance(doc_id, int) or not isinstance(filename, str):
                continue

            allowed_ids.append(doc_id)

            entities = item.get("entities") or []
            keywords = item.get("keywords") or []
            summary = item.get("summary") or ""

            if not isinstance(entities, list):
                entities = []
            if not isinstance(keywords, list):
                keywords = []

            candidates.append({
                "doc_id": doc_id,
                "filename": filename,
                "summary": summary,
                "keywords": keywords,
                "entities_sample": entities,
            })

        if not candidates:
            return _fallback_load_all()

        system_msg = (
            "You are a file selector for a campus chatbot.\n"
            "Given a user query and a list of CANDIDATES (each with doc_id, filename, summary, keywords, entities_sample),\n"
            "select the 5 most relevant doc_id values.\n"
            "\n"
            "STRICT RULES:\n"
            "- You MUST ONLY output doc_id values that appear in the provided CANDIDATES.\n"
            "- Output MUST be valid JSON on a single line, exactly in this shape:\n"
            "  {\"doc_ids\": [1,2,3,4,5]}\n"
            "- doc_ids must be unique, integers, max length 5.\n"
            "- If fewer than 5 are relevant, return fewer.\n"
            "- Do NOT include any extra keys or text.\n"
        )

        user_msg = (
            f"USER_QUERY:\n{query_text}\n\n"
            f"CANDIDATES:\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
            "Return JSON now."
        )

        selected_ids: List[int] = []
        try:
            raw = self._chat_completion(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                model="gpt-4o-mini",
                temperature=0.0,
                max_completion_tokens=80,
            )

            if raw:
                parsed = json.loads(raw)
                doc_ids = parsed.get("doc_ids", [])
                logger.info(
                    f"[LLMController._load_building_docs] LLM suggested doc_ids={doc_ids} for query={query_text!r}")
                if isinstance(doc_ids, list):
                    # Validate strictly: ints only, in allowed set, unique, max 5
                    allowed_set = set(allowed_ids)
                    out: List[int] = []
                    for x in doc_ids:
                        if isinstance(x, int) and x in allowed_set and x not in out:
                            out.append(x)
                        if len(out) >= 5:
                            break
                    selected_ids = out
        except Exception as e:
            logger.warning(f"[LLMController._load_building_docs] LLM selection failed: {e}")
            selected_ids = []

        # If LLM gives nothing usable -> fallback to old behavior (expensive but reliable)
        if not selected_ids:
            return _fallback_load_all()

        # 3) Read and concatenate the selected files
        id_to_filename: Dict[int, str] = {}
        for item in lookup_docs:
            if isinstance(item, dict) and isinstance(item.get("doc_id"), int) and isinstance(item.get("filename"), str):
                id_to_filename[item["doc_id"]] = item["filename"]

        parts: List[str] = []
        for doc_id in selected_ids[:5]:
            filename = id_to_filename.get(doc_id)
            if not filename:
                continue

            path = (docs_dir / filename).resolve()
            # safety: ensure resolved path is inside docs_dir
            try:
                if docs_dir.resolve() not in path.parents and path != docs_dir.resolve():
                    logger.warning(f"[LLMController._load_building_docs] unsafe path skipped: {path}")
                    continue
            except Exception:
                continue

            if not path.exists():
                logger.warning(f"[LLMController._load_building_docs] selected file missing: {path}")
                continue

            try:
                parts.append(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"[LLMController._load_building_docs] Failed to read selected {path}: {e}")

        # If reading selected files fails -> fallback to old behavior
        combined = "\n\n".join([p for p in parts if (p or "").strip()])
        if not combined.strip():
            return _fallback_load_all()

        return combined

    # ---------------------------------------------------------------------
    # ABBREVIATIONS: EXTRACT SINGLE BUILDING / ROOM CODE
    # ---------------------------------------------------------------------
    def extract_abbreviation(self, user_text: str) -> str:
        """
        From a single user message, extract at most ONE relevant
        Radboud building / room code or abbreviation that the user is
        asking about.

        Examples of what to extract:
        - MM, SP, CC, GR, LIN, E, GN, EOS, TvA, COMA, COMB, ELN, HG
        - Room-like codes such as:
          - MM 01.029, SP A/B -1.55, E.01.15, GR 0.100, etc.

        Rules:
        - Return exactly the code/string as it appears in the user text,
          trimmed of leading/trailing spaces.
        - If there are multiple plausible codes, choose the ONE that is
          most central to the user's question.
        - If there is NO realistic building/room code in the message,
          return an empty string.

        The snippet may contain multiple user and assistant messages.
        You must choose the most recent explicit USER mention of a
        realistic building/room code.

        This is used to prefill the abbreviation_raw slot before we ask
        the user anything in abbreviation_flow.
        """

        snippet = _clean(user_text)
        if not snippet:
            return ""
        if not self.client:
            return ""

        system_msg = (
            "You are CampusCompass, a campus assistant at Radboud University Nijmegen.\n"
            "You receive a SHORT CONVERSATION SNIPPET with multiple lines, each starting\n"
            "with 'User:' or 'Assistant:'. Your job is to extract at most ONE building/\n"
            "room abbreviation or code that the user is currently asking about.\n"
            "\n"
            "VALID EXAMPLES:\n"
            "- MM, SP, CC, GR, LIN, E, GN, EOS, TvA, COMA, COMB, ELN, HG\n"
            "- Room codes like 'MM 01.029', 'SP A/B -1.55', 'E.01.15', 'GR 0.100'.\n"
            "\n"
            "RULES:\n"
            "- Only consider USER lines when deciding which code to output.\n"
            "- The snippet is in chronological order. If multiple codes appear in\n"
            "  different USER lines, you MUST choose the most recent one that the\n"
            "  user is clearly confused about or asking to explain.\n"
            "- Return EXACTLY the substring for that code, as it appears in the\n"
            "  user's text (no extra words).\n"
            "- If there is NO realistic building/room code in any USER line,\n"
            "  return an empty string.\n"
            "\n"
            "IMPORTANT OUTPUT RULES:\n"
            "- If you found a code, output ONLY that code, nothing else.\n"
            "- If you did NOT find any code, output exactly an empty string.\n"
        )

        user_msg = f"CONVERSATION_SNIPPET:\n{snippet}"

        text = self._chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model="gpt-4o-mini",
            temperature=0.0,
            max_completion_tokens=16,
        )

        if not text:
            return ""

        # We expect either a single code or an empty string
        code = text.strip()
        # Very small safety: if the LLM mistakenly adds spaces/newlines
        # around an empty string, normalise that.
        if not code:
            return ""

        return code

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
    # ROUTE: CLASSIFY TRAVEL MODE HINT FOR FLOW ROUTING
    # ---------------------------------------------------------------------
    def classify_travel_mode_hint(self, user_text: str) -> str:
        """
        Look at a short conversation snippet and classify the intended
        travel mode into one of:

            - 'walk'
            - 'bike'
            - 'car'
            - 'public_transport'
            - 'unknown'

        The snippet may contain several user and assistant lines. You must
        prioritise the most recent user message that clearly talks about
        travelling or how they want to go somewhere.
        """

        snippet = _clean(user_text)
        if not snippet:
            return "unknown"
        if not self.client:
            return "unknown"

        system_msg = (
            "You are CampusCompass, a campus assistant at Radboud University Nijmegen.\n"
            "You receive a SHORT CONVERSATION SNIPPET with multiple lines, each starting\n"
            "with 'User:' or 'Assistant:'. Your job is to classify the intended main\n"
            "travel mode into exactly ONE of these labels:\n"
            "\n"
            "  - walk\n"
            "  - bike\n"
            "  - car\n"
            "  - public_transport\n"
            "  - unknown\n"
            "\n"
            "RULES:\n"
            "- The snippet is in chronological order (top = earliest, bottom = latest).\n"
            "- If multiple travel modes are mentioned, you MUST prioritise the most\n"
            "  recent USER line that clearly talks about how they want to travel.\n"
            "- Words like 'on foot', 'walking', 'by foot' → 'walk'.\n"
            "- Words like 'by bike', 'bicycle', 'cycling', 'fiets' → 'bike'.\n"
            "- Words like 'by car', 'with my car', 'drive' → 'car'.\n"
            "- Words like 'public transport', 'bus', 'train', 'OV', 'NS', 'Arriva',\n"
            "  'station', 'bus stop' → 'public_transport'.\n"
            "- If the user only says vague speed words such as 'as fast as possible',\n"
            "  'as quickly as possible', without specifying walk/bike/car/bus/train,\n"
            "  this is NOT a clear mode → answer 'unknown'.\n"
            "- If you cannot find a clear mode in any USER line, answer 'unknown'.\n"
            "\n"
            "IMPORTANT OUTPUT RULES:\n"
            "- Answer with EXACTLY ONE word: 'walk', 'bike', 'car', 'public_transport',\n"
            "  or 'unknown'. No extra text.\n"
            "- If the user only uses vague speed words like 'as fast as possible', 'as quickly as possible', 'ASAP',\n"
            " and does NOT explicitly say walk/bike/car/bus/train/public transport,you MUST output 'unknown'."
        )

        user_msg = f"CONVERSATION_SNIPPET:\n{snippet}"

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
            return "unknown"

        label = text.strip().split()[0].lower()

        if label in {"walk", "bike", "car", "public_transport", "unknown"}:
            return label

        # Safety fallback
        return "unknown"


    # ---------------------------------------------------------------------
    # ROUTE: EXTRACT ROUTE
    # ---------------------------------------------------------------------
    def extract_route_entities(
            self,
            user_text: str,
            prev_source: Optional[str] = None,
            prev_destination: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Try to extract a SOURCE and/or DESTINATION location from a short
        conversation history snippet.

        The snippet may contain several user and assistant messages, prefixed
        with "User:" and "Assistant:". You must focus on the most recent user
        question about travelling, but you may use earlier lines as context.

        You may also receive:
          - prev_source: origin of the last completed route in this conversation.
          - prev_destination: destination of the last completed route.

        Returns:
            { "source": "<raw span from user text> or None",
              "destination": "<raw span from user text> or None }
        """

        snippet = _clean(user_text)
        if not snippet or not self.client:
            return {"source": None, "destination": None}

        system_msg = (
            "You are CampusCompass, a campus navigation assistant at Radboud University Nijmegen.\n"
            "You receive a SHORT CONVERSATION SNIPPET with multiple lines, each starting with\n"
            "'User:' or 'Assistant:'. Your job is to decide whether this snippet contains:\n"
            "- a starting point (SOURCE), and/or\n"
            "- a destination (DESTINATION).\n"
            "\n"
            "IMPORTANT:\n"
            "- The snippet is in chronological order (top = earliest, bottom = latest).\n"
            "- If multiple locations are mentioned, you MUST prioritise the MOST RECENT. THIS IS HIGHLY IMPORTANT\n"
            "  user question about travelling.\n"
            "- Later user mentions override earlier ones. For example, if an earlier line\n"
            "  says 'User: I am at Huygens' and a later line says 'User: I am now at\n"
            "  Heyendaal station', then SOURCE must be 'Heyendaal station'.\n"
            "\n"
            "LOCATIONS YOU MAY EXTRACT:\n"
            "- campus buildings (e.g. 'EOS', 'Huygens', 'Maria Montessori', 'Berchmanianum'),\n"
            "- room or building codes (e.g. 'MM00.010', 'HG00.616'),\n"
            "- nearby stations or stops (e.g. 'Heyendaal station', 'Nijmegen Central Station').\n"
            "- You SHOULD prefer phrases that appear in USER lines.\n"
            "- EXCEPTION: if the MOST RECENT user message uses a referring expression like\n"
            "  'there', 'that place', 'that building', or similar, and it clearly points to\n"
            "  a location name that appears only in the immediately preceding ASSISTANT line,\n"
            "  you MAY copy that assistant phrase for SOURCE or DESTINATION.\n"
            "\n"
            "HOW TO DECIDE SOURCE VS DESTINATION:\n"
            "- Look for patterns like: 'from X to Y', 'to Y from X', 'I'm at X and need\n"
            "  to go to Y', 'How can I get there?'.\n"
            "- Use earlier assistant messages to resolve pronouns like 'there' or 'that\n"
            "  building', but the final SOURCE/DESTINATION text should normally come from\n"
            "  user wording when possible, or from the clearly referenced assistant phrase\n"
            "  in the exception described above.\n"
            "- If the user only clearly names a place they want to go to (e.g. 'How do I\n"
            "  get to EOS?'), then DESTINATION = that place, SOURCE = null.\n"
            "- If they clearly say where they are (e.g. 'I'm at Heyendaal station and I need\n"
            "  to go to Berchmanianum.'), then SOURCE = where they are, DESTINATION = target.\n"
            "\n"
            "PREVIOUS ROUTE CONTEXT (if provided):\n"
            "- You may also receive PREV_SOURCE and PREV_DESTINATION values.\n"
            "- They represent the origin and destination of the LAST COMPLETED ROUTE.\n"
            "- If the MOST RECENT user message is something like:\n"
            "    'How do I get back?', 'I want to go back', 'And back?',\n"
            "  and it does NOT explicitly name new buildings or stations, then you MUST:\n"
            "    * set SOURCE      = PREV_DESTINATION\n"
            "    * set DESTINATION = PREV_SOURCE\n"
            "- If the latest user message clearly names new buildings (for example\n"
            "  'Now from EOS to MM'), you MUST ignore PREV_SOURCE/PREV_DESTINATION and\n"
            "  use the new locations instead.\n"
            "\n"
            "OUTPUT FORMAT (VERY STRICT):\n"
            "- Respond with ONE line of valid JSON, no extra text.\n"
            "- The JSON MUST have exactly these two keys: 'source' and 'destination'.\n"
            "- Each value is either a string (exact phrase taken from the USER wording or,\n"
            "  in the exception case, the clearly referenced ASSISTANT phrase) or null.\n"
            "\n"
            "EXAMPLE:\n"
            "User: Oh no! I lost my student card. I'm currently at Heyendaal station.\n"
            "Assistant: You can request a new student card at the Central Student Desk in the Berchmanianum.\n"
            "User: How can I get there as quickly as possible?\n"
            "-> {\"source\": \"Heyendaal station\", \"destination\": \"Berchmanianum\"}\n"
        )

        prev_src_str = prev_source if prev_source else "null"
        prev_dst_str = prev_destination if prev_destination else "null"

        user_msg = (
            "CONVERSATION_SNIPPET:\n"
            f"{snippet}\n\n"
            "PREVIOUS_ROUTE_CONTEXT:\n"
            f"PREV_SOURCE: {prev_src_str}\n"
            f"PREV_DESTINATION: {prev_dst_str}\n"
        )

        raw = self._chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model="gpt-4o-mini",
            temperature=0.0,
            max_completion_tokens=80,
        )

        if not raw:
            return {"source": None, "destination": None}

        raw = raw.strip()
        try:
            data = json.loads(raw)
            source = data.get("source")
            dest = data.get("destination")
            if isinstance(source, str):
                source = source.strip() or None
            else:
                source = None
            if isinstance(dest, str):
                dest = dest.strip() or None
            else:
                dest = None
            return {"source": source, "destination": dest}
        except Exception as e:
            logger.error(f"[LLMController.extract_route_entities] JSON parse failed: {e} raw={raw!r}")
            return {"source": None, "destination": None}


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
        docs_text = self._load_building_docs(raw)

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
            "    0–2 building names that SHOULD be mentioned for that step.\n"
            "  • travel_mode: either walking, bicycling or driving. The mode of transport. Use this when talking about duration (on foot, by bike, by car)\n"
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
            "       - You SHOULD normally use about 3–5 numbered instructions total.\n"
            "         If needed to avoid skipping a street or important turn, you MAY\n"
            "         use up to 7 numbered instructions.\n"
            "  3. Keep the whole answer concise (roughly up to 100 words total).\n"
            "\n"
            "ROUTE FIDELITY (VERY IMPORTANT)\n"
            "  - Treat ROUTE_DATA.steps as ground truth for the path.\n"
            "  - You MUST mention every distinct street name that appears in ROUTE_DATA.steps\n"
            "    at least once somewhere in your numbered instructions. You MUST NOT merge\n"
            "    steps in such a way that a street name disappears completely.\n"
            "  - Keep the overall order of the route, but you MUST MERGE several consecutive\n"
            "    steps into a single numbered instruction when they keep the user on the\n"
            "    same street or are only small adjustments (e.g. \"continue\", \"slight right\",\n"
            "    or \"stay on\" the same road).\n"
            "  - You MUST NOT produce more than 2 numbered instructions for the same street\n"
            "    name in the entire answer. If ROUTE_DATA.steps contains 3 or more steps on\n"
            "    the same street, you MUST combine them into 1 or 2 clearer instructions.\n"
            "  - In particular, avoid giving the user 3–5 separate steps in a row that all\n"
            "    say \"stay on\" or \"continue\" on the same street name; combine them into\n"
            "    one or two clearer instructions instead.\n"
            "  - Whenever two or more consecutive steps in ROUTE_DATA use the same street\n"
            "    name, you MUST make it explicit in your text that they refer to the same\n"
            "    street, for example by saying \"follow this same street\" or \"keep following\n"
            "    this street\".\n"
            "  - Do NOT change the overall path: do not invent new steps and do not skip\n"
            "    important turns.\n"
            "  - Only use street names that actually appear in ROUTE_DATA.steps.\n"
            "  - Do NOT invent extra streets or squares.\n"
            "\n"
            "DISTANCE RULES (PER STEP)\n"
            "  - Many step strings end with a distance in parentheses, such as \"(70 m)\" or \"(0.3 km)\".\n"
            "  - For each numbered instruction, mention at most ONE approximate distance and\n"
            "    keep it simple and rounded (e.g. \"about 50 m\", \"about 200 m\", \"about 500 m\").\n"
            "  - If a distance is given in kilometres but is less than 1.0 km (e.g. \"0.2 km\"),\n"
            "    convert it to metres and round it (e.g. \"about 200 m\").\n"
            "  - Avoid very precise numbers like \"68 m\" or \"32 m\"; use rounded values instead,\n"
            "    or phrases such as \"a few hundred metres\".\n"
            "  - When you merge several consecutive steps on the same street, you MUST refer\n"
            "    to their combined distance in an approximate way (e.g. \"for about 300 m\" or\n"
            "    \"for a few hundred metres\") instead of listing every small distance separately.\n"
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
            "  - If several consecutive instructions keep the user on the same street, avoid repeating\n"
            "    the full street name each time; you MUST explicitly signal that it is the same street,\n"
            "    for example with phrases like \"this street\" or \"the same road\".\n"
            "  - Do NOT add emotional or touristic commentary.\n"
            "  - Do NOT mention JSON, APIs or internal tools.\n"
            "\n"
            "EXAMPLE 1 (GEERT GROOTEPLEIN ZUID)\n"
            "  - Suppose ROUTE_DATA.steps for a part of the route look like this:\n"
            "      • \"Turn left onto Geert Grooteplein Zuid (68 m)\"\n"
            "      • \"Continue straight to stay on Geert Grooteplein Zuid (0.2 km)\"\n"
            "      • \"Slight right to stay on Geert Grooteplein Zuid (32 m)\"\n"
            "      • \"Turn right to stay on Geert Grooteplein Zuid (0.1 km)\"\n"
            "      • \"Turn right toward Geert Grooteplein Noord (53 m)\"\n"
            "      • \"Turn left onto Geert Grooteplein NoordDestination will be on the right (5 m)\"\n"
            "    A GOOD transformation of this part is:\n"
            "      \"Turn left onto Geert Grooteplein Zuid and follow this same street past Forum for\n"
            "       a few hundred metres.\"\n"
            "      \"Then turn right toward Geert Grooteplein Noord and walk the last few metres;\n"
            "       the destination will be on your right.\"\n"
            "    This way all Geert Grooteplein steps are covered in 1–2 instructions and\n"
            "    Geert Grooteplein Zuid and Geert Grooteplein Noord are both mentioned.\n"
            "\n"
            "EXAMPLE 2 (MARIA MONTESSORI → CAFÉ DE AESCULAAF)\n"
            "  - Suppose ROUTE_DATA.steps contain:\n"
            "      • Thomas van Aquinostraat → Max Weberpad → Spinozapad → Pieter Rabuspad\n"
            "      • René Descartesdreef → Van Beverwijcklaan → Geert Grooteplein Zuid\n"
            "      • Geert Grooteplein Noord\n"
            "    A GOOD transformation of this entire route is:\n"
            "      1. \"Head east on Thomas van Aquinostraat for about 100 m, then turn left onto\n"
            "         Max Weberpad and left onto Spinozapad, continuing onto Pieter Rabuspad for\n"
            "         a few hundred metres, passing Collegezalencomplex (CC) and Paviljoen.\"\n"
            "      2. \"Turn right onto René Descartesdreef and then left onto Van Beverwijcklaan;\n"
            "         follow this same street for about 300 m, passing Villa Oud Heyendael and Forum.\"\n"
            "      3. \"Continue onto Geert Grooteplein Zuid for about 200 m, then turn right toward\n"
            "         Geert Grooteplein Noord and walk the last few metres; Café de Aesculaaf will\n"
            "         be on your right.\"\n"
            "    Notice that every distinct street name from ROUTE_DATA.steps appears at least once,\n"
            "    but tiny steps are merged into clearer combined instructions.\n"
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

    # ---------------------------------------------------------------------
    # Q&A: ANSWER QUESTION USING DOCS (NO FACTCHECK/REWRITE PIPELINE)
    # ---------------------------------------------------------------------
    def answer_question_from_docs(self, question_text: str, history_snippet: Optional[str] = None) -> str:
        """
        Answer the user's question using ONLY the selected CAMPUS_DOCS, but return a JSON object:

          {"status":"ANSWER","answer":"..."}
          {"status":"NEED_INFO","question":"..."}
          {"status":"NOT_IN_DOCS"}

        Notes:
        - The LLM MUST NOT ask the user directly as plain text; it must output JSON only.
        - The caller (Rasa action/flow) decides what to do with NEED_INFO / NOT_IN_DOCS.
        """

        def _as_json(obj: dict) -> str:
            # Always return valid JSON string
            return json.dumps(obj, ensure_ascii=False)

        q = _clean(question_text)
        snippet = _clean(history_snippet or "")

        if not q:
            return _as_json({"status": "NEED_INFO", "question": "Could you rephrase your question?"})

        if not self.client:
            # Treat as not answerable right now (caller can show generic fallback)
            return _as_json({"status": "NOT_IN_DOCS"})

        # Heuristic: if the question is pronoun-heavy / context-dependent, use snippet for retrieval too
        q_lower = q.lower()
        needs_context = bool(re.search(r"\b(it|there|that|this|here|back|again|those|them)\b", q_lower)) or len(q) < 18
        retrieval_query = f"{snippet}\n\nLATEST_QUESTION:\n{q}" if (snippet and needs_context) else q

        docs_text = self._load_building_docs(retrieval_query)

        system_msg = (
            "You are CampusCompass, a campus assistant at Radboud University Nijmegen.\n"
            "You must respond with EXACTLY ONE JSON object and nothing else.\n"
            "\n"
            "Output format (choose exactly one):\n"
            "1) If you can answer confidently from CAMPUS_DOCS:\n"
            "   {\"status\":\"ANSWER\",\"answer\":\"<final answer>\"}\n"
            "2) If the user did not provide enough information (missing/ambiguous), but an answer might exist:\n"
            "   {\"status\":\"NEED_INFO\",\"question\":\"<one short clarifying question>\"}\n"
            "3) If the answer cannot be found in CAMPUS_DOCS:\n"
            "   {\"status\":\"NOT_IN_DOCS\"}\n"
            "\n"
            "Hard rules:\n"
            "- Do NOT include any keys other than those shown above.\n"
            "- Do NOT output markdown, code fences, explanations, or extra text.\n"
            "- Do NOT mention or refer to sources, documents, files, context, knowledge base, or 'available information'.\n"
            "- Do NOT invent opening hours, addresses, room numbers, prices, line numbers, or other precise facts.\n"
            "- If multiple interpretations exist and you cannot disambiguate from the text, use NEED_INFO.\n"
            "- If the question is about live/real-time info (e.g., current departures), and it is not explicitly described, use NOT_IN_DOCS.\n"
            "\n"
            "Answer formatting rules (ONLY for the 'answer' string):\n"
            "- Plain text only: no markdown, no bullets, no bold, no headings.\n"
            "- NEVER start a line with '-', '*', '•', or a number like '1.' or '1)'.\n"
            "- If the context contains bullet points, rewrite them into ONE line separated by semicolons.\n"
            "- Newlines are allowed.\n"
            "- Keep it concise and directly useful.\n"
            "- Answer in English.\n"
        )
        user_msg = (
            f"CONVERSATION_CONTEXT (may help resolve pronouns):\n{snippet}\n\n"
            f"USER_QUESTION:\n{q}\n\n"
            f"CAMPUS_DOCS:\n{docs_text}\n\n"
            "Return the JSON object now."
        )

        text = self._chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model="gpt-4o-mini",
            temperature=0.0,
            max_completion_tokens=260,
        )

        raw = (text or "").strip()
        if not raw:
            return _as_json({"status": "NOT_IN_DOCS"})

        # Robust JSON extraction (in case the model wraps it accidentally)
        candidate = raw
        if not (candidate.startswith("{") and candidate.endswith("}")):
            m = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            if m:
                candidate = m.group(0).strip()

        try:
            obj = json.loads(candidate)
        except Exception:
            return _as_json({"status": "NOT_IN_DOCS"})

        # Validate strict schema
        status = (obj.get("status") or "").strip()
        if status == "ANSWER":
            if set(obj.keys()) != {"status", "answer"}:
                return _as_json({"status": "NOT_IN_DOCS"})
            answer = (obj.get("answer") or "").strip()
            if not answer:
                return _as_json({"status": "NOT_IN_DOCS"})
            return _as_json({"status": "ANSWER", "answer": answer})

        if status == "NEED_INFO":
            if set(obj.keys()) != {"status", "question"}:
                return _as_json({"status": "NOT_IN_DOCS"})
            question = (obj.get("question") or "").strip()
            if not question:
                return _as_json({"status": "NOT_IN_DOCS"})
            return _as_json({"status": "NEED_INFO", "question": question})

        if status == "NOT_IN_DOCS":
            if set(obj.keys()) != {"status"}:
                return _as_json({"status": "NOT_IN_DOCS"})
            return _as_json({"status": "NOT_IN_DOCS"})

        return _as_json({"status": "NOT_IN_DOCS"})
