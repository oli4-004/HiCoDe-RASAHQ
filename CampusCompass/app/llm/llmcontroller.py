from typing import Text, Dict, Any, List, Optional
import json
from openai import OpenAI
from CampusCompass.llm.config import OPENAI_API_KEY

class LLMController:
    """
    Controller for interaction with the OpenAI API.
    """
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY missing. Put it in your .env file or config"
            )
        self.client = OpenAI(api_key=self.api_key)

    def normalize_building(self, raw:Text, top_k = 3, restrict_to: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Returns a dict:
        {
          "normalized": "<best_match or 'UNKNOWN'>",
          "confidence": 0.0..1.0,
          "candidates": [{"name": "...", "confidence": 0.72, "reason": "..."}],
          "followup_question": "..."
        }
        """

        if raw is None or raw == "":
            return {
                "normalized": "",
                "confidence": 0.0,
                "candidates": [],
                "followup_question": "What building name or clear landmark do you see?",
            }
        system = (
            "You are an expert of the Radboud University in Nijmegen, the Netherlands helping lost students."
            "The point is to normalize their vague campus descriptions to a controlled list of building names."
            "Always answer in strict JSON. If unsure, return 'UNKNOWN'"
            "Users may use abbreviations or shorter variants, e.g. EOS or Elinor Ostrom instead of Elinor Ostromgebouw"
            "The user may also give a vague description at the Radboud University such as 'green building with flat roof'"
            "Provide top candidates with confidences (0..1) and one discriminating follow-up question."
            "If a 'restrict_to' list is provided, you MUST limit candidates and the final normalized choice to that subset."
        )
        user_payload = {
            "raw": raw,
            "top_k": top_k,
            "restrict_to": restrict_to or [],
            "output_schema": {
                "normalized": "string | 'UNKNOWN'",
                "confidence": "float [0..1]",
                "candidates": "list of {name:str, confidence:float, reason: str}",
                "followup_question": "string"
            }
        }

        try:
            response = self.client.chat.completions.create(
                model = "gpt-4o-mini",
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(user_payload)},
                ],
            )
            data = json.loads(response.choices[0].message.content)
        except Exception:
            return {
                "normalized": None,
                "confidence": 0.0,
                "candidates": [],
                "followup_question": "I am sorry, I did not understand. Could you describe again?",
            }
        return {
            "normalized": data.get("normalized"),
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "candidates": data.get("candidates", []) or [],
            "followup_question": data.get("followup_question") or "I am sorry, I did not understand. Could you describe again?"
        }