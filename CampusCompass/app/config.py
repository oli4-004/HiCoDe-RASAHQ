from pathlib import Path
import os
import json
from dotenv import load_dotenv

# /CampusCompass/app/config.py
APP_ROOT = Path(__file__).resolve().parent        # .../CampusCompass/app
CC_ROOT = APP_ROOT.parent                         # .../CampusCompass

# Load secrets from project root
load_dotenv(CC_ROOT / ".env")

# OpenAI keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Knowledge Base (Radboud University)
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR = CC_ROOT / "docs"
RADBOUD_KB_PATH = KNOWLEDGE_DIR / "Radboud.json"

def load_radboud_kb(path: Path = RADBOUD_KB_PATH) -> dict:
    """Load the Radboud campus knowledge base from JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

try:
    RADBOUD_KB = load_radboud_kb()
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Radboud knowledge base not found at {RADBOUD_KB_PATH}. "
        "Expected file: knowledge/Radboud.json"
    ) from e
except json.JSONDecodeError as e:
    raise ValueError(
        f"Radboud knowledge base JSON is invalid: {RADBOUD_KB_PATH}"
    ) from e
