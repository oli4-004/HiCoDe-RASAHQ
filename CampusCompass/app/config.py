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

# Google Maps key from .env
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# ------------------------------------------------------------------
# Building coordinates (for landmarks langs de route)
# ------------------------------------------------------------------

DOCS_DIR = CC_ROOT / "docs"
BUILDING_COORDS_PATH = DOCS_DIR / "building_coordinates.json"

CHATS_DIR = CC_ROOT / "chats"
CHATS_DIR.mkdir(parents=True, exist_ok=True)

ROUTES_DIR = CC_ROOT / "routes"
ROUTES_DIR.mkdir(parents=True, exist_ok=True)


def _load_building_coords() -> list[dict]:
    """
    Load building coordinates from docs/building_coordinates.json.

    Verwacht bij voorkeur een JSON-lijst:
      [
        {"name": "...", "address": "...", "latitude": 51.x, "longitude": 5.x},
        ...
      ]

    Maar tolereert ook:
      {"buildings": [ ... ]} of een enkel object.
    """
    if not BUILDING_COORDS_PATH.exists():
        return []

    try:
        with BUILDING_COORDS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Bij parse-fouten gewoon geen landmarks gebruiken
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "buildings" in data and isinstance(data["buildings"], list):
            return data["buildings"]
        # single object
        return [data]
    return []


BUILDING_COORDS: list[dict] = _load_building_coords()