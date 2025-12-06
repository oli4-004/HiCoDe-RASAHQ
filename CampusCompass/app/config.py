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