from pathlib import Path
import os
from dotenv import load_dotenv

# /opt/HiCoDe-RASAHQ/CampusCompass/app/config.py
APP_ROOT = Path(__file__).resolve().parent        # .../CampusCompass/app
CC_ROOT = APP_ROOT.parent                         # .../CampusCompass

# laad secrets uit de project root
load_dotenv(CC_ROOT / ".env")

# openai keys uit .env
OPENAI_API_KEY = (
    os.getenv("OPENAI_SPENCE")
    or os.getenv("OPENAI_OLIVIER")
    or os.getenv("OPENAI_API_KEY")
)
