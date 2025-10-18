from pathlib import Path
import os
from dotenv import load_dotenv

# laad .env uit projectroot
ACTIONS_ROOT = Path(__file__).resolve().parent      # .../actions
CC_ROOT = ACTIONS_ROOT.parent # .../CampusCompass
PROJECT_ROOT = CC_ROOT.parent # project map
load_dotenv(PROJECT_ROOT / ".env")

# openai
OPENAI_API_KEY = os.getenv("OPENAI_SPENCE") or os.getenv("OPENAI_OLIVIER") # or ... or ... Keys for the rest of the group