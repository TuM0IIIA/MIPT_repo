import json
import os
from utils.logger import get_logger

logger = get_logger(__name__)

def load_config(path: str = "config/settings.json") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: '{path}'")
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.debug(f"Config loaded from {path}")
    return config
