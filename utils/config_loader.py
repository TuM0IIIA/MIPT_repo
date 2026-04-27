import json
import os
from typing import Any

from utils.exceptions import ConfigError
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str = "config/settings.json") -> dict[str, Any]:
    """Load config from JSON, with optional .env overrides for Telegram credentials.

    Environment variables SMARTBOT_BOT_TOKEN and SMARTBOT_CHAT_ID override
    whatever is in settings.json — this is the standard pattern for server deployments.
    """
    if not os.path.exists(path):
        raise ConfigError(f"Config not found: '{path}'")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional; falls back to settings.json only

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    bot_token = os.getenv("SMARTBOT_BOT_TOKEN")
    chat_id = os.getenv("SMARTBOT_CHAT_ID")
    if bot_token:
        config.setdefault("telegram", {})["bot_token"] = bot_token
    if chat_id:
        config.setdefault("telegram", {})["chat_id"] = chat_id

    logger.debug(f"Config loaded from {path}")
    return config
