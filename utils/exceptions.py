class SmartBotError(Exception):
    """Base exception for all SmartBot domain errors."""


class CameraError(SmartBotError):
    """Camera cannot be opened or a frame cannot be read."""


class ModelError(SmartBotError):
    """YOLO model failed to load or run inference."""


class ConfigError(SmartBotError):
    """Configuration is missing, malformed, or incomplete."""


class TelegramError(SmartBotError):
    """Telegram report delivery failed after all retries."""
