"""
main.py - SmartBot Phase 2
Sends detection reports to Telegram.
Controls: Q or ESC to quit the preview window.
"""

import queue
import signal
import sys
import time

import cv2

from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.report_sender import ReportSender
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger("main")


def _validate_telegram_config(cfg):
    tg = cfg.get("telegram", {})
    token = tg.get("bot_token", "")
    chat_id = tg.get("chat_id", "")
    if not token or token == "YOUR_BOT_TOKEN_HERE":
        raise ValueError(
            "Telegram bot_token is not set! "
            "Open config/settings.json and replace YOUR_BOT_TOKEN_HERE "
            "with the token from @BotFather."
        )
    if not chat_id or chat_id == "YOUR_CHAT_ID_HERE":
        raise ValueError(
            "Telegram chat_id is not set! "
            "Open config/settings.json and replace YOUR_CHAT_ID_HERE "
            "with your chat ID."
        )


def main():
    logger.info("=" * 52)
    logger.info("  SmartBot Phase 2 - Starting")
    logger.info("=" * 52)

    config = load_config("config/settings.json")
    _validate_telegram_config(config)

    preview_enabled = config.get("preview", {}).get("enabled", False)
    window_title = config.get("preview", {}).get("window_title", "SmartBot Preview")

    frame_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=10)
    preview_queue = queue.Queue(maxsize=2) if preview_enabled else None

    camera = CameraService(config["camera"], frame_queue)
    detector = DetectionService(config["detection"], frame_queue, result_queue, preview_queue)
    reporter = ReportSender(config, result_queue)

    def shutdown(sig=None, frame=None):
        print("")
        logger.info("Shutting down...")
        reporter.stop()
        detector.stop()
        camera.stop()
        if preview_enabled:
            cv2.destroyAllWindows()
        logger.info("Stopped. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        logger.info("Starting Telegram report sender...")
        reporter.start()

        logger.info("Loading AI model...")
        detector.start()

        logger.info("Starting camera...")
        camera.start()

    except (RuntimeError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("")
    logger.info("All systems running!")
    logger.info("  Device       : %s", config["device"]["device_name"])
    logger.info("  Watching for : %s", config["detection"]["target_labels"])
    logger.info("  Cooldown     : %ss", config["detection"]["cooldown_seconds"])
    logger.info("  Reports to   : Telegram chat %s", config["telegram"]["chat_id"])
    if preview_enabled:
        logger.info("  Preview      : OPEN  (press Q or ESC to quit)")
    logger.info("")

    if preview_enabled:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 800, 600)

    while True:
        if preview_enabled:
            try:
                frame = preview_queue.get_nowait()
                cv2.imshow(window_title, frame)
            except queue.Empty:
                pass
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                shutdown()
        else:
            time.sleep(5)

        if not camera.is_running():
            logger.error("Camera service crashed!")
        if not detector.is_running():
            logger.error("Detection service crashed!")


if __name__ == "__main__":
    main()
