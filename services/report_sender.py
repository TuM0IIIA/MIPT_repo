"""
report_sender.py
----------------
Phase 2 — replaces console_reporter.py.

Reads DetectionEvents from the result queue and sends them
to a Telegram chat as a photo + formatted text message.

Drop-in replacement: the interface is identical to ConsoleReporter.
main.py only needs one line changed (see bottom of this file).
"""

import os
import queue
import threading
import time
from io import BytesIO
from typing import Optional
from datetime import datetime

import cv2
import requests

from services.detection_service import DetectionEvent
from utils.logger import get_logger

logger = get_logger(__name__)

# Emoji per detected label — makes reports easier to read at a glance
LABEL_EMOJI = {
    "cat":    "🐱",
    "dog":    "🐶",
    "person": "🧍",
    "bird":   "🐦",
}
DEFAULT_EMOJI = "📦"


class ReportSender:
    def __init__(self, config: dict, result_queue: queue.Queue):
        """
        config: the full config dict from settings.json
        result_queue: we read DetectionEvents from here
        """
        telegram_cfg = config["telegram"]
        self.bot_token = telegram_cfg["bot_token"]
        self.chat_id   = telegram_cfg["chat_id"]
        self.device_name = config["device"]["device_name"]

        self.result_queue = result_queue
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._report_count = 0

        # API endpoints
        self._send_photo_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        self._send_msg_url   = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        # Save images for local reference too (same as console_reporter did)
        os.makedirs("logs/reports", exist_ok=True)

    # ── Formatting ────────────────────────────────────────────────

    def _format_caption(self, event: DetectionEvent) -> str:
        """
        Builds the Telegram message caption.
        Telegram supports basic HTML formatting.
        """
        emoji = LABEL_EMOJI.get(event.label, DEFAULT_EMOJI)

        # Format timestamp nicely: 2024-01-15T14:32:07 → Today at 14:32
        try:
            dt = datetime.fromisoformat(event.timestamp)
            time_str = dt.strftime("Today at %H:%M")
        except Exception:
            time_str = event.timestamp

        # Other objects found in the same frame
        others = [
            d["label"] for d in event.all_detections
            if d["label"] != event.label
        ]
        others_str = ", ".join(others) if others else "none"

        caption = (
            f"{emoji} <b>{event.label.capitalize()} detected</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Confidence : {event.confidence:.0%}\n"
            f"Time       : {time_str}\n"
            f"Device     : {self.device_name}\n"
            f"Also seen  : {others_str}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Report #{self._report_count}"
        )
        return caption

    # ── Sending ───────────────────────────────────────────────────

    def _frame_to_jpeg_bytes(self, frame) -> bytes:
        """Convert OpenCV numpy frame to JPEG bytes for Telegram upload."""
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")
        return BytesIO(buffer.tobytes())

    def _send_photo(self, event: DetectionEvent) -> bool:
        """
        Sends the annotated photo + caption to Telegram.
        Returns True on success, False on failure.
        Retries up to 3 times with exponential backoff.
        """
        caption = self._format_caption(event)
        jpeg_bytes = self._frame_to_jpeg_bytes(event.frame)

        for attempt in range(1, 4):
            try:
                response = requests.post(
                    self._send_photo_url,
                    data={
                        "chat_id": self.chat_id,
                        "caption": caption,
                        "parse_mode": "HTML",
                    },
                    files={"photo": ("detection.jpg", jpeg_bytes, "image/jpeg")},
                    timeout=15,
                )

                if response.status_code == 200:
                    logger.info(f"Telegram report #{self._report_count} sent successfully.")
                    return True
                else:
                    logger.warning(
                        f"Telegram returned {response.status_code}: {response.text} "
                        f"(attempt {attempt}/3)"
                    )

            except requests.exceptions.ConnectionError:
                logger.warning(f"No internet connection. Attempt {attempt}/3.")
            except requests.exceptions.Timeout:
                logger.warning(f"Telegram request timed out. Attempt {attempt}/3.")
            except Exception as e:
                logger.error(f"Unexpected error sending to Telegram: {e}")
                return False

            if attempt < 3:
                wait = 2 ** attempt   # 2s, 4s
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
                jpeg_bytes.seek(0)    # reset buffer for retry

        logger.error(f"Failed to send report #{self._report_count} after 3 attempts.")
        return False

    def _save_local(self, event: DetectionEvent):
        """Also save the image locally — useful for debugging."""
        safe_time = event.timestamp.replace(":", "-").replace(".", "-")
        path = f"logs/reports/report_{self._report_count:04d}_{event.label}_{safe_time}.jpg"
        cv2.imwrite(path, event.frame)
        logger.debug(f"Local copy saved: {path}")

    # ── Main loop ─────────────────────────────────────────────────

    def _send_loop(self):
        logger.info("Report sender started. Waiting for detection events...")

        while not self._stop_event.is_set():
            try:
                event: DetectionEvent = self.result_queue.get(timeout=2)
            except queue.Empty:
                continue

            self._report_count += 1
            logger.info(
                f"Sending report #{self._report_count}: "
                f"[{event.label.upper()}] {event.confidence:.0%}"
            )

            self._save_local(event)
            self._send_photo(event)

        logger.info("Report sender stopped.")

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self):
        self._thread = threading.Thread(
            target=self._send_loop, daemon=True, name="ReportSenderThread"
        )
        self._thread.start()
        logger.info("Report sender thread started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ── HOW TO SWITCH FROM PHASE 1 TO PHASE 2 IN main.py ─────────────
#
# Change ONE import and ONE line in main.py:
#
# BEFORE (Phase 1):
#   from services.console_reporter import ConsoleReporter
#   reporter = ConsoleReporter(result_queue)
#
# AFTER (Phase 2):
#   from services.report_sender import ReportSender
#   reporter = ReportSender(config, result_queue)
#
# That's it. Everything else stays identical.
# ─────────────────────────────────────────────────────────────────
