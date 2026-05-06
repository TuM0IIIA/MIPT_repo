"""
report_sender.py
----------------
Phase 3 — replaces direct Telegram delivery.

Reads DetectionEvents from the result queue and POSTs JSON reports
to the cloud API. The cloud handles Telegram notifications.

Drop-in replacement: the interface is identical to ConsoleReporter.
main.py only needs one line changed (see bottom of this file).
"""

import os
import queue
import threading
import time
from datetime import datetime
from typing import Any, Optional

import cv2
import requests

from services.detection_service import DetectionEvent
from utils.logger import get_logger

logger = get_logger(__name__)

# Emoji per detected label — makes reports easier to read at a glance
LABEL_EMOJI: dict[str, str] = {
    "cat":    "🐱",
    "dog":    "🐶",
    "person": "🧍",
    "bird":   "🐦",
}
DEFAULT_EMOJI = "📦"


class ReportSender:
    """POSTs JSON detection reports to the cloud API with retry logic."""

    def __init__(self, config: dict[str, Any], result_queue: queue.Queue) -> None:
        self.device_name = config["device"]["device_name"]

        self.result_queue = result_queue
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._report_count = 0

        self._send_report_api_url = config["api"]["API_ENDPOINT"]
        self._api_key = config["api"]["API_KEY"]

        os.makedirs("logs/reports", exist_ok=True)

    # ── Formatting ────────────────────────────────────────────────

    def _format_caption(self, event: DetectionEvent) -> str:
        """Build the detection report text sent as the status field."""
        emoji = LABEL_EMOJI.get(event.label, DEFAULT_EMOJI)

        try:
            dt = datetime.fromisoformat(event.timestamp)
            time_str = dt.strftime("Today at %H:%M")
        except ValueError:
            time_str = event.timestamp

        others = [
            d["label"] for d in event.all_detections
            if d["label"] != event.label
        ]
        others_str = ", ".join(others) if others else "none"

        return (
            f"{emoji} <b>{event.label.capitalize()} detected</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Confidence : {event.confidence:.0%}\n"
            f"Time       : {time_str}\n"
            f"Device     : {self.device_name}\n"
            f"Also seen  : {others_str}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Report #{self._report_count}"
        )

    # ── Sending ───────────────────────────────────────────────────

    def _send_report(self, event: DetectionEvent) -> bool:
        """POST a JSON detection report to the cloud API, with up to 3 retries.

        Returns True on success, False after all retries are exhausted.
        """
        caption = self._format_caption(event)

        for attempt in range(1, 4):
            try:
                response = requests.post(
                    self._send_report_api_url,
                    json={
                        "deviceId": self.device_name,
                        "status": caption,
                    },
                    headers={"X-API-Key": self._api_key},
                    timeout=15,
                )

                if response.status_code == 200:
                    logger.info(f"Report to API #{self._report_count} sent successfully.")
                    return True

                logger.warning(
                    f"API returned {response.status_code}: {response.text} "
                    f"(attempt {attempt}/3)"
                )

            except requests.exceptions.ConnectionError:
                logger.warning(f"No internet connection. Attempt {attempt}/3.")
            except requests.exceptions.Timeout:
                logger.warning(f"API request timed out. Attempt {attempt}/3.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Unexpected error sending to API: {e}")
                return False

            if attempt < 3:
                wait = 2 ** attempt   # 2s, 4s
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)

        logger.error(f"Failed to send report #{self._report_count} after 3 attempts.")
        return False

    def _save_local(self, event: DetectionEvent) -> None:
        """Save the annotated image locally for debugging."""
        safe_time = event.timestamp.replace(":", "-").replace(".", "-")
        path = f"logs/reports/report_{self._report_count:04d}_{event.label}_{safe_time}.jpg"
        cv2.imwrite(path, event.frame)
        logger.debug(f"Local copy saved: {path}")

    # ── Main loop ─────────────────────────────────────────────────

    def _send_loop(self) -> None:
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
            self._send_report(event)

        logger.info("Report sender stopped.")

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the report sender thread."""
        self._thread = threading.Thread(
            target=self._send_loop, daemon=True, name="ReportSenderThread"
        )
        self._thread.start()
        logger.info("Report sender thread started.")

    def stop(self) -> None:
        """Signal the sender thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def is_running(self) -> bool:
        """Return True if the sender thread is alive."""
        return self._thread is not None and self._thread.is_alive()
