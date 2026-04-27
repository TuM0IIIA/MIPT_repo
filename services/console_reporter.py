"""
console_reporter.py — Phase 1 output.
Prints detection reports and saves annotated JPGs to logs/reports/.
Replaced by report_sender.py in Phase 2.
"""

import os
import queue
import threading
from typing import Optional

import cv2

from services.detection_service import DetectionEvent
from utils.logger import get_logger

logger = get_logger(__name__)
REPORTS_DIR = "logs/reports"


class ConsoleReporter:
    """Prints detection reports to the console and saves annotated images locally."""

    def __init__(self, result_queue: queue.Queue) -> None:
        self.result_queue = result_queue
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._count = 0
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def _format(self, e: DetectionEvent) -> str:
        """Format a DetectionEvent as a human-readable console report."""
        others = [f"    - {d['label']} ({d['confidence']:.0%})"
                  for d in e.all_detections if d["label"] != e.label]
        return (
            f"\n{'=' * 52}\n"
            f"  DETECTION #{self._count}\n"
            f"{'=' * 52}\n"
            f"  Main:       {e.label.upper()}  ({e.confidence:.0%})\n"
            f"  Time:       {e.timestamp}\n"
            f"  Box:        x={e.bounding_box['x']} y={e.bounding_box['y']} "
            f"w={e.bounding_box['w']} h={e.bounding_box['h']}\n"
            f"  Also seen:\n" + ("\n".join(others) if others else "    (none)") +
            f"\n{'=' * 52}\n"
        )

    def _save(self, e: DetectionEvent) -> str:
        """Save the annotated frame to disk and return the file path."""
        safe = e.timestamp.replace(":", "-").replace(".", "-")
        path = f"{REPORTS_DIR}/report_{self._count:04d}_{e.label}_{safe}.jpg"
        cv2.imwrite(path, e.frame)
        return path

    def _loop(self) -> None:
        logger.info("Console reporter ready.")
        while not self._stop_event.is_set():
            try:
                event: DetectionEvent = self.result_queue.get(timeout=2)
            except queue.Empty:
                continue
            self._count += 1
            print(self._format(event))
            path = self._save(event)
            logger.info(f"Image saved: {path}")

    def start(self) -> None:
        """Start the reporter thread."""
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ReporterThread")
        self._thread.start()

    def stop(self) -> None:
        """Signal the reporter thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def is_running(self) -> bool:
        """Return True if the reporter thread is alive."""
        return self._thread is not None and self._thread.is_alive()
