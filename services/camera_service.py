"""
camera_service.py — captures frames from webcam into a queue.
Swap cv2.VideoCapture for picamera2 when Pi Camera Module arrives (see note at bottom).
"""

import queue
import threading
import time
from typing import Any, Optional

import cv2

from utils.exceptions import CameraError
from utils.logger import get_logger

logger = get_logger(__name__)


class CameraService:
    """Captures frames from a camera on a fixed interval and pushes them onto a queue."""

    def __init__(self, config: dict[str, Any], frame_queue: queue.Queue) -> None:
        self.source   = config["source"]
        self.interval = config["capture_interval_seconds"]
        self.width    = config["resolution_width"]
        self.height   = config["resolution_height"]
        self.frame_queue = frame_queue
        self.capture: Optional[cv2.VideoCapture] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _init_camera(self) -> bool:
        """Open the camera device and configure resolution. Returns False if unavailable."""
        logger.info(f"Opening camera (source={self.source})...")
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            logger.error(f"Could not open camera source={self.source}. Is it used by another app?")
            return False
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info(f"Camera opened at {self.width}x{self.height}")
        return True

    def _capture_loop(self) -> None:
        logger.info(f"Camera capturing every {self.interval}s.")
        while not self._stop_event.is_set():
            t0 = time.time()
            ret, frame = self.capture.read()
            if not ret or frame is None:
                logger.warning("Capture failed, retrying in 2s...")
                time.sleep(2)
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
            self._stop_event.wait(timeout=max(0, self.interval - (time.time() - t0)))
        logger.info("Camera loop ended.")

    def start(self) -> None:
        """Open the camera and start the capture thread. Raises CameraError on failure."""
        if not self._init_camera():
            raise CameraError("Failed to open camera.")
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and release the camera device."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self.capture:
            self.capture.release()

    def is_running(self) -> bool:
        """Return True if the capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()


# ── Pi Camera Module swap ──────────────────────────────────────────
# from picamera2 import Picamera2
# self.picam = Picamera2()
# self.picam.configure(self.picam.create_preview_configuration(main={"size": (640, 480)}))
# self.picam.start()
# frame = self.picam.capture_array()   # same numpy array, rest unchanged
# ──────────────────────────────────────────────────────────────────
