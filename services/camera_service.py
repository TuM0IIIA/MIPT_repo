"""
camera_service.py — captures frames from webcam into a queue.
Swap cv2.VideoCapture for picamera2 when Pi Camera Module arrives (see note at bottom).
"""

import queue
import threading
import time
from typing import Optional

import cv2

from utils.logger import get_logger

logger = get_logger(__name__)


class CameraService:
    def __init__(self, config: dict, frame_queue: queue.Queue):
        self.source   = config["source"]
        self.interval = config["capture_interval_seconds"]
        self.width    = config["resolution_width"]
        self.height   = config["resolution_height"]
        self.frame_queue = frame_queue
        self.capture: Optional[cv2.VideoCapture] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _init_camera(self) -> bool:
        logger.info("Opening Pi Camera Module via picamera2...")
        from picamera2 import Picamera2
        self.picam = Picamera2()
        self.picam.configure(self.picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        ))
        self.picam.start()
        time.sleep(2)  # let camera warm up
        logger.info(f"Pi Camera started at {self.width}x{self.height}")
        return True

    def _capture_loop(self):
        logger.info(f"Camera capturing every {self.interval}s.")
        while not self._stop_event.is_set():
            t0 = time.time()
            frame = self.picam.capture_array()
            if frame is None:
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
    
    def start(self):
        if not self._init_camera():
            raise RuntimeError("Failed to open camera.")
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if hasattr(self, 'picam'):
            self.picam.stop()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

# ── Pi Camera Module swap ──────────────────────────────────────────
# from picamera2 import Picamera2
# self.picam = Picamera2()
# self.picam.configure(self.picam.create_preview_configuration(main={"size": (640, 480)}))
# self.picam.start()
# frame = self.picam.capture_array()   # same numpy array, rest unchanged
# ──────────────────────────────────────────────────────────────────
