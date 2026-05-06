"""
camera_service.py — captures frames from camera into a queue.
Uses picamera2 if available (Pi Camera Module), otherwise cv2.VideoCapture (USB webcam/laptop).
"""

import queue
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from utils.exceptions import CameraError
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False


class CameraService:
    """Captures frames from a camera on a fixed interval and pushes them onto a queue."""

    def __init__(self, config: dict[str, Any], frame_queue: queue.Queue) -> None:
        self.source      = config["source"]
        self.interval    = config["capture_interval_seconds"]
        self.width       = config["resolution_width"]
        self.height      = config["resolution_height"]
        self.frame_queue = frame_queue
        self._capture: Optional[cv2.VideoCapture] = None
        self._picam      = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prev_gray = None
        self.motion_threshold = config.get("motion_threshold", 0.0)


    def _init_camera(self) -> bool:
        if _PICAMERA2_AVAILABLE:
            logger.info("picamera2 detected — using Pi Camera Module.")
            self._picam = Picamera2()
            cfg = self._picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self._picam.configure(cfg)
            self._picam.start()
            time.sleep(0.5)   # allow sensor to warm up
            logger.info(f"Pi camera started at {self.width}x{self.height}")
            return True

        logger.info(f"Opening camera (source={self.source})...")
        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            logger.error(f"Could not open camera source={self.source}. Is it used by another app?")
            return False
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info(f"Camera opened at {self.width}x{self.height}")
        return True

    def _read_frame(self) -> Optional[np.ndarray]:
        """Return one BGR frame, or None on failure."""
        if self._picam is not None:
            frame = self._picam.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, frame = self._capture.read()
        return frame if ret else None

    def _capture_loop(self) -> None:
        logger.info(f"Camera capturing every {self.interval}s.")
        while not self._stop_event.is_set():
            t0 = time.time()
            frame = self._read_frame()
            if frame is None:
                logger.warning("Capture failed, retrying in 2s...")
                time.sleep(2)
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev = self._prev_gray
            self._prev_gray = grey
            if self.motion_threshold > 0 and prev is not None:
                diff = cv2.absdiff(prev, grey)
                motion = np.mean(diff)
                if motion < self.motion_threshold:
                    logger.debug(f"Motion {motion:.2f} below threshold {self.motion_threshold}, skipping frame.")
                    continue
            
            self.frame_queue.put(frame)
            self._stop_event.wait(timeout=max(0, self.interval - (time.time() - t0)))
        logger.info("Camera loop ended.")

    def start(self) -> None:
        """Open the camera and start the capture thread. Raises CameraError on failure."""
        if not self._init_camera():
            raise CameraError("Failed to open camera.")
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraThread"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and release the camera device."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._picam is not None:
            self._picam.stop()
        if self._capture is not None:
            self._capture.release()

    def is_running(self) -> bool:
        """Return True if the capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()
