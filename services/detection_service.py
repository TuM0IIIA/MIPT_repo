"""
detection_service.py
--------------------
Runs YOLOv8 on every frame from the camera queue.

Two output queues:
  result_queue   — DetectionEvents for valid detections (reporter/Telegram)
  preview_queue  — Annotated frames for the debug window (every frame, not just detections)

The preview_queue is optional. If None, no preview frames are sent.
"""

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np

from utils.exceptions import ModelError
from utils.logger import get_logger

logger = get_logger(__name__)

# Colours for bounding boxes per label (BGR)
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "cat":    (0,   200,  80),   # green
    "dog":    (0,   140, 255),   # orange
    "person": (200,  60,  60),   # blue-ish
    "bird":   (220, 180,   0),   # cyan
}
DEFAULT_COLOR: tuple[int, int, int] = (160, 160, 160)


@dataclass
class DetectionEvent:
    timestamp:    str
    label:        str
    confidence:   float
    bounding_box: dict[str, int]
    frame:        np.ndarray   # annotated frame
    all_detections: list[dict[str, Any]]


class DetectionService:
    """Runs YOLOv8 inference on incoming frames, applies cooldown, and emits DetectionEvents."""

    def __init__(
        self,
        config: dict[str, Any],
        frame_queue:   queue.Queue,
        result_queue:  queue.Queue,
        preview_queue: Optional[queue.Queue] = None,
    ) -> None:
        self.model_path           = config["model_path"]
        self.confidence_threshold = config["confidence_threshold"]
        self.target_labels        = set(config["target_labels"])
        self.cooldown_seconds     = config["cooldown_seconds"]

        self.frame_queue   = frame_queue
        self.result_queue  = result_queue
        self.preview_queue = preview_queue

        self._last_detected: dict[str, float] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.model = None

    # ── model ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the YOLO model from disk. Raises ModelError on failure."""
        logger.info(f"Loading YOLO model: {self.model_path}")
        logger.info("First run downloads ~6MB — please wait...")
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info("Model loaded.")
        except Exception as e:
            raise ModelError(f"Failed to load YOLO model: {e}") from e

    # ── cooldown ──────────────────────────────────────────────────

    def _in_cooldown(self, label: str) -> bool:
        t = self._last_detected.get(label)
        return t is not None and (time.time() - t) < self.cooldown_seconds

    def _set_cooldown(self, label: str) -> None:
        self._last_detected[label] = time.time()

    def _cooldown_remaining(self, label: str) -> float:
        t = self._last_detected.get(label)
        if t is None:
            return 0.0
        return max(0.0, self.cooldown_seconds - (time.time() - t))

    # ── inference ─────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model(frame, verbose=False)
        detections: list[dict[str, Any]] = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                label = r.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "x": x1, "y": y1,
                    "w": x2 - x1, "h": y2 - y1,
                })
        return detections

    # ── annotation ────────────────────────────────────────────────

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        show_stats: bool = False,
    ) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of the frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Status bar ──────────────────────────────────────────
        if show_stats:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, 28), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
            cv2.putText(out, "SmartBot  |  Debug Preview",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            label_text = "Watching: " + ", ".join(sorted(self.target_labels))
            (tw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(out, label_text,
                        (w - tw - 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (140, 140, 140), 1, cv2.LINE_AA)

        # ── Bounding boxes ──────────────────────────────────────
        for det in detections:
            x, y, bw, bh = det["x"], det["y"], det["w"], det["h"]
            label = det["label"]
            conf  = det["confidence"]
            color = LABEL_COLORS.get(label, DEFAULT_COLOR)

            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)

            caption = f"{label}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            pill_y1 = max(y - th - 10, 30)
            pill_y2 = pill_y1 + th + 8
            cv2.rectangle(out, (x, pill_y1), (x + tw + 10, pill_y2), color, -1)
            cv2.putText(out, caption,
                        (x + 5, pill_y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if label in self.target_labels and self._in_cooldown(label):
                remaining = self._cooldown_remaining(label)
                cv2.putText(out, f"cooldown {remaining:.0f}s",
                            (x + 5, y + bh - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 200, 255), 1, cv2.LINE_AA)

        if not detections:
            cv2.putText(out, "No objects detected",
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (100, 100, 100), 1, cv2.LINE_AA)

        return out

    def _push_preview(self, frame: np.ndarray) -> None:
        """Push an annotated frame to the preview queue, dropping the oldest if full."""
        if self.preview_queue is None:
            return
        if self.preview_queue.full():
            try:
                self.preview_queue.get_nowait()
            except queue.Empty:
                pass
        self.preview_queue.put(frame)

    # ── main loop ─────────────────────────────────────────────────

    def _detection_loop(self) -> None:
        logger.info("Detection service started.")
        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=2)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                detections = self._run_inference(frame)
            except ModelError as e:
                logger.error(f"Inference error: {e}")
                continue

            elapsed = time.time() - t0
            logger.debug(f"Inference {elapsed:.2f}s — {len(detections)} object(s)")

            annotated = self._annotate_frame(frame, detections, show_stats=True)
            self._push_preview(annotated)

            if not detections:
                continue

            found = [f"{d['label']} ({d['confidence']:.0%})" for d in detections]
            logger.info(f"In frame: {', '.join(found)}")

            for det in detections:
                label = det["label"]
                if label not in self.target_labels:
                    continue
                if self._in_cooldown(label):
                    logger.info(f"'{label}' in cooldown ({self._cooldown_remaining(label):.0f}s left)")
                    continue

                self._set_cooldown(label)
                event = DetectionEvent(
                    timestamp=datetime.now().isoformat(),
                    label=label,
                    confidence=det["confidence"],
                    bounding_box={"x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"]},
                    frame=annotated,
                    all_detections=detections,
                )
                self.result_queue.put(event)
                logger.info(f"EVENT: [{label.upper()}] {det['confidence']:.0%}")
                break   # one event per frame

        logger.info("Detection loop ended.")

    # ── lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Load the model and start the detection thread. Raises ModelError on failure."""
        self._load_model()
        self._thread = threading.Thread(
            target=self._detection_loop, daemon=True, name="DetectionThread"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the detection thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def is_running(self) -> bool:
        """Return True if the detection thread is alive."""
        return self._thread is not None and self._thread.is_alive()
