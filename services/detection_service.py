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

# YOLOv8n ONNX was trained on COCO 80 classes
_COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
_INPUT_SIZE = 640   # YOLOv8n ONNX expects 640×640


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
        self._absence_threshold: int = config.get("absence_threshold_frames", 5)

        self.frame_queue   = frame_queue
        self.result_queue  = result_queue
        self.preview_queue = preview_queue

        self._currently_present: set[str] = set()
        self._frames_absent: dict[str, int] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.model = None

    # ── model ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the YOLO ONNX model via cv2.dnn. Raises ModelError on failure."""
        logger.info(f"Loading YOLO model: {self.model_path}")
        try:
            self.model = cv2.dnn.readNetFromONNX(self.model_path)
            logger.info("Model loaded.")
        except Exception as e:
            raise ModelError(f"Failed to load YOLO model: {e}") from e

    # ── presence tracking ─────────────────────────────────────────

    def _mark_present(self, label: str) -> None:
        self._currently_present.add(label)
        self._frames_absent[label] = 0

    def _update_absence(self, detected_target_labels: set[str]) -> None:
        """Increment absence counters for labels no longer in frame; evict when threshold reached."""
        for label in list(self._currently_present):
            if label not in detected_target_labels:
                self._frames_absent[label] = self._frames_absent.get(label, 0) + 1
                if self._frames_absent[label] >= self._absence_threshold:
                    self._currently_present.discard(label)
                    self._frames_absent.pop(label, None)
                    logger.info(f"'{label}' left the scene")
            else:
                self._frames_absent[label] = 0

    # ── inference ─────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray) -> list[dict[str, Any]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (_INPUT_SIZE, _INPUT_SIZE), swapRB=True, crop=False
        )
        self.model.setInput(blob)
        raw = self.model.forward()   # shape: (1, 84, 8400)

        preds = raw[0].T             # (8400, 84): cx cy w h + 80 class scores
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences >= self.confidence_threshold
        preds = preds[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(preds) == 0:
            return []

        scale_x, scale_y = w / _INPUT_SIZE, h / _INPUT_SIZE
        boxes_xywh: list[list[int]] = []
        for cx, cy, bw, bh, *_ in preds:
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            boxes_xywh.append([x1, y1, int(bw * scale_x), int(bh * scale_y)])

        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(), self.confidence_threshold, 0.45)
        flat = indices.flatten() if len(indices) else []

        detections: list[dict[str, Any]] = []
        for i in flat:
            x, y, bw, bh = boxes_xywh[i]
            cls = int(class_ids[i])
            label = _COCO_CLASSES[cls] if cls < len(_COCO_CLASSES) else str(cls)
            detections.append({
                "label": label,
                "confidence": round(float(confidences[i]), 3),
                "x": x, "y": y, "w": bw, "h": bh,
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

            if label in self.target_labels and label in self._currently_present:
                cv2.putText(out, "present",
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
                while not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
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

            detected_target_labels = {
                d["label"] for d in detections if d["label"] in self.target_labels
            }
            self._update_absence(detected_target_labels)

            if not detections:
                continue

            found = [f"{d['label']} ({d['confidence']:.0%})" for d in detections]
            logger.info(f"In frame: {', '.join(found)}")

            for det in detections:
                label = det["label"]
                if label not in self.target_labels:
                    continue
                if label in self._currently_present:
                    continue  # already in scene, no new event

                self._mark_present(label)
                event = DetectionEvent(
                    timestamp=datetime.now().isoformat(),
                    label=label,
                    confidence=det["confidence"],
                    bounding_box={"x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"]},
                    frame=annotated,
                    all_detections=detections,
                )
                self.result_queue.put(event)
                logger.info(f"EVENT: [{label.upper()}] {det['confidence']:.0%} — entered scene")
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
