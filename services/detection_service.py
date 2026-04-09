"""
detection_service.py
--------------------
Runs YOLOv8 ONNX model via cv2.dnn — no ultralytics, no PyTorch needed.
Works on Raspberry Pi Zero 2W with opencv-python-headless only.

Two output queues:
  result_queue   — DetectionEvents for valid detections (Telegram)
  preview_queue  — Annotated frames for debug window (laptop only)
"""

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# YOLOv8 was trained on COCO — these are the 80 class names in order
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

# Bounding box colours per label (BGR)
LABEL_COLORS = {
    "cat":    (0,   200,  80),
    "dog":    (0,   140, 255),
    "person": (200,  60,  60),
    "bird":   (220, 180,   0),
}
DEFAULT_COLOR = (160, 160, 160)

# YOLOv8 ONNX input size
INPUT_SIZE = 640


@dataclass
class DetectionEvent:
    timestamp:      str
    label:          str
    confidence:     float
    bounding_box:   dict
    frame:          np.ndarray
    all_detections: list


class DetectionService:
    def __init__(
        self,
        config:        dict,
        frame_queue:   queue.Queue,
        result_queue:  queue.Queue,
        preview_queue: Optional[queue.Queue] = None,
    ):
        self.model_path           = config["model_path"]
        self.confidence_threshold = config["confidence_threshold"]
        self.target_labels        = set(config["target_labels"])
        self.cooldown_seconds     = config["cooldown_seconds"]

        self.frame_queue   = frame_queue
        self.result_queue  = result_queue
        self.preview_queue = preview_queue

        self._last_detected: dict = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.net = None

    # ── model ────────────────────────────────────────────────────

    def _load_model(self):
        logger.info(f"Loading ONNX model: {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(self.model_path)

        # Force CPU — Pi has no CUDA
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        logger.info("ONNX model loaded successfully.")

    # ── cooldown ─────────────────────────────────────────────────

    def _in_cooldown(self, label: str) -> bool:
        t = self._last_detected.get(label)
        return t is not None and (time.time() - t) < self.cooldown_seconds

    def _set_cooldown(self, label: str):
        self._last_detected[label] = time.time()

    def _cooldown_remaining(self, label: str) -> float:
        t = self._last_detected.get(label)
        return max(0, self.cooldown_seconds - (time.time() - t)) if t else 0

    # ── inference ────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray) -> list:
        """
        Runs YOLOv8 ONNX inference via cv2.dnn.
        Returns list of dicts: [{label, confidence, x, y, w, h}]
        """
        h, w = frame.shape[:2]

        # Preprocess: resize to 640x640, normalise to 0-1, NCHW format
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(INPUT_SIZE, INPUT_SIZE),
            swapRB=True,   # BGR → RGB
            crop=False
        )
        self.net.setInput(blob)

        # Forward pass — YOLOv8 ONNX output shape: (1, 84, 8400)
        # 84 = 4 box coords + 80 class scores
        outputs = self.net.forward()
        output = outputs[0]                    # (1, 84, 8400)
        predictions = np.squeeze(output).T     # (8400, 84)

        # Scale factors to map back to original frame size
        x_scale = w / INPUT_SIZE
        y_scale = h / INPUT_SIZE

        detections = []
        for pred in predictions:
            class_scores = pred[4:]            # 80 class scores
            class_id     = int(np.argmax(class_scores))
            confidence   = float(class_scores[class_id])

            if confidence < self.confidence_threshold:
                continue

            label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

            # YOLOv8 outputs cx, cy, w, h (centre format, normalised to INPUT_SIZE)
            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            x1 = int((cx - bw / 2) * x_scale)
            y1 = int((cy - bh / 2) * y_scale)
            x2 = int((cx + bw / 2) * x_scale)
            y2 = int((cy + bh / 2) * y_scale)

            # Clamp to frame boundaries
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            detections.append({
                "label":      label,
                "confidence": round(confidence, 3),
                "x": x1, "y": y1,
                "w": x2 - x1, "h": y2 - y1,
            })

        # Non-maximum suppression to remove duplicate boxes
        if not detections:
            return []

        boxes  = [[d["x"], d["y"], d["w"], d["h"]] for d in detections]
        scores = [d["confidence"] for d in detections]
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, nms_threshold=0.45)

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]

    # ── annotation ───────────────────────────────────────────────

    def _annotate_frame(self, frame: np.ndarray, detections: list) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # Status bar
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 28), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        cv2.putText(out, "SmartBot  |  Debug Preview",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        for det in detections:
            x, y, bw, bh = det["x"], det["y"], det["w"], det["h"]
            label  = det["label"]
            conf   = det["confidence"]
            color  = LABEL_COLORS.get(label, DEFAULT_COLOR)

            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            caption = f"{label}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            pill_y1 = max(y - th - 10, 30)
            pill_y2 = pill_y1 + th + 8
            cv2.rectangle(out, (x, pill_y1), (x + tw + 10, pill_y2), color, -1)
            cv2.putText(out, caption, (x + 5, pill_y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if label in self.target_labels and self._in_cooldown(label):
                cd = f"cooldown {self._cooldown_remaining(label):.0f}s"
                cv2.putText(out, cd, (x + 5, y + bh - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 200, 255), 1, cv2.LINE_AA)

        if not detections:
            cv2.putText(out, "No objects detected",
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        return out

    def _push_preview(self, frame: np.ndarray):
        if self.preview_queue is None:
            return
        if self.preview_queue.full():
            try:
                self.preview_queue.get_nowait()
            except queue.Empty:
                pass
        self.preview_queue.put(frame)

    # ── main loop ────────────────────────────────────────────────

    def _detection_loop(self):
        logger.info("Detection service started.")

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=2)
            except queue.Empty:
                continue

            t0 = time.time()
            try:
                detections = self._run_inference(frame)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                continue

            elapsed = time.time() - t0
            logger.debug(f"Inference {elapsed:.2f}s — {len(detections)} object(s)")

            annotated = self._annotate_frame(frame, detections)
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
                    bounding_box={"x": det["x"], "y": det["y"],
                                  "w": det["w"], "h": det["h"]},
                    frame=annotated,
                    all_detections=detections,
                )
                self.result_queue.put(event)
                logger.info(f"EVENT: [{label.upper()}] {det['confidence']:.0%}")
                break

        logger.info("Detection loop ended.")

    # ── lifecycle ────────────────────────────────────────────────

    def start(self):
        self._load_model()
        self._thread = threading.Thread(
            target=self._detection_loop, daemon=True, name="DetectionThread"
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
