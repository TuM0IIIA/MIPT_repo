# Repository Guidelines

## Project Overview

SmartBot is a Python computer-vision pipeline that captures webcam frames, runs YOLOv8 object detection, and sends annotated photo alerts to Telegram. Designed to run on Raspberry Pi Zero 2W as well as standard desktop hardware.

## Project Structure & Module Organization

Three independent daemon threads communicate via bounded `queue.Queue` objects:

```
CameraService  →[frame_queue]→  DetectionService  →[result_queue]→  ReportSender
                                       ↓[preview_queue]
                                  OpenCV window (optional)
```

- `services/camera_service.py` — captures frames on an interval, puts raw frames onto `frame_queue`
- `services/detection_service.py` — runs YOLOv8 inference, applies per-label cooldown, emits `DetectionEvent` dataclass to `result_queue`; also pushes annotated frames to `preview_queue` (drop-oldest policy)
- `services/report_sender.py` — reads `DetectionEvent`, sends annotated JPEG + HTML caption to Telegram with 3-attempt exponential backoff; also saves local copy to `logs/reports/`
- `services/console_reporter.py` — Phase 1 console-only reporter, kept for reference
- `utils/config_loader.py` / `utils/logger.py` — shared config and logging helpers

`DetectionEvent` (defined in `detection_service.py`) is the contract between detection and reporting layers — include `timestamp`, `label`, `confidence`, `bounding_box`, `frame` (annotated numpy array), and `all_detections`.

## Build & Development Commands

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py                    # run the system; Ctrl+C to stop
```

No test suite or Makefile exists in this repo.

## Configuration

All runtime behaviour is controlled by `config/settings.json`. Key fields:

| Key | Purpose |
|-----|---------|
| `camera.source` | Camera index (0 = default) |
| `camera.capture_interval_seconds` | Frame capture rate (use 10 on Pi Zero) |
| `detection.model_path` | Path to YOLO model file (`yolov8n.onnx`) |
| `detection.confidence_threshold` | Min confidence to emit an event (0.0–1.0) |
| `detection.cooldown_seconds` | Seconds before re-alerting same label (set 10 for testing) |
| `detection.target_labels` | Labels that trigger Telegram reports |
| `telegram.bot_token` | BotFather token — **never commit a real token** |
| `telegram.chat_id` | Destination chat ID |
| `preview.enabled` | Opens an OpenCV debug window (press Q/ESC to close) |

## Coding Style & Naming Conventions

- Python 3.9+, no formatter or linter config enforced by tooling
- Services follow a lifecycle interface: `start()`, `stop()`, `is_running()` — new services must implement all three
- All services use `get_logger(__name__)` from `utils/logger.py`
- Queues are bounded (`maxsize=2` for frame/preview, `maxsize=10` for results) — producers use `get_nowait` + drop-oldest for preview, and blocking `put` elsewhere

## Commit Guidelines

Commit style observed in history: lowercase imperative subject, descriptive body when needed.

```
config: update settings for Pi Zero 2W deployment
added Telegram reports feature
```

Prefix with `config:`, `fix:`, or similar when the change is scoped.
