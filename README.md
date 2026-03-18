# SmartBot 🤖

An AI-powered home monitoring system that runs on a Raspberry Pi mounted to a
vacuum robot. Detects pets, people, and unusual objects using YOLOv8 and sends
photo reports directly to Telegram.

---

## How it works

The camera captures a frame every N seconds. Each frame is run through a
YOLOv8 ONNX model. If a target object (cat, dog, person, bird) is detected
above the confidence threshold and is not in cooldown, an annotated photo
and text report are sent to a Telegram chat.

```
Camera → YOLOv8 (ONNX) → Cooldown check → Telegram report
```

---

## Project structure

```
smartbot/
  main.py                        Entry point. Starts all services.
  requirements.txt               Python dependencies.
  config/
    settings.json                All configuration (camera, detection, telegram).
  services/
    camera_service.py            Captures frames. Uses cv2 on laptop, picamera2 on Pi.
    detection_service.py         Runs YOLOv8 ONNX inference via cv2.dnn.
    console_reporter.py          Phase 1 only. Prints detections to terminal.
    report_sender.py             Phase 2+. Sends photo + text to Telegram.
  utils/
    logger.py                    Shared logger (console + file).
    config_loader.py             Loads settings.json.
  logs/
    smartbot.log                 Full application logs.
    reports/                     Saved annotated images.
```

---

## Setup — Laptop (development)

**Requirements:** Python 3.9+, working webcam.

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run:**
```bash
python main.py
```

A live preview window opens showing annotated frames in real time.
Press Q or ESC to quit.

---

## Setup — Raspberry Pi Zero 2W (production)

**Requirements:** Raspberry Pi OS, Pi Camera Module, Python 3.9+.

### 1. Export the ONNX model (run once on your laptop)

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640)"
```

This creates `yolov8n.onnx`. Copy it to the Pi along with the project:

```bash
scp -r smartbot/ pi@YOUR_PI_IP:/home/pi/smartbot
scp yolov8n.onnx pi@YOUR_PI_IP:/home/pi/smartbot/
```

### 2. Install system dependencies on the Pi

```bash
sudo apt update
sudo apt install -y python3-picamera2
```

### 3. Create venv with system site packages

The `--system-site-packages` flag is required so the venv can access
`picamera2` and `libcamera`, which are system-level Pi libraries that
cannot be installed via pip.

```bash
cd /home/pi/smartbot
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install opencv-python-headless requests
```

### 4. Configure for Pi

Edit `config/settings.json`:

```json
"camera": {
  "capture_interval_seconds": 10
},
"detection": {
  "model_path": "yolov8n.onnx"
},
"preview": {
  "enabled": false
}
```

### 5. Run

```bash
source venv/bin/activate
python main.py
```

---

## Configuration reference

All settings live in `config/settings.json`. Never hardcode credentials.

| Key | Default | Description |
|-----|---------|-------------|
| `camera.source` | `0` | Camera index. 0 = default webcam. Ignored on Pi (uses picamera2). |
| `camera.capture_interval_seconds` | `3` (laptop) / `10` (Pi) | Seconds between frame captures. |
| `camera.resolution_width/height` | `480 x 360` | Capture resolution. Lower = faster inference on Pi. |
| `detection.model_path` | `yolov8n.onnx` | Path to ONNX model file. |
| `detection.confidence_threshold` | `0.60` | Minimum confidence to trigger a detection. |
| `detection.target_labels` | `["cat","dog","person","bird"]` | Objects that trigger reports. |
| `detection.cooldown_seconds` | `300` | Minimum seconds between reports of the same label. |
| `telegram.bot_token` | — | Bot token from @BotFather. Keep secret, never commit. |
| `telegram.chat_id` | — | Your personal Telegram chat ID. |
| `preview.enabled` | `true` (laptop) / `false` (Pi) | Show live annotated video window. |

**Tip:** Set `cooldown_seconds` to `10` while testing so reports fire quickly.

---

## Telegram bot setup

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts to get a bot token
3. Start a conversation with your new bot (send it any message)
4. Open `https://api.telegram.org/botYOUR_TOKEN/getUpdates` in a browser
5. Find `"chat": {"id": 123456789}` in the response — that is your chat ID
6. Add both values to `config/settings.json`

---

## Report decision logic

A Telegram report is sent only when all three conditions pass:

1. **Confidence ≥ threshold** — model must be confident enough (default 60%)
2. **Label in target list** — detected object must be in `target_labels`
3. **Not in cooldown** — same label not reported in the last `cooldown_seconds`

If any condition fails the frame is skipped silently.

---

## AI model

- **Model:** YOLOv8 nano (Ultralytics)
- **Format:** ONNX — runs via `cv2.dnn`, no PyTorch needed on Pi
- **Inference time:** ~200ms on laptop, ~5–10s on Pi Zero 2W
- **Pre-trained on:** COCO dataset (80 classes including cat, dog, person, bird)

To regenerate the ONNX file from the original PyTorch model:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640)"
```

---

## .gitignore

```
config/settings.json
yolov8n.pt
yolov8n.onnx
venv/
logs/
__pycache__/
*.pyc
*.onnx
*.pt
```

Keep `config/settings.json` out of git — it contains your Telegram credentials.
Commit a `config/settings.example.json` with placeholder values instead.

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Done | Pi + camera + YOLOv8 detection |
| Phase 2 | ✅ Done | Telegram bot reports |
| Phase 3 | ⬜ Next | Cloud backend (Node.js, PostgreSQL, Redis) |
| Phase 4 | ⬜ Later | Admin dashboard, beta launch |
