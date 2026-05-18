# SmartBot — Pi Edge

> Raspberry Pi Zero 2W edge component of the SmartBot home monitoring system. Runs YOLOv8n ONNX object detection on a robot-mounted camera and POSTs annotated photo alerts to the [SmartBot cloud API](https://github.com/Ali-Jabowr/smartbot-api).

---

## What It Does

Captures frames from a Pi Camera every 10 seconds, runs local AI inference to detect pets and people, and — when a detection passes confidence and cooldown checks — sends an annotated JPEG to the cloud API, which triggers a Telegram alert on your phone.

```
Pi Camera (picamera2)
    → frame every 10s
    → YOLOv8n ONNX (cv2.dnn) — no PyTorch, no cloud inference
    → confidence ≥ 60% + 5-min cooldown per label
    → annotated JPEG + payload
    → HTTPS POST to SmartBot Cloud API
        → Telegram photo alert on your phone
```

---

## Hardware

| Component | Spec |
|-----------|------|
| Board | Raspberry Pi Zero 2W |
| RAM | 512MB |
| Camera | Arducam IMX219 / Pi Camera Module 3 Wide |
| Mounted on | Robot vacuum (passive observation — no movement control) |

---

## Detection Logic

An alert fires only when **all three** pass:

1. **Confidence** — YOLO score ≥ `confidence_threshold` (default: 60%)
2. **Label filter** — detected class is in `target_labels` (default: `cat`, `dog`, `person`, `bird`)
3. **Cooldown** — same label not reported within `cooldown_seconds` (default: 5 min)

---

## Project Structure

```
smartbot-pi/
├── main.py                     Entry point — starts all 3 services as threads
├── services/
│   ├── camera_service.py       picamera2 on Pi / cv2.VideoCapture on laptop (dev)
│   ├── detection_service.py    ONNX inference, confidence filter, cooldown logic, annotation
│   └── report_sender.py        Compresses image, POSTs multipart to cloud API
├── utils/
│   ├── config_loader.py        Loads config/settings.json
│   └── logger.py               Console + file logging
├── config/
│   ├── settings.json           Live config — gitignored (contains credentials)
│   └── settings.example.json   Template — safe to commit
├── logs/
│   └── smartbot.log            Application logs
└── requirements.txt
```

---

## Setup

### Prerequisites
- Raspberry Pi Zero 2W running Raspberry Pi OS Lite
- Pi Camera Module connected and enabled (`sudo raspi-config` → Interface Options → Camera)

### Install

```bash
sudo apt update
sudo apt install -y python3-picamera2
```

> `picamera2` must be installed via `apt` — it depends on `libcamera`, a system-level C++ library that cannot be pip-installed. The venv must use `--system-site-packages` for this reason.

```bash
git clone <this-repo-url> smartbot
cd smartbot

python3 -m venv venv --system-site-packages
source venv/bin/activate

pip install opencv-python-headless requests
```

### Configure

```bash
cp config/settings.example.json config/settings.json
```

Edit `config/settings.json`:

```json
{
  "device": {
    "device_id": "pi-zero-01"
  },
  "camera": {
    "capture_interval_seconds": 10,
    "resolution_width": 480,
    "resolution_height": 360
  },
  "detection": {
    "model_path": "yolov8n.onnx",
    "confidence_threshold": 0.60,
    "target_labels": ["cat", "dog", "person", "bird"],
    "cooldown_seconds": 300
  },
  "api": {
    "url": "https://<your-railway-url>/device/report",
    "key": "YOUR_API_KEY"
  }
}
```

> `settings.json` is gitignored — never commit credentials.

### Add the ONNX model

Generate on your laptop (requires GPU or patience):

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640)"
```

Copy to Pi:

```bash
scp yolov8n.onnx pi@<pi-ip>:~/smartbot/
```

### Run

```bash
source venv/bin/activate
python main.py
```

Stop with `Ctrl+C`.

---

## Local Development (Laptop)

The camera service auto-detects the environment — on a laptop it falls back to `cv2.VideoCapture` (webcam). Everything else works identically.

```bash
python3 -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install opencv-python-headless requests
python main.py
```

Set `cooldown_seconds` to `10` in `settings.json` for faster testing.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'libcamera'`**
→ Venv was created without `--system-site-packages`. Recreate: `python3 -m venv venv --system-site-packages`

**Camera not opening**
→ Run `sudo raspi-config` → Interface Options → Camera → Enable. Reboot.

**Nothing detected**
→ Lower `confidence_threshold` to `0.40` in `settings.json`. Check that the object is well lit and clearly in frame.

**Alert not firing despite detection**
→ Check `cooldown_seconds` — same label may have been reported recently. Set to `10` to test.

**Upload timing out**
→ Pi Zero 2W has weak single-core WiFi. The image is already compressed to 320×240 at JPEG quality 40. If timeouts persist, check your network signal at the Pi's location.

---

## Related

- **Cloud API:** [smartbot-api](https://github.com/Ali-Jabowr/smartbot-api) — Railway backend, Cloudinary, PostgreSQL, Telegram alerts
