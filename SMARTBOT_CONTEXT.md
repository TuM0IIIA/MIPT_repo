# Project Context

## Project Overview

**SmartBot** is a startup that transforms standard vacuum robots into AI-powered home monitoring systems. The core idea is to mount a Raspberry Pi and camera onto an existing robot vacuum, run computer vision on the edge to detect pets and people, and deliver real-time photo reports to the owner via Telegram.

**Purpose:** Give homeowners passive awareness of their pets and home environment without requiring them to check cameras manually. The robot moves through the home naturally while the AI monitors continuously.

**Target users:** Pet owners who want to know where their pets are, whether they are safe, and whether anything unusual is happening at home — delivered as a simple Telegram message with a photo.

**Business context:** Investors are secured and market research is complete. The team is two people (co-founders). The strategy is to validate the core AI detection value before building any frontend. Telegram is the delivery channel for MVP — an app comes later only if the product proves out.

**Current status:** Phases 1 and 2 are fully complete and deployed on real Raspberry Pi Zero 2W hardware. The system detects pets and people and sends annotated photo reports to Telegram. Phase 3 (cloud backend) is planned but not yet started.

---

## Tech Stack & Architecture

### Edge device (Raspberry Pi)
- **Hardware:** Raspberry Pi Zero 2W (512MB RAM, quad-core 1GHz)
- **Camera:** Arducam IMX219 / Pi Camera Module (IMX708 sensor on Camera Module 3 Wide)
- **Language:** Python 3.11
- **Camera library:** picamera2 — installed via `sudo apt install python3-picamera2`, NOT via pip. libcamera is a system-level C++ dependency that cannot be pip-installed. The venv must be created with `--system-site-packages` flag.
- **AI inference:** YOLOv8 nano exported to ONNX format, run via `cv2.dnn.readNetFromONNX()`. No PyTorch or ultralytics on the Pi — the Pi has only 512MB RAM and ultralytics crashes during install.
- **Telegram delivery:** Direct HTTP calls via `requests` library to Telegram Bot API (`sendPhoto` endpoint)

### Cloud (Phase 3 — planned, not built yet)
- **Backend API:** Node.js + TypeScript + Express
- **Database:** PostgreSQL with Prisma ORM
- **Queue/cache:** Redis
- **Infrastructure:** Single VPS (Hetzner or DigitalOcean ~$20/month) with Docker Compose
- **Reverse proxy:** Nginx + Let's Encrypt SSL

### Development environment
- Windows laptop with webcam for local testing
- RTX 3060 GPU available for future custom model training
- SSH into Pi for deployment
- Git for version control (credentials kept out of repo)

### Current data flow (Phase 2)
```
Pi Camera → capture_array() every 10s
  → frame_queue (maxsize=2, drops oldest)
  → detection_service: cv2.dnn ONNX inference
  → confidence filter (≥60%) + label filter + cooldown check
  → result_queue
  → report_sender: resize to 320x240, JPEG quality 40
  → Telegram Bot API sendPhoto (timeout=30s, 3 retries)
  → User's phone
```

### Planned data flow (Phase 3)
```
Pi → HTTPS POST to Cloud API
  → API saves to PostgreSQL + pushes to Redis queue
  → Report worker pops queue → sends Telegram
  → User's phone
```

---

## Current State

### What is built and working

**Project file structure:**
```
smartbot/
  main.py                     Entry point. Starts all 3 services.
  requirements.txt            opencv-python-headless, requests
  config/
    settings.json             All config (NOT in git — contains credentials)
    settings.example.json     Placeholder version for git
  services/
    camera_service.py         picamera2 on Pi / cv2.VideoCapture on laptop
    detection_service.py      cv2.dnn ONNX inference, cooldown logic, annotation
    console_reporter.py       Phase 1 only — prints detections to terminal
    report_sender.py          Phase 2 — sends photo + caption to Telegram
  utils/
    logger.py                 Console + file logging, no external imports
    config_loader.py          Loads settings.json
  logs/
    smartbot.log              Application logs
    reports/                  Saved annotated JPEG images
  yolov8n.onnx                Model file (NOT in git — generated locally)
```

**settings.json structure:**
```json
{
  "device": {
    "device_id": "dev_pi_001",
    "device_name": "Pi Zero 2W"
  },
  "camera": {
    "source": 0,
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
  "telegram": {
    "bot_token": "REAL_TOKEN_HERE",
    "chat_id": "REAL_CHAT_ID_HERE"
  },
  "preview": {
    "enabled": false,
    "window_title": "SmartBot — Debug Preview"
  },
  "environment": "development"
}
```

**Pi setup commands (what was done to get Pi working):**
```bash
sudo apt update
sudo apt install -y python3-picamera2
cd /home/pi/smartbot
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install opencv-python-headless requests
```

**ONNX model generation (run once on laptop):**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640)"
```

**Pi network:** 192.168.31.x range, accessible via SSH as pi@[IP]

### Notion project tracker
https://www.notion.so/3230745436cd81798cacf804c63a8d4a

Sub-pages:
- README (full setup guide)
- Claude Handoff — Continue From Here (session resume prompt)

---

## Key Design Decisions

### 1. Telegram instead of a mobile app
**Decision:** Use Telegram Bot for MVP delivery instead of building iOS/Android app.
**Reasoning:** Validates the AI detection value before investing months in frontend development. Users already have Telegram. Photo + text reports work perfectly via Bot API. An app can be added later once the product is proven. This saved an estimated 3-4 months of development.

### 2. ONNX via cv2.dnn instead of ultralytics on Pi
**Decision:** Export YOLOv8 to ONNX on laptop, run via cv2.dnn on Pi.
**Reasoning:** Pi Zero 2W has 512MB RAM. Ultralytics requires PyTorch which is ~500MB and consistently crashes the Pi during install. cv2.dnn is built into opencv-python-headless (already installed), adds zero extra dependencies, and runs the same model with identical accuracy. Inference is 5-10s on Pi Zero but acceptable for 10s capture intervals.

### 3. picamera2 installed via apt, not pip
**Decision:** `sudo apt install python3-picamera2`, venv with `--system-site-packages`.
**Reasoning:** libcamera is a system C++ library. It has no pip package — `pip install picamera2` pulls it as a dependency but the underlying C library still needs to be system-installed. The cleanest approach is apt install and --system-site-packages venv. Trying to pip install it fails with `ModuleNotFoundError: No module named 'libcamera'`.

### 4. Single VPS + Docker Compose (not Kubernetes)
**Decision:** One server, Docker Compose, all services together.
**Reasoning:** At MVP scale (50-100 devices), Kubernetes is massive overkill. Docker Compose is simpler to manage, easier to debug, and costs a fraction. Can migrate to Kubernetes if scale demands it.

### 5. PostgreSQL (not MongoDB)
**Decision:** Relational database.
**Reasoning:** The data model is inherently relational: users own devices, devices generate reports, users have settings. Foreign keys and joins are the right tool. MongoDB adds flexibility that isn't needed here.

### 6. Pi does NOT control robot movement
**Decision:** Passive observation only — camera rides on the robot, no movement control.
**Reasoning:** Reverse engineering robot movement APIs is a separate complex project. The core value (AI detection + alerts) can be validated without it. This keeps scope realistic for MVP.

### 7. Image compression before Telegram upload
**Decision:** Resize to 320x240, JPEG quality 40 before sending.
**Reasoning:** Pi Zero 2W has a weak single-core WiFi antenna. Full 480x360 frames at quality 85 timed out consistently (15s timeout hit). Compressed images upload in 2-5 seconds. Quality is sufficient for a detection alert viewed on a phone screen.

---

## Open Questions & Unresolved Items

### Immediate (Phase 3)
- VPS not yet accessible — no deployment target. Phase 3 cloud backend being built and tested locally first.
- Docker not yet learned — study needed before VPS deployment step.
- API key strategy for device auth — how to generate, store (hashed), and rotate device API keys.

### Medium term
- Motion pre-filter not yet implemented — planned optimisation to skip YOLO inference on frames with no pixel change. Reduces Pi CPU load significantly.
- Text-only Telegram fallback not yet implemented — if photo upload fails 2 times, send text-only message instead.
- Custom model training not started — current model is pre-trained COCO. Floor-level robot camera perspective (low angle, moving) would benefit from fine-tuning. RTX 3060 available for this.

### Future / unresolved
- Monetisation strategy not designed — subscription tiers, pricing not decided.
- Multi-user / multi-device architecture — one user, one device in MVP. Multi-device support requires device registration flow in Phase 3.
- Privacy policy for person detection — person detection is in target_labels but is sensitive. Should be opt-in with clear disclosure.

---

## Constraints & Requirements

### Hard hardware constraints
- Pi Zero 2W: 512MB RAM maximum. No PyTorch, no large pip packages.
- Pi Zero 2W: Single-core WiFi. Images must be compressed before upload.
- Pi Zero 2W: 5-10s inference time. Capture interval must be ≥10s.
- picamera2/libcamera: System-level dependency. Venv must use --system-site-packages.
- SD card: Avoid excessive disk writes. Images kept in memory where possible, not written to disk during inference.

### Non-negotiables
- Telegram credentials (bot_token, chat_id) must NEVER be committed to git.
- ONNX and .pt model files must be in .gitignore (large binaries).
- settings.json must be in .gitignore — contains credentials.
- venv/ and __pycache__/ must be in .gitignore.

### .gitignore (confirmed)
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

### Developer background
- Medium Python experience, some ML/CV background.
- Very little JavaScript/TypeScript — learning as we build Phase 3.
- No Docker experience — needs study before deployment step.
- No Linux server administration experience — needs study before VPS step.
- Prefers thorough planning and explanation before implementation.
- Wants study resources included when learning new technologies.
- Values honest assessments over purely positive feedback.

---

## Planned Next Steps

### Immediate: Phase 3, Step 1 — Study TypeScript + Node.js
Before writing any backend code:
- Read TypeScript handbook intro (first 4 sections): typescriptlang.org/docs/handbook/intro.html
- Watch Node.js crash course (Traversy Media on YouTube, 90 min)
- Understand async/await pattern specifically — this is everywhere in Node.js backend code

Install now while studying:
- Node.js LTS from nodejs.org
- Postman from postman.com (for testing API endpoints)
- VS Code with TypeScript and Prisma extensions

### Phase 3, Step 2 — Build Express API locally
- Set up Node.js + TypeScript project from scratch
- Install Express and configure TypeScript
- Build `POST /device/report` endpoint (accepts JSON detection data, returns success)
- Test with Postman — send fake detection payloads, confirm correct responses
- Reference: expressjs.com Getting Started guide

### Phase 3, Step 3 — PostgreSQL + Prisma locally
- Install PostgreSQL on laptop
- Read Prisma Getting Started: prisma.io/docs/getting-started
- Define schema: users, devices, reports tables (schema documented in blueprint)
- Connect Prisma to Express API
- Test: POST to API → data appears in database

### Phase 3, Step 4 — Redis + Report Worker locally
- Install Redis on laptop
- API pushes report_id to Redis queue after saving to DB
- Build report worker process: reads from queue, sends Telegram message
- Test full local flow: fake POST → DB saved → queue pushed → Telegram message received

### Phase 3, Step 5 — VPS deployment (blocked, no VPS access yet)
- Rent VPS (Hetzner EU ~€5-10/month, or DigitalOcean ~$20/month)
- Study Docker: docs.docker.com/get-started (official tutorial, ~3 hours)
- Set up Docker Compose with all services
- Configure Nginx + Let's Encrypt SSL
- Deploy and test

### Phase 3, Step 6 — Update Pi (final step)
- Modify report_sender.py to POST to cloud API instead of Telegram directly
- Add device_id and api_key to settings.json
- Test end-to-end: Pi detection → cloud API → DB → queue → worker → Telegram

### Pending optimisations (can do anytime in parallel)
- Add motion pre-filter to camera_service.py (skip frames with no movement)
- Add text-only Telegram fallback in report_sender.py (when photo upload fails)
- Lower log level to WARNING on Pi to reduce SD card writes

---

## Important Background Context

### Why the Pi Zero 2W was chosen
It was the available hardware. It's not ideal (Pi 4 with 2GB+ would be better for inference speed) but it's sufficient for the use case — periodic detection every 10 seconds, not real-time video. The ONNX approach specifically solves the RAM constraint.

### Why the project starts with "passive observation"
The robot already moves through the home on its cleaning schedule. Mounting a camera and running AI on the footage requires zero changes to the robot's behavior. Controlling movement would require reverse-engineering proprietary firmware — a completely separate project. This passive approach lets the product be validated quickly.

### The Telegram feedback buttons plan
Each Telegram report should eventually have inline keyboard buttons: 👍 Correct, 👎 Wrong, 🔕 Mute 1hr. The 👍/👎 feedback is stored in the database and used later for model fine-tuning. This closes the product loop: real user feedback → better model → better product. Not yet implemented.

### Model training plan for Phase 4
The pre-trained COCO model detects cats and dogs well in normal photos but was trained on human-photographed images (eye level, good lighting). Robot camera footage is floor-level, moving, often poorly lit. Fine-tuning on real robot footage using Roboflow (labeling) + RTX 3060 (training) is planned for Phase 4. Until then the pre-trained model is good enough for MVP.

### How detection decisions work
A Telegram report fires only when ALL THREE pass:
1. YOLO confidence ≥ confidence_threshold (default 0.60)
2. Detected label is in target_labels list (default: cat, dog, person, bird)
3. Same label has not been reported within cooldown_seconds (default: 300s = 5 min)

Cooldown is per-label and stored in memory (dict in detection_service.py). It resets if the Pi restarts. This is intentional for MVP — persistent cooldown across restarts is not worth the complexity yet.

### Session continuity
Claude stores project memories automatically between sessions. For full technical continuity, a handoff prompt is maintained at:
https://www.notion.so/3320745436cd819c9139c9900ed6cdf0

Paste the contents of that page as the first message in a new Claude chat session to restore full context.
