
# Phase 1 Setup Guide

## What This Does
Captures frames from your webcam, runs YOLOv8 object detection,
and prints a report whenever it detects a cat, dog, or person.
Saves annotated images to logs/reports/.

---

## Prerequisites
- Python 3.9 or higher
- A working webcam
- Internet connection for first run (downloads YOLOv8 model, about 6MB)

---

## Setup (Do This Once)

### 1. Create a virtual environment
From the smartbot/ folder:
    python3 -m venv venv

### 2. Activate it
Mac/Linux:   source venv/bin/activate
Windows:     venv\Scripts\activate

You will see (venv) in your terminal prompt when it is active.

### 3. Install dependencies
    pip install -r requirements.txt

Takes 2-5 minutes. Installs OpenCV and YOLOv8.

---

## Run the System

    python main.py

First run will download the YOLOv8n model (~6MB). This happens once.
Stop with Ctrl+C.

---

## Configuration: config/settings.json

camera.source                  0 means default webcam. Try 1 or 2 if wrong camera opens.
camera.capture_interval_seconds  How often to capture a frame. 3 = every 3 seconds.
detection.confidence_threshold   0.60 = 60% confidence required. Lower = more detections.
detection.target_labels          List of objects to watch for and alert on.
detection.cooldown_seconds       300 = wait 5 min before alerting same object again.

TIP: Set cooldown_seconds to 10 for quick testing, 300 for real use.

---

## Output
Annotated images saved to: logs/reports/
Full logs saved to:         logs/smartbot.log

---

## Troubleshooting

ERROR: Could not open camera
  - Another app is using your webcam (Zoom, Teams, browser). Close it first.
  - Try changing camera.source to 1 in settings.json

Nothing is being detected
  - Lower confidence_threshold to 0.40 in settings.json
  - Make sure the object is well lit and clearly in frame
  - Check logs/reports/ to see what images look like

Detects objects but no report prints
  - Check target_labels - is your object in the list?
  - Check cooldown_seconds - maybe same object detected recently. Set to 10 to test.

---

## Project Structure

  smartbot/
    main.py                    <- Run this
    requirements.txt           <- Install this
    config/
      settings.json            <- All configuration
    services/
      camera_service.py        <- Captures frames from camera
      detection_service.py     <- Runs YOLOv8, applies cooldown
      console_reporter.py      <- Prints reports and saves images (Phase 1)
    utils/
      logger.py
      config_loader.py
    logs/
      smartbot.log             <- Full logs
      reports/                 <- Saved annotated images

---

## Moving to Raspberry Pi Zero 2W

1. Copy the entire smartbot/ folder to Pi via SCP or USB
2. Run the same setup steps (python3 -m venv, pip install)
3. Change capture_interval_seconds to 10 (Pi Zero is slower)
4. For Pi Camera Module: swap cv2.VideoCapture for picamera2
   (instructions are in comments inside camera_service.py)
