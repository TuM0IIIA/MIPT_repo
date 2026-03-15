"""
main.py — SmartBot Phase 1

Starts three background services:
  CameraService     — captures frames
  DetectionService  — runs YOLOv8 inference
  ConsoleReporter   — prints results + saves images

If preview.enabled = true in settings.json, also opens a live
cv2 window showing every frame with bounding boxes drawn.
The preview window runs on the MAIN THREAD (OpenCV requirement).

Controls:
  Q  or  ESC  — quit
"""

import queue
import signal
import sys
import time

import cv2

from services.camera_service    import CameraService
from services.detection_service import DetectionService
from services.console_reporter  import ConsoleReporter
from utils.config_loader        import load_config
from utils.logger               import get_logger

logger = get_logger("main")


def main():
    logger.info("=" * 52)
    logger.info("  SmartBot Phase 1 — Starting")
    logger.info("=" * 52)

    config = load_config("config/settings.json")
    preview_enabled = config.get("preview", {}).get("enabled", False)
    window_title    = config.get("preview", {}).get("window_title", "SmartBot Preview")

    # ── Queues ──────────────────────────────────────────────────
    frame_queue   = queue.Queue(maxsize=2)
    result_queue  = queue.Queue(maxsize=10)
    preview_queue = queue.Queue(maxsize=2) if preview_enabled else None

    # ── Services ────────────────────────────────────────────────
    camera   = CameraService(config["camera"], frame_queue)
    detector = DetectionService(config["detection"], frame_queue, result_queue, preview_queue)
    reporter = ConsoleReporter(result_queue)

    # ── Graceful shutdown ────────────────────────────────────────
    def shutdown(sig=None, frame=None):
        print("")
        logger.info("Shutting down...")
        reporter.stop()
        detector.stop()
        camera.stop()
        if preview_enabled:
            cv2.destroyAllWindows()
        logger.info("All services stopped. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Start background services ────────────────────────────────
    try:
        logger.info("Starting reporter...")
        reporter.start()

        logger.info("Loading AI model (first run downloads ~6MB)...")
        detector.start()

        logger.info("Starting camera...")
        camera.start()

    except RuntimeError as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)

    # ── Print startup summary ────────────────────────────────────
    logger.info("")
    logger.info("All systems running!")
    logger.info(f"  Capture interval : {config['camera']['capture_interval_seconds']}s")
    logger.info(f"  Watching for     : {config['detection']['target_labels']}")
    logger.info(f"  Confidence min   : {config['detection']['confidence_threshold']:.0%}")
    logger.info(f"  Cooldown         : {config['detection']['cooldown_seconds']}s")
    if preview_enabled:
        logger.info(f"  Preview window   : ENABLED  (press Q or ESC to quit)")
    else:
        logger.info(f"  Preview window   : off  (set preview.enabled=true to enable)")
    logger.info(f"  Saved images     : logs/reports/")
    logger.info("")

    # ── Main loop ────────────────────────────────────────────────
    # OpenCV imshow MUST be called from the main thread.
    # We block here, polling the preview_queue and refreshing the window.

    if preview_enabled:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 800, 600)

    while True:
        if preview_enabled:
            # Try to get the latest annotated frame
            frame_to_show = None
            try:
                frame_to_show = preview_queue.get_nowait()
            except queue.Empty:
                pass

            if frame_to_show is not None:
                cv2.imshow(window_title, frame_to_show)

            # waitKey keeps the window alive; also checks for Q / ESC
            key = cv2.waitKey(30) & 0xFF   # 30ms = ~33fps refresh
            if key in (ord('q'), ord('Q'), 27):   # 27 = ESC
                logger.info("Preview window closed by user.")
                shutdown()
        else:
            time.sleep(5)

        # Health check
        if not camera.is_running():
            logger.error("Camera service crashed!")
        if not detector.is_running():
            logger.error("Detection service crashed!")


if __name__ == "__main__":
    main()
