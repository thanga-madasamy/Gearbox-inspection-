"""
Bolt & Washer Detection / Counting Pipeline
--------------------------------------------
Flow:
  Camera frame -> YOLOv8 detects bolts/washers -> ByteTrack assigns IDs
  -> auto-start trigger on first bolt in assembly zone
  -> unique-ID counting until 32/32 reached -> pass/fail check

Run:
  python main.py
"""

import time
from ultralytics import YOLO

import config
from zone_utils import in_assembly_zone

# ---- State ----
started = False
start_time = None
bolt_count = 0
washer_count = 0
tracked_bolt_ids = set()
tracked_washer_ids = set()


def handle_detection(box):
    """Process a single tracked detection box for one frame."""
    global started, start_time, bolt_count, washer_count

    if box.id is None:
        return  # tracker hasn't assigned an ID yet, skip

    track_id = int(box.id)
    cls = int(box.cls)
    xyxy = box.xyxy[0].tolist()

    is_bolt = cls == config.BOLT_CLASS
    is_washer = cls == config.WASHER_CLASS

    # --- Auto-start trigger (Option A: zone-based) ---
    if not started and is_bolt and in_assembly_zone(xyxy, config.ASSEMBLY_ZONE):
        started = True
        start_time = time.time()
        print("Auto-start triggered — first bolt detected in assembly zone")

    if not started:
        return  # don't count anything before the trigger fires

    # --- Counting ---
    if is_bolt and track_id not in tracked_bolt_ids:
        tracked_bolt_ids.add(track_id)
        bolt_count += 1
        print(f"Bolt {bolt_count}/{config.TOTAL_BOLTS} counted (ID {track_id})")

    if is_washer and track_id not in tracked_washer_ids:
        tracked_washer_ids.add(track_id)
        washer_count += 1
        print(f"Washer {washer_count}/{config.TOTAL_WASHERS} counted (ID {track_id})")


def run():
    model = YOLO(config.MODEL_PATH)

    results = model.track(
        source=config.VIDEO_SOURCE,
        tracker=config.TRACKER_CONFIG,
        conf=config.CONF_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        persist=True,
        stream=True,
    )

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            handle_detection(box)

        if bolt_count >= config.TOTAL_BOLTS and washer_count >= config.TOTAL_WASHERS:
            cycle_time = time.time() - start_time
            print(f"\nAll {config.TOTAL_BOLTS} bolts and {config.TOTAL_WASHERS} washers detected.")
            print(f"Cycle time: {cycle_time:.1f}s")
            print("Result: PASS")
            break


if __name__ == "__main__":
    run()
