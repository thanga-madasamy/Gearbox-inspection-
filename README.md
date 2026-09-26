# Bolt & Washer Detection Project

Auto-counts 32 bolts and 32 washers during assembly using YOLOv8 + ByteTrack,
with an automatic start trigger (no manual "Start" button needed).

## Folder structure

```
bolt_washer_project/
├── config.py        # all tunable settings (model path, thresholds, zone, etc.)
├── zone_utils.py     # helper functions for zone-based trigger logic
├── main.py           # main pipeline: detect -> track -> auto-start -> count
├── requirements.txt
├── models/           # put your trained .pt weights file here
└── logs/             # (empty for now — add logging here later if needed)
```

## Setup

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Put your trained YOLOv8 model file into `models/` and update
   `MODEL_PATH` in `config.py` if the filename is different.

4. Adjust `ASSEMBLY_ZONE` in `config.py` to match where bolts appear
   in your camera frame (x1, y1, x2, y2 in pixels).

5. Run it:
   ```
   python main.py
   ```

## How it works

1. The camera/video runs continuously in standby.
2. The moment a bolt is detected inside `ASSEMBLY_ZONE`, the auto-start
   trigger fires (`START_TRIGGER_MODE = "zone"` in config.py — this is
   Option A, the simplest of the three approaches we discussed).
3. ByteTrack assigns a unique ID to each bolt/washer so each one is
   only counted once, even across multiple frames.
4. Once 32/32 bolts and 32/32 washers are counted, the cycle time is
   printed and a PASS result is reported.

## Next steps to customize

- **Switch trigger mode**: if zone-detection gives false starts, move to
  Option B (hand + bolt proximity) — this needs a hand-detection class
  added to your training data.
- **Tune IOU/track buffer** in `config.py` if you see ID switching between
  bolts that pass close together.
- **Add logging**: write counts/timestamps to `logs/` for traceability.
