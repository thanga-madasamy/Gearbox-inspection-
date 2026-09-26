"""
Central configuration for the Bolt & Washer Detection/Counting project.
Change values here instead of digging through main.py.
"""

# ---- Model ----
MODEL_PATH = "models/your_trained_model.pt"   # path to your trained YOLOv8 weights

# ---- Video source ----
VIDEO_SOURCE = 0   # 0 = default webcam, or path to a video file e.g. "video_stream.mp4"

# ---- Tracker ----
TRACKER_CONFIG = "bytetrack.yaml"   # ships with Ultralytics, no extra install needed
CONF_THRESHOLD = 0.3                # lower = catches more low-confidence boxes
IOU_THRESHOLD = 0.5                 # tune if IDs are switching between close-together bolts
TRACK_BUFFER = 30                   # frames an object can be "lost" before ID is dropped

# ---- Class IDs (must match your model's training labels) ----
BOLT_CLASS = 0
WASHER_CLASS = 1

# ---- Counting targets ----
TOTAL_BOLTS = 32
TOTAL_WASHERS = 32

# ---- Auto-start trigger ----
# "zone"  = Option A: first bolt entering the assembly zone starts the count
# "hand"  = Option B: hand + bolt proximity starts the count (needs a hand-detection model)
# "tray"  = Option C: tray count dropping from 32 starts the count
START_TRIGGER_MODE = "zone"

# Assembly zone as (x1, y1, x2, y2) in pixel coordinates — adjust to your camera framing
ASSEMBLY_ZONE = (200, 150, 500, 450)
