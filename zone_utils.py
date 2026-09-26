"""
Helper functions for zone-based trigger logic (Option A).
"""

def box_center(box_xyxy):
    """Return (cx, cy) center point of a box given (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box_xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def in_assembly_zone(box_xyxy, zone):
    """
    Check whether a detection's center point falls inside the assembly zone.
    box_xyxy: (x1, y1, x2, y2) of the detected object
    zone: (zx1, zy1, zx2, zy2) of the assembly area
    """
    cx, cy = box_center(box_xyxy)
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2
