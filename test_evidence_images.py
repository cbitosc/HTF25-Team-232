"""
test_evidence_images.py
Test that evidence images are being generated with proper red boxes
"""

import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from src.violations.evidence_generator import _draw_event_overlay

# Create a test frame (blank white image)
test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255

# Draw some reference elements
cv2.putText(test_frame, "TEST FRAME", (500, 360), 
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

print("🧪 Testing Evidence Image Generation\n")
print("=" * 60)

# Test 1: Helmetless violation
print("\n📍 Test 1: Helmetless Violation Evidence")
print("-" * 60)

helmetless_event = {
    "type": "helmetless",
    "track_id": 42,
    "confidence": 0.92,
    "timestamp_utc": "2025-10-26T10:30:00Z",
    "bike_bbox": [400, 200, 600, 500],
    "riders": [
        {
            "bbox": [420, 220, 480, 350],
            "helmet": False,
            "helmet_confidence": 0.92
        }
    ],
    "details": {
        "class": "person",
        "helmet_confidence": 0.92
    }
}

annotated_helmetless = _draw_event_overlay(test_frame.copy(), helmetless_event)
output_path = "test_evidence_helmetless.jpg"
cv2.imwrite(output_path, annotated_helmetless)
print(f"✅ Generated: {output_path}")
print("   - Should show thick RED box around bike")
print("   - Should show '⚠️ HELMETLESS' label")
print("   - Should show 'NO HELMET' tag on rider")

# Test 2: Red light jump violation
print("\n📍 Test 2: Red Light Jump Violation Evidence")
print("-" * 60)

signal_event = {
    "type": "red_light_jump",
    "track_id": 101,
    "confidence": 0.95,
    "timestamp_utc": "2025-10-26T10:35:00Z",
    "vehicle_bbox": [700, 300, 900, 550],
    "details": {
        "class": "car",
        "traffic_light_state": "RED",
        "center": [800, 425]
    }
}

annotated_signal = _draw_event_overlay(test_frame.copy(), signal_event)
output_path2 = "test_evidence_signal_jump.jpg"
cv2.imwrite(output_path2, annotated_signal)
print(f"✅ Generated: {output_path2}")
print("   - Should show thick RED box around car")
print("   - Should show '🚨 RED LIGHT JUMP' label")
print("   - Should show 'CAR' below the box")

# Test 3: Triple riding violation
print("\n📍 Test 3: Triple Riding Violation Evidence")
print("-" * 60)

triple_event = {
    "type": "triple_riding",
    "track_id": 15,
    "confidence": 1.0,
    "timestamp_utc": "2025-10-26T10:40:00Z",
    "bike_bbox": [200, 150, 450, 500],
    "rider_count": 3,
    "riders": [
        {"bbox": [220, 170, 280, 300], "helmet": True},
        {"bbox": [280, 180, 340, 310], "helmet": False},
        {"bbox": [340, 190, 400, 320], "helmet": False}
    ],
    "details": {
        "rider_count": 3
    }
}

annotated_triple = _draw_event_overlay(test_frame.copy(), triple_event)
output_path3 = "test_evidence_triple_riding.jpg"
cv2.imwrite(output_path3, annotated_triple)
print(f"✅ Generated: {output_path3}")
print("   - Should show thick RED box around bike")
print("   - Should show '⚠️ TRIPLE RIDING' label")
print("   - Should show 3 rider boxes with helmet status")

print("\n" + "=" * 60)
print("✅ Evidence image tests complete!")
print(f"\n📁 Check the generated images:")
print(f"   - {output_path}")
print(f"   - {output_path2}")
print(f"   - {output_path3}")
print("\n💡 These images should have:")
print("   ✓ Thick RED boxes (4px)")
print("   ✓ Large warning labels with red background")
print("   ✓ Semi-transparent red overlay on violation area")
print("   ✓ Timestamp and confidence at bottom")
print("   ✓ 'EVIDENCE' watermark at top")