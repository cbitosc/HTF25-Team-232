"""
test_signal_detection.py
Quick test to verify signal jump detection is working
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from violations.rules import check_signal_jump

# Test data: car inside stop line zone
test_detections = [
    {
        "track_id": 101,
        "class": "car",
        "bbox": [250, 400, 350, 500],  # Car bbox
        "confidence": 0.95
    },
    {
        "track_id": 102,
        "class": "motorbike",
        "bbox": [150, 450, 250, 550],  # Bike bbox
        "confidence": 0.88
    },
    {
        "track_id": 103,
        "class": "person",
        "bbox": [50, 300, 100, 400],  # Person outside zone
        "confidence": 0.92
    }
]

# Stop line polygon (cars inside this when red = violation)
STOP_LINE_POLYGON = [(100, 500), (400, 500), (400, 520), (100, 520)]

print("🧪 Testing Signal Jump Detection\n")
print("=" * 60)

# Test 1: Red light - should detect violations
print("\n📍 Test 1: RED light with vehicles in stop zone")
print("-" * 60)
violations = check_signal_jump(
    detections=test_detections,
    traffic_light_state="RED",
    stop_line_polygon=STOP_LINE_POLYGON,
    min_conf_vehicle=0.4
)

if violations:
    print(f"✅ Detected {len(violations)} violation(s):")
    for v in violations:
        print(f"   - Track ID {v['track_id']}: {v['details']['class']} at {v['details']['center']}")
else:
    print("❌ No violations detected (BUG!)")

# Test 2: Green light - should NOT detect violations
print("\n📍 Test 2: GREEN light with vehicles in stop zone")
print("-" * 60)
violations = check_signal_jump(
    detections=test_detections,
    traffic_light_state="GREEN",
    stop_line_polygon=STOP_LINE_POLYGON,
    min_conf_vehicle=0.4
)

if not violations:
    print("✅ Correctly ignored vehicles (light is green)")
else:
    print(f"❌ False positive: detected {len(violations)} violations on green light")

# Test 3: Vehicle outside stop zone
print("\n📍 Test 3: RED light with vehicle OUTSIDE stop zone")
print("-" * 60)
test_detections_outside = [
    {
        "track_id": 200,
        "class": "car",
        "bbox": [50, 200, 150, 300],  # Way above stop line
        "confidence": 0.95
    }
]

violations = check_signal_jump(
    detections=test_detections_outside,
    traffic_light_state="RED",
    stop_line_polygon=STOP_LINE_POLYGON,
    min_conf_vehicle=0.4
)

if not violations:
    print("✅ Correctly ignored vehicle outside stop zone")
else:
    print(f"❌ False positive: detected vehicle outside zone")

print("\n" + "=" * 60)
print("✅ Signal detection tests complete!\n")

# Helper: visualize the stop zone
print("📊 Stop Line Polygon Coordinates:")
print(f"   {STOP_LINE_POLYGON}")
print("\n💡 Make sure your video's stop line matches these coordinates!")
print("   You can adjust STOP_LINE_POLYGON in run_video.py")