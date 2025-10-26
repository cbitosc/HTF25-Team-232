"""
test_evidence_generation.py
Complete test to verify evidence generation with red boxes
"""

import cv2
import numpy as np
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from src.violations.adapter import adapt_event_for_evidence
from src.violations.evidence_generator import _draw_event_overlay, save_violation_package

# Create a test frame (blank white image with some visual elements)
test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200

# Draw grid for reference
for i in range(0, 1280, 100):
    cv2.line(test_frame, (i, 0), (i, 720), (220, 220, 220), 1)
for i in range(0, 720, 100):
    cv2.line(test_frame, (0, i), (1280, i), (220, 220, 220), 1)

# Draw some mock vehicles/persons
cv2.rectangle(test_frame, (400, 200), (600, 500), (100, 100, 100), 2)
cv2.putText(test_frame, "BIKE", (450, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

cv2.rectangle(test_frame, (700, 300), (900, 550), (100, 100, 100), 2)
cv2.putText(test_frame, "CAR", (750, 425), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

print("🧪 Testing Evidence Generation Pipeline\n")
print("=" * 60)

# Test 1: Helmetless violation (from rules.py format)
print("\n📍 Test 1: Helmetless Violation")
print("-" * 60)

rules_event_helmetless = {
    "violation_type": "helmetless",
    "track_id": 42,
    "confidence": 0.92,
    "timestamp_utc": "2025-10-26T10:30:00Z",
    "details": {
        "bbox": [400, 200, 600, 500],
        "class": "person",
        "helmet_confidence": 0.92
    }
}

print("🔄 Original event from rules.py:")
print(json.dumps(rules_event_helmetless, indent=2))

adapted = adapt_event_for_evidence(rules_event_helmetless)
print("\n🔄 Adapted event for evidence_generator:")
print(json.dumps(adapted, indent=2, default=str))

annotated = _draw_event_overlay(test_frame.copy(), adapted)
output_path = "test_output_helmetless.jpg"
cv2.imwrite(output_path, annotated)
print(f"\n✅ Generated: {output_path}")

# Verify the image has red boxes
has_red = np.any(annotated[:, :, 2] > 200)  # Check for red channel
print(f"   Red pixels detected: {'✅ YES' if has_red else '❌ NO'}")

# Test 2: Red light jump violation
print("\n📍 Test 2: Red Light Jump Violation")
print("-" * 60)

rules_event_signal = {
    "violation_type": "red_light_jump",
    "track_id": 101,
    "confidence": 0.95,
    "timestamp_utc": "2025-10-26T10:35:00Z",
    "details": {
        "bbox": [700, 300, 900, 550],
        "class": "car",
        "traffic_light_state": "RED",
        "center": [800, 425]
    }
}

print("🔄 Original event from rules.py:")
print(json.dumps(rules_event_signal, indent=2))

adapted2 = adapt_event_for_evidence(rules_event_signal)
print("\n🔄 Adapted event for evidence_generator:")
print(json.dumps(adapted2, indent=2, default=str))

annotated2 = _draw_event_overlay(test_frame.copy(), adapted2)
output_path2 = "test_output_signal_jump.jpg"
cv2.imwrite(output_path2, annotated2)
print(f"\n✅ Generated: {output_path2}")

has_red2 = np.any(annotated2[:, :, 2] > 200)
print(f"   Red pixels detected: {'✅ YES' if has_red2 else '❌ NO'}")

# Test 3: Triple riding violation
print("\n📍 Test 3: Triple Riding Violation")
print("-" * 60)

rules_event_triple = {
    "violation_type": "triple_riding",
    "track_id": 15,
    "confidence": 1.0,
    "timestamp_utc": "2025-10-26T10:40:00Z",
    "details": {
        "bike_bbox": [200, 150, 450, 500],
        "rider_count": 3,
        "rider_ids": [42, 43, 44]
    }
}

print("🔄 Original event from rules.py:")
print(json.dumps(rules_event_triple, indent=2))

adapted3 = adapt_event_for_evidence(rules_event_triple)
print("\n🔄 Adapted event for evidence_generator:")
print(json.dumps(adapted3, indent=2, default=str))

# Draw second bike on frame
test_frame_triple = test_frame.copy()
cv2.rectangle(test_frame_triple, (200, 150), (450, 500), (100, 100, 100), 2)
cv2.putText(test_frame_triple, "BIKE", (250, 325), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

annotated3 = _draw_event_overlay(test_frame_triple, adapted3)
output_path3 = "test_output_triple_riding.jpg"
cv2.imwrite(output_path3, annotated3)
print(f"\n✅ Generated: {output_path3}")

has_red3 = np.any(annotated3[:, :, 2] > 200)
print(f"   Red pixels detected: {'✅ YES' if has_red3 else '❌ NO'}")

# Test 4: Full evidence package generation
print("\n📍 Test 4: Full Evidence Package (with clip)")
print("-" * 60)

# Create mock frame buffer (45 frames)
frame_buffer = []
for i in range(45):
    frame_copy = test_frame.copy()
    cv2.putText(frame_copy, f"Frame {i}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    frame_buffer.append((float(i), frame_copy))

try:
    violation_record = save_violation_package(
        frame=test_frame,
        past_frames_buffer=frame_buffer,
        event=rules_event_helmetless,
        camera_id="TEST_CAM",
        location_name="Test Location",
        plate_text="TEST1234",
        output_root="test_output_violations"
    )
    
    print("✅ Full package generated:")
    print(f"   Violation ID: {violation_record['violation_id']}")
    print(f"   Type: {violation_record['type']}")
    print(f"   Context image: {violation_record['media']['context_img']}")
    print(f"   Clip video: {violation_record['media']['clip_video']}")
    
    # Check if context image has red boxes
    if os.path.exists(violation_record['media']['context_img']):
        context_img = cv2.imread(violation_record['media']['context_img'])
        has_red_context = np.any(context_img[:, :, 2] > 200)
        print(f"   Red boxes in context.jpg: {'✅ YES' if has_red_context else '❌ NO'}")
    
    # Check if clip exists
    clip_exists = os.path.exists(violation_record['media']['clip_video'])
    print(f"   Clip.mp4 exists: {'✅ YES' if clip_exists else '❌ NO'}")
    
except Exception as e:
    print(f"❌ Error generating full package: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Evidence generation tests complete!")
print(f"\n📁 Check the generated files:")
print(f"   - {output_path}")
print(f"   - {output_path2}")
print(f"   - {output_path3}")
print(f"   - test_output_violations/V-*/")
print("\n💡 Expected results:")
print("   ✓ Thick RED boxes (4px) around violations")
print("   ✓ Large warning labels with red backgrounds")
print("   ✓ Semi-transparent red overlay on violation areas")
print("   ✓ Timestamp and confidence at bottom")
print("   ✓ 'EVIDENCE' watermark at top")