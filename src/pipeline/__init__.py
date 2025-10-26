"""
pipeline package

Handles the orchestration of the entire AI Traffic Violation Detector workflow:
- Reading video/CCTV feeds
- Managing frame buffers and ROI overlays
- Running detection → rule evaluation → evidence generation
- Sending structured violations to the backend

Owned primarily by Member 3 (pipeline integration).
"""

from . import frame_utils
from . import config