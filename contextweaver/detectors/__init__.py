"""
Context Injection Detectors

Detect optimal moments for context injection.
"""

from .base import BaseDetector, DetectionResult
from .silence_detector import SilenceDetector
from .pause_detector import PauseDetector
from .topic_change_detector import TopicChangeDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "SilenceDetector",
    "PauseDetector", 
    "TopicChangeDetector"
]