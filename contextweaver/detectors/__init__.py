"""
Context Injection Detectors

Detect optimal moments for context injection.
"""

from .base import BaseDetector, DetectionResult
from .silence_detector import SilenceDetector
from .pause_detector import PauseDetector
from .topic_change_detector import TopicChangeDetector
from .response_timing_detector import ResponseTimingDetector
from .conversation_flow_detector import ConversationFlowDetector, ConversationPhase

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "SilenceDetector",
    "PauseDetector", 
    "TopicChangeDetector",
    "ResponseTimingDetector",
    "ConversationFlowDetector",
    "ConversationPhase"
]