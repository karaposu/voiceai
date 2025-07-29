"""
Base Detector Interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class DetectionResult:
    """Result from a detector"""
    detected: bool
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    metadata: Dict[str, Any]
    
    @property
    def should_inject(self) -> bool:
        """Whether confidence is high enough to inject"""
        return self.detected and self.confidence > 0.7


class BaseDetector(ABC):
    """Base class for all context injection detectors"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.last_detection = None
    
    @abstractmethod
    async def detect(self, state: Any) -> DetectionResult:
        """
        Detect if context should be injected.
        
        Args:
            state: Current conversation state
            
        Returns:
            DetectionResult indicating if injection should occur
        """
        pass
    
    def reset(self):
        """Reset detector state"""
        self.last_detection = None