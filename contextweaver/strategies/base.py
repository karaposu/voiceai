"""
Base Strategy for Context Injection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


@dataclass
class InjectionDecision:
    """Decision about whether and how to inject context"""
    should_inject: bool
    context_to_inject: Optional['ContextToInject']  # Forward reference
    priority: float  # 0.0 to 1.0
    reason: str
    timing: str  # "immediate", "delayed", "next_turn"
    

class InjectionStrategy(ABC):
    """Base class for injection strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.injection_history: List[datetime] = []
        self.last_injection: Optional[datetime] = None
    
    @abstractmethod
    async def decide(
        self,
        detections: List['DetectionResult'],
        state: Any,
        available_context: Dict[str, 'ContextToInject']
    ) -> InjectionDecision:
        """
        Decide whether to inject context based on detections.
        
        Args:
            detections: Results from all detectors
            state: Current conversation state
            available_context: Context available for injection
            
        Returns:
            InjectionDecision
        """
        pass
    
    def record_injection(self):
        """Record that an injection occurred"""
        now = datetime.now()
        self.injection_history.append(now)
        self.last_injection = now
        
        # Keep only last 100 injections
        if len(self.injection_history) > 100:
            self.injection_history = self.injection_history[-100:]
    
    def get_injection_rate(self, window_seconds: int = 60) -> float:
        """Get injection rate in the time window"""
        if not self.injection_history:
            return 0.0
            
        now = datetime.now()
        recent = [
            inj for inj in self.injection_history
            if (now - inj).total_seconds() <= window_seconds
        ]
        
        return len(recent) / window_seconds