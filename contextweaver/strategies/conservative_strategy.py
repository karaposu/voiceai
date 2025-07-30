"""
Conservative Strategy for Context Injection

Injects context carefully to avoid disrupting conversation flow.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from .base import InjectionStrategy, InjectionDecision
from contextweaver.schema import ContextToInject, InjectionTiming, ContextPriority


class ConservativeStrategy(InjectionStrategy):
    """
    Conservative injection strategy.
    
    - Waits for natural pauses
    - Limits injection frequency
    - Prioritizes high-value contexts
    """
    
    def __init__(
        self,
        min_interval_seconds: int = 30,
        confidence_threshold: float = 0.8
    ):
        super().__init__(name="conservative")
        self.min_interval_seconds = min_interval_seconds
        self.confidence_threshold = confidence_threshold
    
    async def decide(
        self,
        detections: List['DetectionResult'],
        state: Any,
        available_context: Dict[str, ContextToInject]
    ) -> InjectionDecision:
        """
        Conservative decision making.
        
        Only injects when:
        1. High confidence detection
        2. Sufficient time since last injection
        3. High priority context available
        
        Adapts based on VAD mode if available in state.
        """
        
        # Adapt thresholds based on VAD mode if available
        vad_mode = getattr(state, 'vad_mode', None)
        auto_response = getattr(state, 'auto_response', True)
        injection_mode = getattr(state, 'injection_mode', 'adaptive')
        
        # Adjust confidence threshold for server VAD with auto-response
        effective_threshold = self.confidence_threshold
        if vad_mode == 'server' and auto_response:
            # More lenient in server+auto mode since time is limited
            effective_threshold *= 0.8
        
        # Check time since last injection
        min_interval = self.min_interval_seconds
        if injection_mode == 'immediate':
            # Reduce minimum interval for immediate mode
            min_interval = max(5, min_interval // 4)
        
        if self.last_injection:
            time_since = datetime.now() - self.last_injection
            if time_since < timedelta(seconds=min_interval):
                return InjectionDecision(
                    should_inject=False,
                    context_to_inject=None,
                    priority=0.0,
                    reason="too_soon_after_last_injection",
                    timing="delayed"
                )
        
        # Find highest confidence detection
        best_detection = None
        for detection in detections:
            if detection.detected and detection.confidence >= effective_threshold:
                if not best_detection or detection.confidence > best_detection.confidence:
                    best_detection = detection
        
        if not best_detection:
            return InjectionDecision(
                should_inject=False,
                context_to_inject=None,
                priority=0.0,
                reason="no_high_confidence_detection",
                timing="delayed"
            )
        
        # Find highest priority context that should inject
        best_context = None
        best_priority = 0
        
        for context_id, context in available_context.items():
            # Check if context should inject
            if not context.should_inject(state.__dict__ if hasattr(state, '__dict__') else state):
                continue
                
            # Check timing compatibility
            if context.timing == InjectionTiming.IMMEDIATE:
                # Conservative strategy normally doesn't do immediate injections
                # But make exception for server VAD with auto-response
                if not (vad_mode == 'server' and auto_response):
                    continue
                
            if context.priority_value > best_priority:
                best_context = context
                best_priority = context.priority_value
        
        if not best_context:
            return InjectionDecision(
                should_inject=False,
                context_to_inject=None,
                priority=0.0,
                reason="no_suitable_context",
                timing="delayed"
            )
        
        # Decide based on context priority
        if best_priority >= ContextPriority.HIGH.value:
            # High priority contexts can inject with lower confidence
            threshold = min(0.7, effective_threshold * 0.9)
        else:
            threshold = effective_threshold
        
        if best_detection.confidence >= threshold:
            # Determine timing based on VAD mode
            timing = "immediate" if injection_mode == "immediate" else "next_pause"
            
            return InjectionDecision(
                should_inject=True,
                context_to_inject=best_context,
                priority=best_priority / 10.0,  # Normalize to 0-1
                reason=f"high_confidence_{best_detection.__class__.__name__}",
                timing=timing
            )
        
        return InjectionDecision(
            should_inject=False,
            context_to_inject=None,
            priority=0.0,
            reason="confidence_below_threshold",
            timing="delayed"
        )