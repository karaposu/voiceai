"""
Aggressive Strategy for Context Injection

Actively injects context to enhance conversation.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from .base import InjectionStrategy, InjectionDecision
from contextweaver.schema import ContextToInject, InjectionTiming, ContextPriority


class AggressiveStrategy(InjectionStrategy):
    """
    Aggressive injection strategy.
    
    - Injects frequently to maximize context
    - Lower detection thresholds
    - Prioritizes completeness over flow
    """
    
    def __init__(
        self,
        min_interval_seconds: int = 5,
        confidence_threshold: float = 0.5
    ):
        super().__init__(name="aggressive")
        self.min_interval_seconds = min_interval_seconds
        self.confidence_threshold = confidence_threshold
    
    async def decide(
        self,
        detections: List['DetectionResult'],
        state: Any,
        available_context: Dict[str, ContextToInject]
    ) -> InjectionDecision:
        """
        Aggressive decision making.
        
        Injects when:
        1. Any reasonable detection
        2. Any pending context
        3. Minimal time spacing
        
        Becomes even more aggressive with server VAD + auto-response.
        """
        
        # Adapt based on VAD mode
        vad_mode = getattr(state, 'vad_mode', None)
        auto_response = getattr(state, 'auto_response', True)
        injection_mode = getattr(state, 'injection_mode', 'adaptive')
        
        # Very aggressive for server VAD with auto-response
        if vad_mode == 'server' and auto_response:
            effective_threshold = self.confidence_threshold * 0.6  # Much lower
            effective_interval = max(1, self.min_interval_seconds // 5)
        else:
            effective_threshold = self.confidence_threshold
            effective_interval = self.min_interval_seconds
        
        # Quick cooldown check
        if self.last_injection:
            time_since = datetime.now() - self.last_injection
            if time_since < timedelta(seconds=effective_interval):
                # Still check for critical contexts
                for context in available_context.values():
                    if context.priority_value >= ContextPriority.CRITICAL.value:
                        return InjectionDecision(
                            should_inject=True,
                            context_to_inject=context,
                            priority=1.0,
                            reason="critical_priority_override",
                            timing="immediate"
                        )
                
                return InjectionDecision(
                    should_inject=False,
                    context_to_inject=None,
                    priority=0.0,
                    reason="cooldown_period",
                    timing="delayed"
                )
        
        # Find any detection above threshold
        valid_detection = None
        for detection in detections:
            if detection.detected and detection.confidence >= effective_threshold:
                valid_detection = detection
                break
        
        # Find any injectable context
        injectable_contexts = []
        for context_id, context in available_context.items():
            if context.should_inject(state.__dict__ if hasattr(state, '__dict__') else state):
                injectable_contexts.append(context)
        
        if not injectable_contexts:
            return InjectionDecision(
                should_inject=False,
                context_to_inject=None,
                priority=0.0,
                reason="no_injectable_context",
                timing="delayed"
            )
        
        # Sort by priority and timing urgency
        def context_score(ctx):
            timing_scores = {
                InjectionTiming.IMMEDIATE: 10,
                InjectionTiming.NEXT_TURN: 8,
                InjectionTiming.NEXT_PAUSE: 6,
                InjectionTiming.ON_TOPIC: 5,
                InjectionTiming.ON_TRIGGER: 4,
                InjectionTiming.LAZY: 2,
                InjectionTiming.SCHEDULED: 3,
                InjectionTiming.MANUAL: 0
            }
            return ctx.priority_value + timing_scores.get(ctx.timing, 0)
        
        injectable_contexts.sort(key=context_score, reverse=True)
        best_context = injectable_contexts[0]
        
        # Aggressive: inject if we have any detection or high priority
        if valid_detection or best_context.priority_value >= ContextPriority.MEDIUM.value:
            return InjectionDecision(
                should_inject=True,
                context_to_inject=best_context,
                priority=min(1.0, best_context.priority_value / 10.0),
                reason=f"aggressive_inject_{valid_detection.__class__.__name__ if valid_detection else 'priority'}",
                timing="immediate"
            )
        
        # Even without detection, inject low priority after some time
        # Extra aggressive in server+auto mode
        target_rate = 0.8 if (vad_mode == 'server' and auto_response) else 0.5
        if self.get_injection_rate(60) < target_rate:
            return InjectionDecision(
                should_inject=True,
                context_to_inject=best_context,
                priority=0.3,
                reason="maintaining_injection_rate",
                timing="immediate" if injection_mode == "immediate" else "next_turn"
            )
        
        return InjectionDecision(
            should_inject=False,
            context_to_inject=None,
            priority=0.0,
            reason="no_trigger_conditions",
            timing="delayed"
        )