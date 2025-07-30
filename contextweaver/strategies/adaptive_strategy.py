"""
Adaptive Strategy for Context Injection

Dynamically adjusts injection behavior based on conversation.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from .base import InjectionStrategy, InjectionDecision
from contextweaver.schema import ContextToInject, InjectionTiming, ContextPriority


class AdaptiveStrategy(InjectionStrategy):
    """
    Adaptive injection strategy.
    
    - Learns from conversation patterns
    - Adjusts thresholds dynamically
    - Balances flow and context richness
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.7,
        learning_rate: float = 0.1
    ):
        super().__init__(name="adaptive")
        self.base_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.learning_rate = learning_rate
        
        # Adaptive metrics
        self.successful_injections = 0
        self.failed_injections = 0
        self.conversation_metrics = {
            "interruptions": 0,
            "smooth_turns": 0,
            "avg_turn_duration": 0.0
        }
    
    async def decide(
        self,
        detections: List['DetectionResult'],
        state: Any,
        available_context: Dict[str, ContextToInject]
    ) -> InjectionDecision:
        """
        Adaptive decision making.
        
        Adjusts strategy based on:
        1. Conversation flow metrics
        2. Previous injection success
        3. User engagement signals
        4. VAD mode and response settings
        """
        
        # Get VAD mode info
        vad_mode = getattr(state, 'vad_mode', None)
        auto_response = getattr(state, 'auto_response', True)
        injection_mode = getattr(state, 'injection_mode', 'adaptive')
        
        # Update conversation metrics
        self._update_metrics(state)
        
        # Adapt threshold based on VAD mode
        mode_adjustment = 0.0
        if vad_mode == 'server' and auto_response:
            # Need to be more aggressive in server+auto mode
            mode_adjustment = -0.15
        elif vad_mode == 'client' and not auto_response:
            # Can be more conservative with full control
            mode_adjustment = 0.1
        
        # Adjust threshold based on conversation flow
        flow_adjustment = 0.0
        if self.conversation_metrics["interruptions"] > self.conversation_metrics["smooth_turns"]:
            # Too aggressive, increase threshold
            flow_adjustment = self.learning_rate
        elif self.get_injection_rate(60) < 0.2:
            # Too conservative, decrease threshold
            flow_adjustment = -self.learning_rate
        
        self.current_threshold = max(0.4, min(0.9, 
            self.base_threshold + mode_adjustment + flow_adjustment))
        
        # Find best detection
        best_detection = None
        for detection in detections:
            if detection.detected and detection.confidence >= self.current_threshold:
                if not best_detection or detection.confidence > best_detection.confidence:
                    best_detection = detection
        
        # Get injectable contexts
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
        
        # Adaptive scoring based on conversation state
        def adaptive_score(ctx):
            base_score = ctx.priority_value
            
            # VAD mode adjustments
            if vad_mode == 'server' and auto_response:
                # Boost immediate and high priority in server+auto
                if ctx.timing == InjectionTiming.IMMEDIATE:
                    base_score += 4
                if ctx.priority_value >= ContextPriority.HIGH.value:
                    base_score += 2
            
            # Boost score if conversation is flowing well
            if self.conversation_metrics["smooth_turns"] > 5:
                if ctx.timing in [InjectionTiming.NEXT_PAUSE, InjectionTiming.LAZY]:
                    base_score += 2
            
            # Boost immediate contexts if user seems stuck
            if hasattr(state, 'audio') and hasattr(state.audio, 'silence_duration_ms'):
                silence_threshold = 2000 if vad_mode == 'server' else 3000
                if state.audio.silence_duration_ms > silence_threshold:
                    if ctx.timing == InjectionTiming.IMMEDIATE:
                        base_score += 3
            
            # Penalize frequent injection types
            if ctx.timing == InjectionTiming.NEXT_TURN and self.get_injection_rate(30) > 1.0:
                base_score -= 2
            
            return base_score
        
        injectable_contexts.sort(key=adaptive_score, reverse=True)
        best_context = injectable_contexts[0]
        
        # Adaptive decision based on multiple factors
        confidence_factor = best_detection.confidence if best_detection else 0.5
        priority_factor = min(1.0, best_context.priority_value / 10.0)
        flow_factor = self.conversation_metrics["smooth_turns"] / (self.conversation_metrics["interruptions"] + 1)
        
        combined_score = (confidence_factor + priority_factor + flow_factor) / 3
        
        # Adjust threshold based on VAD mode
        score_threshold = 0.5 if (vad_mode == 'server' and auto_response) else 0.6
        
        if combined_score >= score_threshold:
            # Determine timing based on conversation state and VAD mode
            if injection_mode == "immediate":
                timing = "immediate"
            elif best_context.timing == InjectionTiming.IMMEDIATE and flow_factor < 0.5:
                # Poor flow, delay immediate injections (unless forced by VAD mode)
                timing = "next_turn"
            else:
                timing = "immediate"
            
            return InjectionDecision(
                should_inject=True,
                context_to_inject=best_context,
                priority=combined_score,
                reason=f"adaptive_score_{combined_score:.2f}",
                timing=timing
            )
        
        return InjectionDecision(
            should_inject=False,
            context_to_inject=None,
            priority=0.0,
            reason=f"below_adaptive_threshold_{combined_score:.2f}",
            timing="delayed"
        )
    
    def _update_metrics(self, state):
        """Update conversation flow metrics"""
        if hasattr(state, 'metrics'):
            self.conversation_metrics["interruptions"] = getattr(
                state.metrics, 'interruption_count', 0
            )
        
        if hasattr(state, 'turns') and state.turns:
            self.conversation_metrics["smooth_turns"] = len([
                t for t in state.turns 
                if hasattr(t, 'is_complete') and t.is_complete
            ])
    
    def record_injection_result(self, success: bool):
        """Record injection outcome for learning"""
        if success:
            self.successful_injections += 1
        else:
            self.failed_injections += 1
        
        # Adjust threshold based on success rate
        if (self.successful_injections + self.failed_injections) % 10 == 0:
            success_rate = self.successful_injections / (self.successful_injections + self.failed_injections)
            if success_rate < 0.7:
                # Too many failures, be more conservative
                self.current_threshold = min(0.9, self.current_threshold + self.learning_rate / 2)
            elif success_rate > 0.9:
                # Very successful, can be more aggressive
                self.current_threshold = max(0.5, self.current_threshold - self.learning_rate / 2)