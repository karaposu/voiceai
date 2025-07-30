"""
Response Timing Detector

Predicts when AI responses will trigger and identifies optimal injection points.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import statistics
from .base import BaseDetector, DetectionResult


class ResponseTimingDetector(BaseDetector):
    """
    Detects and predicts response timing for optimal context injection.
    
    Features:
    - Predicts when AI will start responding
    - Identifies pre-response injection windows
    - Learns from conversation patterns
    - Adapts to different response trigger modes
    """
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        history_size: int = 20,
        prediction_window_ms: int = 500
    ):
        super().__init__(threshold=min_confidence)
        
        # Response timing history
        self.response_times: deque = deque(maxlen=history_size)
        self.turn_durations: deque = deque(maxlen=history_size)
        
        # Current state tracking
        self.last_user_input_end: Optional[datetime] = None
        self.last_response_start: Optional[datetime] = None
        self.current_turn_start: Optional[datetime] = None
        self.response_pending = False
        
        # Prediction parameters
        self.prediction_window_ms = prediction_window_ms
        self.predicted_response_time: Optional[datetime] = None
        
        # Response mode awareness
        self.response_mode: Optional[str] = None  # 'manual', 'automatic', 'hybrid'
        self.vad_mode: Optional[str] = None
        self.auto_response_enabled = True
        
        # Pattern recognition
        self.response_patterns: Dict[str, List[float]] = {
            'user_to_response': [],  # Time from user input end to response start
            'silence_to_response': [],  # Silence duration before response
            'turn_duration': []  # Total turn duration
        }
    
    async def detect(self, state) -> DetectionResult:
        """Detect response timing opportunities"""
        
        # Update configuration from state
        self._update_configuration(state)
        
        # Track conversation flow
        now = datetime.now()
        self._track_conversation_flow(state, now)
        
        # Check if we're in a pre-response window
        if self.response_pending and self.predicted_response_time:
            time_to_response = (self.predicted_response_time - now).total_seconds() * 1000
            
            if 0 < time_to_response <= self.prediction_window_ms:
                # We're in the injection window!
                confidence = self._calculate_prediction_confidence(time_to_response)
                
                return DetectionResult(
                    detected=True,
                    confidence=confidence,
                    timestamp=now,
                    metadata={
                        "type": "pre_response_window",
                        "time_to_response_ms": time_to_response,
                        "window_size_ms": self.prediction_window_ms,
                        "response_mode": self.response_mode,
                        "prediction_method": self._get_prediction_method()
                    }
                )
        
        # Check for other timing opportunities
        opportunity = self._detect_timing_opportunity(state, now)
        if opportunity:
            return opportunity
        
        # No immediate opportunity, but provide prediction info
        return DetectionResult(
            detected=False,
            confidence=0.0,
            timestamp=now,
            metadata={
                "response_pending": self.response_pending,
                "predicted_response_time": self.predicted_response_time.isoformat() if self.predicted_response_time else None,
                "time_since_input_end": self._get_time_since_input_end(),
                "average_response_time": self._get_average_response_time()
            }
        )
    
    def _update_configuration(self, state):
        """Update detector configuration from state"""
        if hasattr(state, 'vad_mode'):
            self.vad_mode = state.vad_mode
        
        if hasattr(state, 'auto_response'):
            self.auto_response_enabled = state.auto_response
        
        if hasattr(state, 'injection_mode'):
            # Map injection mode to response mode
            if state.injection_mode == "immediate":
                self.response_mode = "automatic"
            elif state.injection_mode == "controlled":
                self.response_mode = "manual"
            else:
                self.response_mode = "hybrid"
    
    def _track_conversation_flow(self, state, now: datetime):
        """Track conversation events for pattern learning"""
        
        # Check for user input end
        if hasattr(state, 'audio') and hasattr(state.audio, 'is_listening'):
            if not state.audio.is_listening and self.last_user_input_end is None:
                self.last_user_input_end = now
                self.response_pending = True
                
                # Predict response time
                self.predicted_response_time = self._predict_response_time(now)
        
        # Check for response start
        if hasattr(state, 'status'):
            if state.status == 'responding' and not self.last_response_start:
                self.last_response_start = now
                self.response_pending = False
                
                # Record timing pattern
                if self.last_user_input_end:
                    delay = (now - self.last_user_input_end).total_seconds() * 1000
                    self.response_patterns['user_to_response'].append(delay)
                    self.response_times.append({
                        'timestamp': now,
                        'delay_ms': delay,
                        'mode': self.response_mode
                    })
        
        # Check for turn completion
        if hasattr(state, 'current_turn'):
            if state.current_turn is None and self.current_turn_start:
                # Turn completed
                duration = (now - self.current_turn_start).total_seconds() * 1000
                self.turn_durations.append(duration)
                self.response_patterns['turn_duration'].append(duration)
                
                # Reset for next turn
                self.current_turn_start = None
                self.last_user_input_end = None
                self.last_response_start = None
            elif state.current_turn and not self.current_turn_start:
                # New turn started
                self.current_turn_start = now
    
    def _predict_response_time(self, input_end: datetime) -> Optional[datetime]:
        """Predict when response will start"""
        
        # Use different prediction strategies based on mode
        if self.response_mode == "automatic":
            return self._predict_automatic_response(input_end)
        elif self.response_mode == "manual":
            return self._predict_manual_response(input_end)
        else:
            return self._predict_hybrid_response(input_end)
    
    def _predict_automatic_response(self, input_end: datetime) -> Optional[datetime]:
        """Predict automatic response timing"""
        # Server VAD typically responds very quickly
        if self.vad_mode == "server":
            # Use historical data or default to 200ms
            avg_delay = self._get_average_delay_for_mode("automatic")
            delay_ms = avg_delay if avg_delay > 0 else 200
        else:
            # Client VAD with auto-response
            avg_delay = self._get_average_delay_for_mode("automatic")
            delay_ms = avg_delay if avg_delay > 0 else 500
        
        return input_end + timedelta(milliseconds=delay_ms)
    
    def _predict_manual_response(self, input_end: datetime) -> Optional[datetime]:
        """Predict manual response timing"""
        # Manual mode depends on context injection timing
        # Use historical patterns
        if self.response_patterns['user_to_response']:
            recent_delays = self.response_patterns['user_to_response'][-5:]
            avg_delay = statistics.mean(recent_delays)
            
            # Add some buffer for manual triggering
            delay_ms = avg_delay * 1.2
        else:
            # Default estimate for manual mode
            delay_ms = 1500
        
        return input_end + timedelta(milliseconds=delay_ms)
    
    def _predict_hybrid_response(self, input_end: datetime) -> Optional[datetime]:
        """Predict hybrid response timing"""
        # Hybrid mode - blend automatic and manual predictions
        auto_prediction = self._predict_automatic_response(input_end)
        manual_prediction = self._predict_manual_response(input_end)
        
        if auto_prediction and manual_prediction:
            # Take weighted average
            auto_ms = (auto_prediction - input_end).total_seconds() * 1000
            manual_ms = (manual_prediction - input_end).total_seconds() * 1000
            
            # Weight towards automatic if recent responses were fast
            weight = 0.7 if self._recent_responses_fast() else 0.3
            hybrid_ms = auto_ms * weight + manual_ms * (1 - weight)
            
            return input_end + timedelta(milliseconds=hybrid_ms)
        
        return auto_prediction or manual_prediction
    
    def _get_average_delay_for_mode(self, mode: str) -> float:
        """Get average response delay for specific mode"""
        mode_delays = [
            r['delay_ms'] for r in self.response_times
            if r.get('mode') == mode
        ]
        
        if mode_delays:
            return statistics.mean(mode_delays[-5:])  # Use recent samples
        return 0
    
    def _recent_responses_fast(self) -> bool:
        """Check if recent responses were fast"""
        if len(self.response_times) < 3:
            return False
        
        recent = list(self.response_times)[-3:]
        avg_recent = statistics.mean(r['delay_ms'] for r in recent)
        
        return avg_recent < 500  # Fast if under 500ms
    
    def _calculate_prediction_confidence(self, time_to_response: float) -> float:
        """Calculate confidence based on prediction accuracy"""
        # Base confidence on how close we are to predicted time
        time_confidence = 1.0 - (time_to_response / self.prediction_window_ms)
        
        # Historical accuracy bonus
        accuracy_bonus = self._get_historical_accuracy()
        
        # Mode confidence
        mode_confidence = {
            'automatic': 0.9,  # High confidence for automatic
            'manual': 0.7,     # Lower for manual
            'hybrid': 0.8      # Medium for hybrid
        }.get(self.response_mode, 0.8)
        
        # Combine factors
        confidence = time_confidence * accuracy_bonus * mode_confidence
        
        return min(1.0, max(0.0, confidence))
    
    def _get_historical_accuracy(self) -> float:
        """Calculate historical prediction accuracy"""
        # This would track actual vs predicted times
        # For now, use pattern consistency as proxy
        if len(self.response_patterns['user_to_response']) >= 5:
            delays = self.response_patterns['user_to_response'][-10:]
            if delays:
                std_dev = statistics.stdev(delays) if len(delays) > 1 else 0
                mean_delay = statistics.mean(delays)
                
                # Lower variance = higher accuracy
                if mean_delay > 0:
                    consistency = 1.0 - min(1.0, std_dev / mean_delay)
                    return 0.7 + (0.3 * consistency)
        
        return 0.8  # Default accuracy
    
    def _detect_timing_opportunity(self, state, now: datetime) -> Optional[DetectionResult]:
        """Detect other timing opportunities"""
        
        # Check for silence-based opportunity
        if hasattr(state, 'audio') and hasattr(state.audio, 'vad_active'):
            if not state.audio.vad_active and self.last_user_input_end:
                silence_duration = (now - self.last_user_input_end).total_seconds() * 1000
                
                # Check if this matches typical pre-response silence
                if self._is_pre_response_silence(silence_duration):
                    return DetectionResult(
                        detected=True,
                        confidence=0.7,
                        timestamp=now,
                        metadata={
                            "type": "pre_response_silence",
                            "silence_duration_ms": silence_duration,
                            "typical_silence_ms": self._get_typical_pre_response_silence()
                        }
                    )
        
        return None
    
    def _is_pre_response_silence(self, silence_ms: float) -> bool:
        """Check if silence duration matches pre-response pattern"""
        typical = self._get_typical_pre_response_silence()
        if typical > 0:
            # Within 20% of typical
            return 0.8 * typical <= silence_ms <= 1.2 * typical
        return False
    
    def _get_typical_pre_response_silence(self) -> float:
        """Get typical silence duration before responses"""
        if self.response_patterns['silence_to_response']:
            return statistics.median(self.response_patterns['silence_to_response'][-10:])
        
        # Default based on mode
        if self.response_mode == "automatic":
            return 300
        elif self.response_mode == "manual":
            return 1000
        return 500
    
    def _get_prediction_method(self) -> str:
        """Get the prediction method used"""
        if len(self.response_patterns['user_to_response']) >= 5:
            return "historical_patterns"
        elif self.response_mode:
            return f"mode_based_{self.response_mode}"
        return "default_estimates"
    
    def _get_time_since_input_end(self) -> Optional[float]:
        """Get time since user input ended (ms)"""
        if self.last_user_input_end:
            return (datetime.now() - self.last_user_input_end).total_seconds() * 1000
        return None
    
    def _get_average_response_time(self) -> float:
        """Get average response time"""
        if self.response_patterns['user_to_response']:
            return statistics.mean(self.response_patterns['user_to_response'][-10:])
        return 0
    
    def update_response_mode(self, mode: str, vad_mode: str, auto_response: bool):
        """Update detector for mode changes"""
        self.response_mode = mode
        self.vad_mode = vad_mode
        self.auto_response_enabled = auto_response
        
        # Adjust prediction window based on mode
        if mode == "automatic" and vad_mode == "server":
            self.prediction_window_ms = 200  # Very tight window
        elif mode == "manual":
            self.prediction_window_ms = 1000  # Relaxed window
        else:
            self.prediction_window_ms = 500  # Default
    
    def record_response_event(self, event_type: str, timestamp: datetime, metadata: Dict[str, Any]):
        """Record response events for learning"""
        if event_type == "response_triggered":
            # Update patterns based on actual trigger time
            if self.last_user_input_end:
                actual_delay = (timestamp - self.last_user_input_end).total_seconds() * 1000
                
                # Compare with prediction if available
                if self.predicted_response_time:
                    predicted_delay = (self.predicted_response_time - self.last_user_input_end).total_seconds() * 1000
                    error = abs(actual_delay - predicted_delay)
                    
                    # Store for accuracy tracking
                    if not hasattr(self, 'prediction_errors'):
                        self.prediction_errors = deque(maxlen=20)
                    self.prediction_errors.append(error)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        stats = {
            "response_mode": self.response_mode,
            "vad_mode": self.vad_mode,
            "prediction_window_ms": self.prediction_window_ms,
            "average_response_delay": self._get_average_response_time(),
            "response_count": len(self.response_times),
            "pattern_counts": {k: len(v) for k, v in self.response_patterns.items()}
        }
        
        # Add prediction accuracy if available
        if hasattr(self, 'prediction_errors') and self.prediction_errors:
            stats["average_prediction_error_ms"] = statistics.mean(self.prediction_errors)
            stats["prediction_accuracy"] = self._get_historical_accuracy()
        
        return stats