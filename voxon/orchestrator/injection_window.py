"""
Injection Window Manager

Calculates and manages optimal windows for context injection based on
conversation state, VAD mode, and response timing.
"""

import time
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import logging


class WindowType(str, Enum):
    """Types of injection windows"""
    SILENCE = "silence"  # During silence/pause
    PRE_RESPONSE = "pre_response"  # Before AI response
    POST_INPUT = "post_input"  # After user input
    TURN_TRANSITION = "turn_transition"  # Between turns
    IMMEDIATE = "immediate"  # ASAP


@dataclass
class InjectionWindow:
    """Represents an injection opportunity window"""
    window_type: WindowType
    start_time: float
    duration_ms: float
    confidence: float = 1.0  # How confident we are about this window
    priority: int = 5  # Priority of using this window
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def end_time(self) -> float:
        """Calculate window end time"""
        return self.start_time + (self.duration_ms / 1000.0)
    
    @property
    def is_active(self) -> bool:
        """Check if window is currently active"""
        current_time = time.time()
        return self.start_time <= current_time <= self.end_time
    
    @property
    def time_remaining(self) -> float:
        """Time remaining in window (seconds)"""
        return max(0, self.end_time - time.time())
    
    @property
    def time_elapsed(self) -> float:
        """Time elapsed since window start (seconds)"""
        return max(0, time.time() - self.start_time)
    
    def can_inject(self, required_time_ms: float) -> bool:
        """Check if there's enough time for injection"""
        return self.time_remaining * 1000 >= required_time_ms


class InjectionWindowManager:
    """
    Manages injection windows and opportunities.
    
    Tracks conversation flow and identifies optimal moments
    for context injection without disrupting the user experience.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Active windows
        self.active_windows: List[InjectionWindow] = []
        self.window_history: List[InjectionWindow] = []
        self.max_history = 100
        
        # Timing configuration (ms)
        self.min_silence_duration = 300  # Min silence to create window
        self.pre_response_buffer = 100  # Time before response
        self.post_input_delay = 200  # Delay after user input
        self.default_window_duration = 500  # Default window size
        
        # VAD-specific timing
        self.vad_timings = {
            ("server", True): {  # Server VAD + auto response
                "pre_response_buffer": 50,
                "default_window": 100,
                "max_window": 200
            },
            ("server", False): {  # Server VAD + manual response
                "pre_response_buffer": 200,
                "default_window": 500,
                "max_window": 1000
            },
            ("client", True): {  # Client VAD + auto response
                "pre_response_buffer": 100,
                "default_window": 300,
                "max_window": 500
            },
            ("client", False): {  # Client VAD + manual response
                "pre_response_buffer": 300,
                "default_window": 1000,
                "max_window": 2000
            }
        }
        
        # State tracking
        self.last_user_input_time = 0
        self.last_ai_response_time = 0
        self.last_silence_start = 0
        self.is_user_speaking = False
        self.is_ai_responding = False
        self.vad_mode = "client"
        self.auto_response = True
        
        # Metrics
        self.metrics = {
            "windows_created": 0,
            "windows_used": 0,
            "windows_missed": 0,
            "avg_window_duration_ms": 0,
            "optimal_injections": 0
        }
    
    def update_vad_mode(self, vad_mode: str, auto_response: bool):
        """Update VAD configuration"""
        self.vad_mode = vad_mode
        self.auto_response = auto_response
        
        # Update timings based on VAD mode
        timings = self.vad_timings.get((vad_mode, auto_response), {})
        if timings:
            self.pre_response_buffer = timings.get("pre_response_buffer", self.pre_response_buffer)
            self.default_window_duration = timings.get("default_window", self.default_window_duration)
        
        self.logger.info(f"Window manager updated: VAD={vad_mode}, auto_response={auto_response}")
    
    def on_user_input_start(self):
        """Handle user input start"""
        self.is_user_speaking = True
        self._close_active_windows()  # Close windows when user speaks
    
    def on_user_input_end(self):
        """Handle user input end"""
        self.is_user_speaking = False
        self.last_user_input_time = time.time()
        
        # Create post-input window
        window = InjectionWindow(
            window_type=WindowType.POST_INPUT,
            start_time=time.time() + (self.post_input_delay / 1000.0),
            duration_ms=self._get_window_duration(WindowType.POST_INPUT),
            confidence=0.8,
            priority=7
        )
        self._add_window(window)
    
    def on_ai_response_pending(self, estimated_delay_ms: Optional[float] = None):
        """Handle pending AI response"""
        # Create pre-response window if we have time
        if estimated_delay_ms and estimated_delay_ms > self.pre_response_buffer:
            window = InjectionWindow(
                window_type=WindowType.PRE_RESPONSE,
                start_time=time.time(),
                duration_ms=min(estimated_delay_ms - self.pre_response_buffer, 
                               self._get_max_window_duration()),
                confidence=0.9,
                priority=9
            )
            self._add_window(window)
    
    def on_ai_response_start(self):
        """Handle AI response start"""
        self.is_ai_responding = True
        self.last_ai_response_time = time.time()
        self._close_active_windows()  # Close windows during AI response
    
    def on_ai_response_end(self):
        """Handle AI response end"""
        self.is_ai_responding = False
        
        # Create turn transition window
        window = InjectionWindow(
            window_type=WindowType.TURN_TRANSITION,
            start_time=time.time() + 0.1,  # Small delay
            duration_ms=self._get_window_duration(WindowType.TURN_TRANSITION),
            confidence=0.7,
            priority=6
        )
        self._add_window(window)
    
    def on_silence_detected(self, duration_ms: float):
        """Handle silence detection"""
        if duration_ms >= self.min_silence_duration and not self.is_user_speaking and not self.is_ai_responding:
            # Create silence window
            window = InjectionWindow(
                window_type=WindowType.SILENCE,
                start_time=time.time(),
                duration_ms=self._get_window_duration(WindowType.SILENCE),
                confidence=0.6,
                priority=5,
                metadata={"silence_duration": duration_ms}
            )
            self._add_window(window)
    
    def get_best_window(self, required_time_ms: float = 100) -> Optional[InjectionWindow]:
        """
        Get the best available injection window.
        
        Args:
            required_time_ms: Minimum time needed for injection
            
        Returns:
            Best available window or None
        """
        # Clean up expired windows
        self._cleanup_windows()
        
        # Find best active window
        valid_windows = [
            w for w in self.active_windows 
            if w.is_active and w.can_inject(required_time_ms)
        ]
        
        if not valid_windows:
            return None
        
        # Sort by priority and confidence
        valid_windows.sort(key=lambda w: (w.priority, w.confidence), reverse=True)
        return valid_windows[0]
    
    def create_immediate_window(self, duration_ms: Optional[float] = None) -> InjectionWindow:
        """Create an immediate injection window"""
        window = InjectionWindow(
            window_type=WindowType.IMMEDIATE,
            start_time=time.time(),
            duration_ms=duration_ms or self._get_window_duration(WindowType.IMMEDIATE),
            confidence=1.0,
            priority=10
        )
        self._add_window(window)
        return window
    
    def mark_window_used(self, window: InjectionWindow):
        """Mark a window as used"""
        self.metrics["windows_used"] += 1
        
        # Check if it was optimal timing
        if window.time_elapsed < window.duration_ms * 0.5 / 1000.0:
            self.metrics["optimal_injections"] += 1
        
        # Remove from active windows
        if window in self.active_windows:
            self.active_windows.remove(window)
    
    def get_injection_recommendation(self) -> Dict[str, Any]:
        """Get recommendation for injection timing"""
        best_window = self.get_best_window()
        
        if best_window:
            urgency = self._calculate_urgency(best_window)
            return {
                "recommended": True,
                "window_type": best_window.window_type.value,
                "time_available_ms": best_window.time_remaining * 1000,
                "confidence": best_window.confidence,
                "urgency": urgency,
                "reason": self._get_recommendation_reason(best_window)
            }
        else:
            return {
                "recommended": False,
                "reason": "No suitable injection window available",
                "next_opportunity_ms": self._estimate_next_window()
            }
    
    def _add_window(self, window: InjectionWindow):
        """Add a new injection window"""
        self.active_windows.append(window)
        self.window_history.append(window)
        
        # Maintain history size
        if len(self.window_history) > self.max_history:
            self.window_history.pop(0)
        
        self.metrics["windows_created"] += 1
        
        # Update average duration
        total = self.metrics["windows_created"]
        old_avg = self.metrics["avg_window_duration_ms"]
        self.metrics["avg_window_duration_ms"] = (
            (old_avg * (total - 1) + window.duration_ms) / total
        )
        
        self.logger.debug(f"Created {window.window_type.value} window: {window.duration_ms}ms")
    
    def _close_active_windows(self):
        """Close all active windows"""
        for window in self.active_windows:
            if window.is_active:
                self.metrics["windows_missed"] += 1
        self.active_windows.clear()
    
    def _cleanup_windows(self):
        """Remove expired windows"""
        self.active_windows = [w for w in self.active_windows if w.is_active]
    
    def _get_window_duration(self, window_type: WindowType) -> float:
        """Get appropriate duration for window type"""
        timings = self.vad_timings.get((self.vad_mode, self.auto_response), {})
        
        durations = {
            WindowType.SILENCE: self.default_window_duration * 1.5,
            WindowType.PRE_RESPONSE: timings.get("default_window", self.default_window_duration),
            WindowType.POST_INPUT: self.default_window_duration,
            WindowType.TURN_TRANSITION: self.default_window_duration * 1.2,
            WindowType.IMMEDIATE: timings.get("default_window", self.default_window_duration) * 0.5
        }
        
        return durations.get(window_type, self.default_window_duration)
    
    def _get_max_window_duration(self) -> float:
        """Get maximum window duration for current mode"""
        timings = self.vad_timings.get((self.vad_mode, self.auto_response), {})
        return timings.get("max_window", 1000)
    
    def _calculate_urgency(self, window: InjectionWindow) -> str:
        """Calculate injection urgency"""
        time_remaining_ratio = window.time_remaining / (window.duration_ms / 1000.0)
        
        if time_remaining_ratio < 0.2:
            return "critical"
        elif time_remaining_ratio < 0.5:
            return "high"
        elif time_remaining_ratio < 0.8:
            return "medium"
        else:
            return "low"
    
    def _get_recommendation_reason(self, window: InjectionWindow) -> str:
        """Get human-readable recommendation reason"""
        reasons = {
            WindowType.SILENCE: "Natural pause in conversation",
            WindowType.PRE_RESPONSE: "Before AI response begins",
            WindowType.POST_INPUT: "After user input processed",
            WindowType.TURN_TRANSITION: "Between conversation turns",
            WindowType.IMMEDIATE: "Immediate injection required"
        }
        return reasons.get(window.window_type, "Injection opportunity detected")
    
    def _estimate_next_window(self) -> float:
        """Estimate time until next window (ms)"""
        # Simple estimation based on conversation patterns
        time_since_input = (time.time() - self.last_user_input_time) * 1000
        time_since_response = (time.time() - self.last_ai_response_time) * 1000
        
        if time_since_input < 1000:
            return self.post_input_delay
        elif time_since_response < 2000:
            return 500  # Typical turn transition time
        else:
            return 1000  # Default estimate
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get window manager metrics"""
        window_success_rate = (
            self.metrics["windows_used"] / max(1, self.metrics["windows_created"])
        )
        
        return {
            **self.metrics,
            "active_windows": len(self.active_windows),
            "window_success_rate": window_success_rate,
            "vad_mode": self.vad_mode,
            "auto_response": self.auto_response
        }