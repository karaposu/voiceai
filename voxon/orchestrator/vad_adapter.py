"""
VAD Mode Adapter

Provides a unified interface for different VAD modes and response control.
"""

from typing import Dict, Any, Optional, Literal
from datetime import datetime, timedelta
import logging


class VADModeAdapter:
    """
    Adapts behavior based on VAD mode and response settings.
    
    Provides recommendations for:
    - Injection timing
    - Response control
    - Detection thresholds
    - Monitoring frequency
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Current mode settings
        self.vad_mode: Literal["client", "server"] = "client"
        self.auto_response: bool = True
        self.server_silence_ms: int = 500  # Server VAD silence duration
        
        # Injection windows (in milliseconds)
        self.injection_windows = {
            ("server", True): 200,    # Server VAD + auto: very short window
            ("server", False): 1000,  # Server VAD + manual: comfortable window
            ("client", True): 500,    # Client VAD + auto: medium window
            ("client", False): 2000   # Client VAD + manual: long window
        }
        
        # Detection sensitivities
        self.detection_configs = {
            ("server", True): {
                "silence_threshold_ms": 300,
                "confidence_boost": 0.2,
                "urgency_multiplier": 2.0
            },
            ("server", False): {
                "silence_threshold_ms": 500,
                "confidence_boost": 0.1,
                "urgency_multiplier": 1.5
            },
            ("client", True): {
                "silence_threshold_ms": 400,
                "confidence_boost": 0.15,
                "urgency_multiplier": 1.5
            },
            ("client", False): {
                "silence_threshold_ms": 800,
                "confidence_boost": 0.0,
                "urgency_multiplier": 1.0
            }
        }
    
    def update_mode(self, vad_mode: str, auto_response: bool, server_silence_ms: Optional[int] = None):
        """Update VAD mode and response settings"""
        self.vad_mode = vad_mode
        self.auto_response = auto_response
        if server_silence_ms is not None:
            self.server_silence_ms = server_silence_ms
        
        self.logger.info(f"VAD adapter updated: {vad_mode} VAD, auto-response={auto_response}")
    
    def get_injection_window(self) -> int:
        """Get available injection window in milliseconds"""
        key = (self.vad_mode, self.auto_response)
        window = self.injection_windows.get(key, 1000)
        
        # Adjust based on server silence duration
        if self.vad_mode == "server" and self.auto_response:
            # Must inject before server detects silence end
            window = min(window, self.server_silence_ms * 0.4)
        
        return int(window)
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration for current mode"""
        key = (self.vad_mode, self.auto_response)
        return self.detection_configs.get(key, self.detection_configs[("client", False)])
    
    def adjust_confidence(self, base_confidence: float) -> float:
        """Adjust detection confidence based on mode"""
        config = self.get_detection_config()
        boost = config.get("confidence_boost", 0.0)
        return min(1.0, base_confidence + boost)
    
    def calculate_urgency(self, priority: float, timing: str) -> float:
        """Calculate injection urgency based on mode and context"""
        config = self.get_detection_config()
        multiplier = config.get("urgency_multiplier", 1.0)
        
        # Base urgency from priority and timing
        timing_scores = {
            "immediate": 1.0,
            "next_turn": 0.8,
            "next_pause": 0.6,
            "on_trigger": 0.4,
            "lazy": 0.2
        }
        
        base_urgency = priority * timing_scores.get(timing, 0.5)
        
        # Apply mode multiplier
        return min(1.0, base_urgency * multiplier)
    
    def should_force_injection(self, elapsed_ms: int, urgency: float) -> bool:
        """Determine if injection should be forced based on timing"""
        window = self.get_injection_window()
        
        # Force injection if running out of time
        if elapsed_ms >= window * 0.8:
            return True
        
        # Force high urgency items in server+auto mode
        if self.vad_mode == "server" and self.auto_response and urgency >= 0.8:
            return elapsed_ms >= window * 0.3
        
        return False
    
    def get_monitoring_interval(self) -> float:
        """Get recommended monitoring interval in seconds"""
        if self.vad_mode == "server" and self.auto_response:
            return 0.025  # 25ms - very fast
        elif self.vad_mode == "server":
            return 0.05   # 50ms - fast
        elif not self.auto_response:
            return 0.1    # 100ms - normal
        else:
            return 0.075  # 75ms - medium
    
    def get_strategy_recommendation(self) -> str:
        """Recommend injection strategy based on mode"""
        if self.vad_mode == "server" and self.auto_response:
            return "aggressive"  # Must be aggressive to inject in time
        elif self.vad_mode == "client" and not self.auto_response:
            return "conservative"  # Can be conservative with full control
        else:
            return "adaptive"  # Adapt to situation
    
    def estimate_response_time(self, silence_start: Optional[datetime] = None) -> Optional[int]:
        """Estimate time until auto-response triggers (in ms)"""
        if not self.auto_response:
            return None  # No auto-response
        
        if self.vad_mode == "server":
            # Server will trigger after silence_duration_ms
            if silence_start:
                elapsed = (datetime.now() - silence_start).total_seconds() * 1000
                remaining = max(0, self.server_silence_ms - elapsed)
                return int(remaining)
            else:
                return self.server_silence_ms
        else:
            # Client VAD - estimate based on typical patterns
            return 800  # Rough estimate
    
    def format_state_info(self) -> Dict[str, Any]:
        """Format current state information"""
        return {
            "vad_mode": self.vad_mode,
            "auto_response": self.auto_response,
            "injection_window_ms": self.get_injection_window(),
            "monitoring_interval_s": self.get_monitoring_interval(),
            "recommended_strategy": self.get_strategy_recommendation(),
            "detection_config": self.get_detection_config()
        }