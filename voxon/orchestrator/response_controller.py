"""
Response Controller

Manages response timing and coordination between context injection
and voice engine response triggering. Ensures smooth integration
with both manual and automatic response modes.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging


class ResponseMode(str, Enum):
    """Response triggering modes"""
    MANUAL = "manual"  # Explicit trigger_response() calls
    AUTOMATIC = "automatic"  # Server-side VAD auto-response
    HYBRID = "hybrid"  # Automatic with manual override


@dataclass
class InjectionRequest:
    """Request to inject context before response"""
    context: str
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None  # Latest time to inject
    callback: Optional[Callable] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if request has expired"""
        if self.deadline is None:
            return False
        return time.time() > self.deadline
    
    @property
    def time_remaining(self) -> float:
        """Time remaining before deadline"""
        if self.deadline is None:
            return float('inf')
        return max(0, self.deadline - time.time())


class ResponseController:
    """
    Controls response timing and context injection coordination.
    
    Key responsibilities:
    - Monitor response readiness
    - Queue and prioritize injection requests
    - Coordinate injection-response sequencing
    - Handle timeout scenarios
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Response control state
        self.response_mode = ResponseMode.AUTOMATIC
        self.response_ready = False
        self.response_pending = False
        self.last_response_time = 0.0
        
        # Injection queue
        self.injection_queue: List[InjectionRequest] = []
        self._queue_lock = asyncio.Lock()
        
        # Timing configuration
        self.min_injection_gap_ms = 50  # Min time between injections
        self.max_injection_delay_ms = 2000  # Max wait before forcing response
        self.injection_timeout_ms = 1000  # Timeout for individual injection
        
        # Performance tracking
        self.metrics = {
            "total_injections": 0,
            "successful_injections": 0,
            "timeout_injections": 0,
            "expired_injections": 0,
            "avg_injection_time_ms": 0.0,
            "response_triggers": 0
        }
        
        # Callbacks
        self.on_injection_complete: Optional[Callable] = None
        self.on_response_triggered: Optional[Callable] = None
    
    async def set_response_mode(self, mode: ResponseMode, auto_response: bool = True):
        """Update response mode based on VAD configuration"""
        self.response_mode = mode
        self.logger.info(f"Response mode set to: {mode.value}, auto_response: {auto_response}")
        
        # Adjust timing based on mode
        if mode == ResponseMode.AUTOMATIC and auto_response:
            # Need fast injection for auto-response
            self.max_injection_delay_ms = 200
            self.injection_timeout_ms = 100
        elif mode == ResponseMode.MANUAL:
            # More relaxed timing for manual control
            self.max_injection_delay_ms = 2000
            self.injection_timeout_ms = 1000
        else:
            # Hybrid mode - balanced timing
            self.max_injection_delay_ms = 500
            self.injection_timeout_ms = 300
    
    async def request_injection(
        self,
        context: str,
        priority: int = 5,
        deadline_ms: Optional[int] = None
    ) -> bool:
        """
        Request context injection before next response.
        
        Args:
            context: Context to inject
            priority: Priority (1-10, higher = more important)
            deadline_ms: Max time to wait for injection opportunity
            
        Returns:
            True if request queued, False if rejected
        """
        # Calculate deadline
        deadline = None
        if deadline_ms is not None:
            deadline = time.time() + (deadline_ms / 1000.0)
        
        request = InjectionRequest(
            context=context,
            priority=priority,
            deadline=deadline
        )
        
        async with self._queue_lock:
            # Check if we can accept request
            if len(self.injection_queue) >= 10:
                self.logger.warning("Injection queue full, rejecting request")
                return False
            
            # Add to queue (sorted by priority)
            self.injection_queue.append(request)
            self.injection_queue.sort(key=lambda r: r.priority, reverse=True)
            
        self.logger.debug(f"Injection request queued: priority={priority}, queue_size={len(self.injection_queue)}")
        return True
    
    async def get_injection_window(self) -> Dict[str, Any]:
        """
        Calculate current injection window based on response state.
        
        Returns:
            Window information including available time and recommended action
        """
        current_time = time.time()
        
        # Remove expired requests
        await self._prune_expired_requests()
        
        # Calculate window based on mode
        if self.response_mode == ResponseMode.AUTOMATIC:
            # Very short window for auto-response
            window_ms = 100
            urgency = "immediate"
        elif self.response_mode == ResponseMode.MANUAL:
            # Comfortable window for manual control
            window_ms = 1000
            urgency = "relaxed"
        else:
            # Hybrid - medium window
            window_ms = 300
            urgency = "moderate"
        
        # Check if response is pending
        if self.response_pending:
            window_ms = min(window_ms, 50)  # Very short window
            urgency = "critical"
        
        # Get next injection request
        next_request = None
        async with self._queue_lock:
            if self.injection_queue:
                next_request = self.injection_queue[0]
        
        return {
            "window_ms": window_ms,
            "urgency": urgency,
            "has_pending": next_request is not None,
            "pending_priority": next_request.priority if next_request else None,
            "queue_size": len(self.injection_queue),
            "response_pending": self.response_pending,
            "response_mode": self.response_mode.value
        }
    
    async def execute_injection(
        self,
        inject_callback: Callable[[str], asyncio.Task],
        trigger_response_callback: Optional[Callable] = None
    ) -> bool:
        """
        Execute pending injection and optionally trigger response.
        
        Args:
            inject_callback: Async function to inject context
            trigger_response_callback: Optional function to trigger response
            
        Returns:
            True if injection executed, False if nothing to inject
        """
        start_time = time.time()
        
        # Get next request
        request = None
        async with self._queue_lock:
            if self.injection_queue:
                request = self.injection_queue.pop(0)
        
        if not request:
            return False
        
        # Check if expired
        if request.is_expired:
            self.metrics["expired_injections"] += 1
            self.logger.warning("Injection request expired, skipping")
            return False
        
        try:
            # Execute injection with timeout
            self.logger.debug(f"Executing injection: priority={request.priority}")
            await asyncio.wait_for(
                inject_callback(request.context),
                timeout=self.injection_timeout_ms / 1000.0
            )
            
            # Update metrics
            injection_time_ms = (time.time() - start_time) * 1000
            self._update_injection_metrics(injection_time_ms, success=True)
            
            # Callback
            if self.on_injection_complete:
                self.on_injection_complete(request.context, injection_time_ms)
            
            # Trigger response if needed
            if trigger_response_callback and self.should_trigger_response():
                await self._trigger_response(trigger_response_callback)
            
            return True
            
        except asyncio.TimeoutError:
            self.metrics["timeout_injections"] += 1
            self.logger.error(f"Injection timeout after {self.injection_timeout_ms}ms")
            self._update_injection_metrics(self.injection_timeout_ms, success=False)
            return False
        except Exception as e:
            self.logger.error(f"Injection error: {e}")
            return False
    
    def should_trigger_response(self) -> bool:
        """Check if response should be triggered after injection"""
        if self.response_mode == ResponseMode.AUTOMATIC:
            # Auto-response handles it
            return False
        elif self.response_mode == ResponseMode.MANUAL:
            # Always trigger in manual mode
            return True
        else:
            # Hybrid - trigger if no recent response
            time_since_response = time.time() - self.last_response_time
            return time_since_response > 2.0  # 2 seconds threshold
    
    async def _trigger_response(self, callback: Callable):
        """Trigger response through callback"""
        try:
            self.logger.debug("Triggering response after injection")
            await callback()
            self.last_response_time = time.time()
            self.response_pending = False
            self.metrics["response_triggers"] += 1
            
            if self.on_response_triggered:
                self.on_response_triggered()
                
        except Exception as e:
            self.logger.error(f"Error triggering response: {e}")
    
    async def _prune_expired_requests(self):
        """Remove expired injection requests"""
        async with self._queue_lock:
            before_count = len(self.injection_queue)
            self.injection_queue = [
                req for req in self.injection_queue
                if not req.is_expired
            ]
            expired_count = before_count - len(self.injection_queue)
            
            if expired_count > 0:
                self.metrics["expired_injections"] += expired_count
                self.logger.debug(f"Pruned {expired_count} expired requests")
    
    def _update_injection_metrics(self, time_ms: float, success: bool):
        """Update injection performance metrics"""
        self.metrics["total_injections"] += 1
        if success:
            self.metrics["successful_injections"] += 1
        
        # Update average time
        total = self.metrics["total_injections"]
        old_avg = self.metrics["avg_injection_time_ms"]
        self.metrics["avg_injection_time_ms"] = (old_avg * (total - 1) + time_ms) / total
    
    def mark_response_ready(self):
        """Mark that response is ready to be triggered"""
        self.response_ready = True
        self.response_pending = False
    
    def mark_response_pending(self):
        """Mark that response is about to be triggered"""
        self.response_pending = True
        self.response_ready = False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get controller metrics"""
        return {
            **self.metrics,
            "queue_size": len(self.injection_queue),
            "response_mode": self.response_mode.value,
            "success_rate": (self.metrics["successful_injections"] / 
                           max(1, self.metrics["total_injections"]))
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            "total_injections": 0,
            "successful_injections": 0,
            "timeout_injections": 0,
            "expired_injections": 0,
            "avg_injection_time_ms": 0.0,
            "response_triggers": 0
        }