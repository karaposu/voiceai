"""
Event Emitter System

A flexible, async-first event emitter for the voice engine.
Supports type-safe event handling, filtering, and error handling.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Optional, Union, Set, Any, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict
import weakref
import inspect
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from .event_types import Event, EventType, ErrorEvent


# Type definitions
EventHandler = Union[
    Callable[[Event], None],
    Callable[[Event], asyncio.Future]
]

EventFilter = Callable[[Event], bool]

T = TypeVar('T', bound=Event)


@dataclass
class HandlerInfo:
    """Information about a registered handler"""
    handler: EventHandler
    event_types: Set[EventType]
    filter: Optional[EventFilter] = None
    once: bool = False
    weak: bool = False
    priority: int = 0
    handler_id: str = ""
    
    def __post_init__(self):
        if not self.handler_id:
            self.handler_id = f"{id(self.handler)}_{hash(tuple(self.event_types))}"


class EventEmitter:
    """
    Async event emitter for voice engine events.
    
    Features:
    - Type-safe event handling
    - Async and sync handler support
    - Event filtering
    - Priority-based handler execution
    - Weak references for auto-cleanup
    - Error handling with fallback
    - Event replay capability
    """
    
    def __init__(
        self,
        name: str = "VoiceEngine",
        max_handlers_per_event: int = 100,
        enable_history: bool = True,
        history_size: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        self.name = name
        self.logger = logger or logging.getLogger(f"{__name__}.{name}")
        self.max_handlers_per_event = max_handlers_per_event
        
        # Handler storage
        self._handlers: Dict[EventType, List[HandlerInfo]] = defaultdict(list)
        self._global_handlers: List[HandlerInfo] = []  # Handlers for all events
        self._once_handlers: Set[str] = set()
        
        # Error handling
        self._error_handlers: List[Callable[[Exception, Event], None]] = []
        
        # Event history
        self.enable_history = enable_history
        self._history: List[Event] = []
        self._history_size = history_size
        
        # Metrics
        self._metrics = {
            "events_emitted": 0,
            "events_handled": 0,
            "errors_caught": 0,
            "handlers_registered": 0
        }
        
        # Thread pool for sync handlers
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Event queue for buffered emission (use regular queue for thread safety)
        import queue
        self._event_queue: queue.Queue[Event] = queue.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Start background processor if event loop exists
        try:
            loop = asyncio.get_running_loop()
            self._start_processor()
        except RuntimeError:
            pass
    
    # Core Methods
    
    def on(
        self,
        event_type: Union[EventType, List[EventType], str],
        handler: EventHandler,
        *,
        filter: Optional[EventFilter] = None,
        once: bool = False,
        weak: bool = False,
        priority: int = 0
    ) -> str:
        """
        Register an event handler.
        
        Args:
            event_type: Event type(s) to listen for, or "*" for all events
            handler: Function to handle the event
            filter: Optional filter function
            once: Remove handler after first call
            weak: Use weak reference (auto-cleanup)
            priority: Handler priority (higher = earlier execution)
            
        Returns:
            Handler ID for removal
        """
        # Convert to list of event types
        if event_type == "*":
            event_types = None  # Global handler
        elif isinstance(event_type, str):
            event_types = {EventType(event_type)}
        elif isinstance(event_type, EventType):
            event_types = {event_type}
        else:
            event_types = set(event_type)
        
        # Create handler info
        handler_info = HandlerInfo(
            handler=handler,
            event_types=event_types or set(),
            filter=filter,
            once=once,
            weak=weak,
            priority=priority
        )
        
        # Store handler
        if event_types is None:
            # Global handler
            self._global_handlers.append(handler_info)
            self._global_handlers.sort(key=lambda h: h.priority, reverse=True)
        else:
            # Specific event types
            for evt_type in event_types:
                if len(self._handlers[evt_type]) >= self.max_handlers_per_event:
                    self.logger.warning(
                        f"Maximum handlers ({self.max_handlers_per_event}) "
                        f"reached for event {evt_type}"
                    )
                    continue
                
                self._handlers[evt_type].append(handler_info)
                self._handlers[evt_type].sort(key=lambda h: h.priority, reverse=True)
        
        self._metrics["handlers_registered"] += 1
        self.logger.debug(f"Registered handler {handler_info.handler_id} for {event_type}")
        
        return handler_info.handler_id
    
    def off(self, handler_id: str) -> bool:
        """
        Remove a handler by ID.
        
        Args:
            handler_id: Handler ID returned by on()
            
        Returns:
            True if handler was found and removed
        """
        removed = False
        
        # Remove from specific handlers
        for event_type, handlers in self._handlers.items():
            for i, handler_info in enumerate(handlers):
                if handler_info.handler_id == handler_id:
                    handlers.pop(i)
                    removed = True
                    break
        
        # Remove from global handlers
        for i, handler_info in enumerate(self._global_handlers):
            if handler_info.handler_id == handler_id:
                self._global_handlers.pop(i)
                removed = True
                break
        
        if removed:
            self._metrics["handlers_registered"] -= 1
            self.logger.debug(f"Removed handler {handler_id}")
        
        return removed
    
    async def emit(self, event: Event, sync: bool = False) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event: Event to emit
            sync: If True, wait for all handlers to complete
        """
        self._metrics["events_emitted"] += 1
        
        # Add to history
        if self.enable_history:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history.pop(0)
        
        # Get applicable handlers
        handlers = self._get_handlers_for_event(event)
        
        if sync:
            # Synchronous emission
            await self._emit_sync(event, handlers)
        else:
            # Asynchronous emission (fire-and-forget)
            asyncio.create_task(self._emit_async(event, handlers))
    
    def emit_sync(self, event: Event) -> None:
        """Synchronous version of emit for non-async contexts"""
        # Store event for processing
        self._event_queue.put_nowait(event)
        
        # Try to schedule processing if there's a running loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, process immediately
            loop.create_task(self._process_queued_event())
        except RuntimeError:
            # No running loop - events will be processed when one is available
            pass
    
    async def _process_queued_event(self) -> None:
        """Process a single queued event"""
        try:
            event = self._event_queue.get_nowait()
            await self.emit(event, sync=False)
        except asyncio.QueueEmpty:
            pass
    
    # Handler execution
    
    async def _emit_sync(self, event: Event, handlers: List[HandlerInfo]) -> None:
        """Emit event and wait for all handlers"""
        tasks = []
        
        for handler_info in handlers:
            if handler_info.filter and not handler_info.filter(event):
                continue
            
            task = self._call_handler(handler_info, event)
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _emit_async(self, event: Event, handlers: List[HandlerInfo]) -> None:
        """Emit event without waiting for handlers"""
        for handler_info in handlers:
            if handler_info.filter and not handler_info.filter(event):
                continue
            
            asyncio.create_task(self._call_handler(handler_info, event))
    
    async def _call_handler(self, handler_info: HandlerInfo, event: Event) -> None:
        """Call a single handler with error handling"""
        try:
            handler = handler_info.handler
            
            # Handle weak references
            if handler_info.weak and isinstance(handler, weakref.ref):
                handler = handler()
                if handler is None:
                    # Handler was garbage collected
                    self.off(handler_info.handler_id)
                    return
            
            # Check if handler is async
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                # Run sync handler in thread pool
                await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool, handler, event
                )
            
            self._metrics["events_handled"] += 1
            
            # Handle once handlers
            if handler_info.once:
                self.off(handler_info.handler_id)
                
        except Exception as e:
            self._metrics["errors_caught"] += 1
            self.logger.exception(f"Error in handler {handler_info.handler_id}: {e}")
            
            # Call error handlers
            for error_handler in self._error_handlers:
                try:
                    error_handler(e, event)
                except Exception as eh_error:
                    self.logger.exception(f"Error in error handler: {eh_error}")
            
            # Emit error event
            error_event = ErrorEvent(
                type=EventType.ERROR_GENERAL,
                error=e,
                error_message=f"Handler error: {e}",
                data={"original_event": event.to_dict()},
                source=self.name
            )
            await self.emit(error_event)
    
    def _get_handlers_for_event(self, event: Event) -> List[HandlerInfo]:
        """Get all handlers that should receive this event"""
        handlers = []
        
        # Add specific handlers
        if event.type in self._handlers:
            handlers.extend(self._handlers[event.type])
        
        # Add global handlers
        handlers.extend(self._global_handlers)
        
        # Sort by priority
        handlers.sort(key=lambda h: h.priority, reverse=True)
        
        return handlers
    
    # Utility methods
    
    def once(self, event_type: Union[EventType, List[EventType]], handler: EventHandler) -> str:
        """Register a one-time handler"""
        return self.on(event_type, handler, once=True)
    
    def on_error(self, handler: Callable[[Exception, Event], None]) -> None:
        """Register an error handler"""
        self._error_handlers.append(handler)
    
    def remove_all_handlers(self, event_type: Optional[EventType] = None) -> None:
        """Remove all handlers for an event type (or all if None)"""
        if event_type:
            self._handlers[event_type].clear()
        else:
            self._handlers.clear()
            self._global_handlers.clear()
            self._metrics["handlers_registered"] = 0
    
    def get_handler_count(self, event_type: Optional[EventType] = None) -> int:
        """Get number of registered handlers"""
        if event_type:
            return len(self._handlers.get(event_type, []))
        else:
            total = len(self._global_handlers)
            for handlers in self._handlers.values():
                total += len(handlers)
            return total
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get emitter metrics"""
        return {
            **self._metrics,
            "handler_count": self.get_handler_count(),
            "event_types_monitored": len(self._handlers),
            "history_size": len(self._history)
        }
    
    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Get event history"""
        if not self.enable_history:
            return []
        
        history = self._history
        if event_type:
            history = [e for e in history if e.type == event_type]
        
        return history[-limit:]
    
    def _start_processor(self) -> None:
        """Start the background event processor"""
        if not self._running:
            self._running = True
            loop = asyncio.get_running_loop()
            self._processing_task = loop.create_task(self._process_event_queue())
    
    async def _process_event_queue(self) -> None:
        """Process events from the queue"""
        while self._running:
            try:
                # Check for events in queue
                event = self._event_queue.get_nowait()
                await self.emit(event, sync=False)
            except:
                # No events, wait a bit
                await asyncio.sleep(0.01)
    
    # Context manager for temporary handlers
    
    @asynccontextmanager
    async def listen(self, event_type: Union[EventType, List[EventType]], handler: EventHandler):
        """Context manager for temporary event listening"""
        handler_id = self.on(event_type, handler)
        try:
            yield handler_id
        finally:
            self.off(handler_id)
    
    # Cleanup
    
    async def close(self) -> None:
        """Clean up resources"""
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        self._thread_pool.shutdown(wait=True)
        self.remove_all_handlers()
        self._history.clear()


# Convenience decorators

def event_handler(event_type: Union[EventType, List[EventType]], **kwargs):
    """Decorator for event handlers"""
    def decorator(func):
        func._event_type = event_type
        func._event_kwargs = kwargs
        return func
    return decorator


# Global emitter instance (optional)
_global_emitter: Optional[EventEmitter] = None


def get_global_emitter() -> EventEmitter:
    """Get or create global emitter instance"""
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = EventEmitter("Global")
    return _global_emitter