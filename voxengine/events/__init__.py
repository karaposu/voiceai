"""
Voice Engine Event System

A modern event-driven architecture for handling all voice engine events.
"""

from .event_types import (
    Event,
    EventType,
    AudioEvent,
    TextEvent,
    ConnectionEvent,
    ConversationEvent,
    TimingEvent,
    FunctionCallEvent,
    MetricsEvent,
    ErrorEvent
)

from .event_emitter import (
    EventEmitter,
    EventHandler,
    EventFilter,
    HandlerInfo,
    event_handler,
    get_global_emitter
)

__all__ = [
    # Event types
    "Event",
    "EventType",
    "AudioEvent", 
    "TextEvent",
    "ConnectionEvent",
    "ConversationEvent",
    "TimingEvent",
    "FunctionCallEvent",
    "MetricsEvent",
    "ErrorEvent",
    
    # Emitter
    "EventEmitter",
    "EventHandler",
    "EventFilter",
    "HandlerInfo",
    "event_handler",
    "get_global_emitter"
]