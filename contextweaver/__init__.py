"""
Context Weaver

Intelligently weaves context into voice conversations.
Detects optimal moments and strategies for context integration.
"""

from .engine import ContextWeaver
from .detectors import SilenceDetector, PauseDetector, TopicChangeDetector
from .strategies import InjectionStrategy
from .schema import ContextToInject, InjectionTiming, ContextPriority

__all__ = [
    "ContextWeaver",
    "SilenceDetector", 
    "PauseDetector",
    "TopicChangeDetector",
    "InjectionStrategy",
    "ContextToInject",
    "InjectionTiming",
    "ContextPriority"
]