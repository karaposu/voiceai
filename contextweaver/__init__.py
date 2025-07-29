"""
Context Weaver

Intelligently weaves context into voice conversations.
Detects optimal moments and strategies for context integration.
"""

from .engine import ContextWeaver
from .detectors import SilenceDetector, PauseDetector, TopicChangeDetector
from .strategies import InjectionStrategy

__all__ = [
    "ContextWeaver",
    "SilenceDetector", 
    "PauseDetector",
    "TopicChangeDetector",
    "InjectionStrategy"
]