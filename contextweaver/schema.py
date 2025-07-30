# Schema definitions for ContextWeaver


from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
import uuid
import time


class InjectionTiming(Enum):
    """When the context should be injected"""
    IMMEDIATE = "immediate"           # Inject ASAP, potentially interrupting
    NEXT_TURN = "next_turn"          # At next speaker change
    NEXT_PAUSE = "next_pause"        # During next silence/pause
    ON_TOPIC = "on_topic"            # When topic matches conditions
    ON_TRIGGER = "on_trigger"        # When specific trigger occurs
    SCHEDULED = "scheduled"          # At specific time
    LAZY = "lazy"                    # Whenever convenient
    MANUAL = "manual"                # Only when explicitly requested


class ContextPriority(Enum):
    """Priority levels for context injection"""
    CRITICAL = 10    # Safety, legal, must be delivered
    HIGH = 8         # Important for conversation quality
    MEDIUM = 5       # Standard context
    LOW = 3          # Nice to have
    BACKGROUND = 1   # Only if nothing else pending


@dataclass
class ContextToInject:
    """
    Context data structure for injection into AI conversations.
    
    Three core dimensions:
    - information: What the AI should know (facts, state, data)
    - strategy: How the AI should behave (tone, approach, rules)  
    - attention: What the AI should focus on (priorities, goals)
    
    Control dimensions:
    - timing: When to inject this context
    - conditions: Requirements for injection to be valid
    """
    
    # Core context dimensions
    information: Dict[str, Any] = field(default_factory=dict)
    strategy: Dict[str, Any] = field(default_factory=dict)
    attention: Dict[str, Any] = field(default_factory=dict)
    
    # Control fields
    timing: InjectionTiming = InjectionTiming.NEXT_PAUSE
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    priority: Union[ContextPriority, int] = ContextPriority.MEDIUM
    ttl_seconds: Optional[int] = None
    source: Optional[str] = None  # What system created this context
    
    # Injection tracking
    injection_count: int = 0
    last_injected_at: Optional[float] = None
    max_injections: Optional[int] = None  # Limit how many times to inject
    
    def __post_init__(self):
        """Validate and normalize after initialization"""
        # Convert priority to enum if needed
        if isinstance(self.priority, int):
            # Find closest priority level
            if self.priority >= 9:
                self.priority = ContextPriority.CRITICAL
            elif self.priority >= 7:
                self.priority = ContextPriority.HIGH
            elif self.priority >= 4:
                self.priority = ContextPriority.MEDIUM
            elif self.priority >= 2:
                self.priority = ContextPriority.LOW
            else:
                self.priority = ContextPriority.BACKGROUND
    
    @property
    def priority_value(self) -> int:
        """Get numeric priority value"""
        if isinstance(self.priority, ContextPriority):
            return self.priority.value
        return int(self.priority)
    
    def is_expired(self) -> bool:
        """Check if context has expired based on TTL"""
        if self.ttl_seconds is None:
            return False
        age = time.time() - self.created_at
        return age > self.ttl_seconds
    
    def is_empty(self) -> bool:
        """Check if context has any content"""
        return not (self.information or self.strategy or self.attention)
    
    def should_inject(self, conversation_state: Dict[str, Any]) -> bool:
        """
        Determine if context should be injected based on conditions.
        
        Args:
            conversation_state: Current state of conversation
            
        Returns:
            True if all conditions are met
        """
        if self.is_expired():
            return False
            
        if self.max_injections and self.injection_count >= self.max_injections:
            return False
        
        # Check each condition type
        for condition_type, condition_value in self.conditions.items():
            if not self._check_condition(condition_type, condition_value, conversation_state):
                return False
                
        return True
    
    def _check_condition(self, condition_type: str, condition_value: Any, 
                        state: Dict[str, Any]) -> bool:
        """Check a single condition"""
        
        # Topic matching
        if condition_type == "topics_include":
            current_topics = state.get("current_topics", [])
            return any(topic in current_topics for topic in condition_value)
            
        elif condition_type == "topics_exclude":
            current_topics = state.get("current_topics", [])
            return not any(topic in current_topics for topic in condition_value)
        
        # Mood matching
        elif condition_type == "user_mood":
            current_mood = state.get("user_mood")
            if isinstance(condition_value, list):
                return current_mood in condition_value
            return current_mood == condition_value
        
        # Conversation length
        elif condition_type == "conversation_length_min":
            return state.get("message_count", 0) >= condition_value
            
        elif condition_type == "conversation_length_max":
            return state.get("message_count", 0) <= condition_value
        
        # Time-based
        elif condition_type == "time_since_last_injection_min":
            if self.last_injected_at is None:
                return True
            return (time.time() - self.last_injected_at) >= condition_value
        
        # Custom function
        elif condition_type == "custom" and callable(condition_value):
            return condition_value(state)
        
        # Unknown condition type - be permissive
        return True
    
    def merge_with(self, other: 'ContextToInject', 
                   merge_strategy: str = "other_priority") -> 'ContextToInject':
        """
        Merge with another context.
        
        Args:
            other: Context to merge with
            merge_strategy: How to handle conflicts
                - "other_priority": Other context values override
                - "self_priority": Self values override  
                - "combine": Merge dictionaries deeply
        """
        if merge_strategy == "other_priority":
            return ContextToInject(
                information={**self.information, **other.information},
                strategy={**self.strategy, **other.strategy},
                attention={**self.attention, **other.attention},
                timing=other.timing if other.timing != InjectionTiming.NEXT_PAUSE else self.timing,
                conditions={**self.conditions, **other.conditions},
                priority=max(self.priority_value, other.priority_value),
                ttl_seconds=min(self.ttl_seconds or float('inf'), 
                              other.ttl_seconds or float('inf')) if self.ttl_seconds or other.ttl_seconds else None,
                source=f"{self.source}+{other.source}" if self.source and other.source else (self.source or other.source)
            )
        # Add other strategies as needed
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
    
    def mark_injected(self):
        """Mark that this context was injected"""
        self.injection_count += 1
        self.last_injected_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "context_id": self.context_id,
            "information": self.information,
            "strategy": self.strategy,
            "attention": self.attention,
            "timing": self.timing.value,
            "conditions": self.conditions,
            "priority": self.priority_value,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "source": self.source,
            "injection_count": self.injection_count,
            "last_injected_at": self.last_injected_at,
            "max_injections": self.max_injections
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextToInject':
        """Create from dictionary"""
        # Handle timing enum
        timing_value = data.get("timing", "next_pause")
        timing = InjectionTiming(timing_value) if isinstance(timing_value, str) else timing_value
        
        return cls(
            information=data.get("information", {}),
            strategy=data.get("strategy", {}),
            attention=data.get("attention", {}),
            timing=timing,
            conditions=data.get("conditions", {}),
            context_id=data.get("context_id", str(uuid.uuid4())),
            created_at=data.get("created_at", time.time()),
            priority=data.get("priority", ContextPriority.MEDIUM),
            ttl_seconds=data.get("ttl_seconds"),
            source=data.get("source"),
            injection_count=data.get("injection_count", 0),
            last_injected_at=data.get("last_injected_at"),
            max_injections=data.get("max_injections")
        )
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"ContextToInject(id={self.context_id[:8]}..., "
                f"priority={self.priority_value}, "
                f"timing={self.timing.value}, "
                f"expired={self.is_expired()}, "
                f"injections={self.injection_count})")


# Convenience factory functions
def create_immediate_context(information: Dict[str, Any] = None,
                           strategy: Dict[str, Any] = None,
                           attention: Dict[str, Any] = None) -> ContextToInject:
    """Create a context for immediate injection"""
    return ContextToInject(
        information=information or {},
        strategy=strategy or {},
        attention=attention or {},
        timing=InjectionTiming.IMMEDIATE,
        priority=ContextPriority.HIGH
    )


def create_conditional_context(conditions: Dict[str, Any],
                             information: Dict[str, Any] = None,
                             strategy: Dict[str, Any] = None,
                             attention: Dict[str, Any] = None) -> ContextToInject:
    """Create a context that injects based on conditions"""
    return ContextToInject(
        information=information or {},
        strategy=strategy or {},
        attention=attention or {},
        timing=InjectionTiming.ON_TRIGGER,
        conditions=conditions,
        priority=ContextPriority.MEDIUM
    )