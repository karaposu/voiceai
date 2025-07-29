"""
Conversation Class

High-level conversation abstraction that coordinates engines.
"""

import asyncio
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority


class Conversation:
    """
    High-level conversation abstraction.
    
    Manages a single conversation session, coordinating between
    VoxEngine and ContextWeaver through the EngineCoordinator.
    """
    
    def __init__(
        self,
        conversation_id: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        voice_engine=None,
        context_engine=None,
        coordinator=None
    ):
        self.id = conversation_id or str(uuid.uuid4())
        self.template = template or "general"
        self.initial_context = context or {}
        self.user_id = user_id
        
        # Engine references
        self.voice_engine = voice_engine
        self.context_engine = context_engine
        self.coordinator = coordinator
        
        # State
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        self.is_active = False
        
        # Conversation data
        self.metadata: Dict[str, Any] = {}
        self.summary: Optional[str] = None
    
    async def start(self):
        """Start the conversation"""
        if self.is_active:
            return
        
        self.started_at = datetime.now()
        self.is_active = True
        
        # Initialize context for conversation
        if self.initial_context:
            self.inject_context(
                information=self.initial_context,
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.HIGH,
                source="conversation_start"
            )
    
    async def end(self):
        """End the conversation"""
        if not self.is_active:
            return
        
        self.ended_at = datetime.now()
        self.is_active = False
        
        # Could generate summary here
    
    async def resume(self):
        """Resume a previously ended conversation"""
        self.is_active = True
        
        # Inject resumption context
        self.inject_context(
            information={"conversation_resumed": True},
            timing=InjectionTiming.IMMEDIATE,
            source="conversation_resume"
        )
    
    def inject_context(
        self,
        information: Optional[Dict[str, Any]] = None,
        strategy: Optional[Dict[str, Any]] = None,
        attention: Optional[Dict[str, Any]] = None,
        timing: InjectionTiming = InjectionTiming.NEXT_PAUSE,
        priority: ContextPriority = ContextPriority.MEDIUM,
        **kwargs
    ):
        """
        Inject context into this conversation.
        
        Args:
            information: What the AI should know
            strategy: How the AI should behave
            attention: What the AI should focus on
            timing: When to inject the context
            priority: Priority level
            **kwargs: Additional context parameters
        """
        if not self.context_engine:
            return
        
        context = ContextToInject(
            information=information or {},
            strategy=strategy or {},
            attention=attention or {},
            timing=timing,
            priority=priority,
            source=kwargs.get("source", f"conversation_{self.id}"),
            **kwargs
        )
        
        self.context_engine.add_context(context)
    
    def get_duration(self) -> Optional[float]:
        """Get conversation duration in seconds"""
        if not self.started_at:
            return None
        
        end_time = self.ended_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "template": self.template,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.get_duration(),
            "is_active": self.is_active,
            "metadata": self.metadata,
            "summary": self.summary
        }