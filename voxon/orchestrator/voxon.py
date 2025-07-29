"""
Voxon - Main Orchestration Class

High-level interface for intelligent voice conversations.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..conversation import Conversation, ConversationManager
from .engine_coordinator import EngineCoordinator


@dataclass
class VoxonConfig:
    """Configuration for Voxon orchestrator"""
    
    # Engine configurations
    voxengine_config: Optional[Dict[str, Any]] = None
    context_engine_config: Optional[Dict[str, Any]] = None
    
    # Conversation settings
    enable_memory: bool = True
    memory_backend: str = "sqlite"  # sqlite, redis, postgres
    memory_path: Optional[str] = None
    
    # Templates
    default_template: str = "general"
    template_path: Optional[str] = None
    
    # Performance
    max_concurrent_conversations: int = 100
    conversation_timeout_minutes: int = 30
    
    # Features
    enable_analytics: bool = True
    enable_learning: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_conversations: bool = True


class Voxon:
    """
    Voxon - Conversation Intelligence Orchestrator
    
    Manages the coordination between VoxEngine (voice I/O) and
    ContextWeaver (intelligent context) to create
    sophisticated voice conversations.
    
    Example:
        voxon = Voxon()
        voxon.set_voice_engine(voxengine)
        voxon.set_context_engine(context_engine)
        
        conversation = await voxon.start_conversation(
            template="customer_support",
            context={"user_id": "123"}
        )
    """
    
    def __init__(self, config: Optional[VoxonConfig] = None):
        self.config = config or VoxonConfig()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Engine references
        self.voice_engine = None
        self.context_engine = None
        
        # Managers
        self.conversation_manager = ConversationManager(
            max_concurrent=self.config.max_concurrent_conversations,
            timeout_minutes=self.config.conversation_timeout_minutes
        )
        self.engine_coordinator = EngineCoordinator(logger=self.logger)
        
        # State
        self.is_initialized = False
        self._active_conversations: Dict[str, Conversation] = {}
        
        self.logger.info("Voxon initialized")
    
    def set_voice_engine(self, engine):
        """
        Set the VoxEngine instance.
        
        Args:
            engine: VoxEngine instance for voice I/O
        """
        self.voice_engine = engine
        self.engine_coordinator.set_voice_engine(engine)
        self.logger.info("Voice engine set")
    
    def set_context_engine(self, engine):
        """
        Set the ContextWeaver instance.
        
        Args:
            engine: ContextWeaver for intelligent context
        """
        self.context_engine = engine
        self.engine_coordinator.set_context_engine(engine)
        self.logger.info("Context engine set")
    
    async def initialize(self):
        """Initialize Voxon and its components"""
        if self.is_initialized:
            return
        
        # Validate engines are set
        if not self.voice_engine:
            raise ValueError("Voice engine not set. Call set_voice_engine() first.")
        if not self.context_engine:
            raise ValueError("Context engine not set. Call set_context_engine() first.")
        
        # Initialize coordinator
        await self.engine_coordinator.initialize()
        
        # Initialize conversation manager
        await self.conversation_manager.initialize(self.config)
        
        self.is_initialized = True
        self.logger.info("Voxon initialized successfully")
    
    async def start_conversation(
        self,
        template: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Conversation:
        """
        Start a new conversation.
        
        Args:
            template: Conversation template to use
            context: Initial context for the conversation
            user_id: User identifier for memory/history
            **kwargs: Additional conversation parameters
            
        Returns:
            Conversation instance
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Create conversation
        conversation = await self.conversation_manager.create_conversation(
            template=template or self.config.default_template,
            context=context or {},
            user_id=user_id,
            voice_engine=self.voice_engine,
            context_engine=self.context_engine,
            coordinator=self.engine_coordinator,
            **kwargs
        )
        
        # Track active conversation
        self._active_conversations[conversation.id] = conversation
        
        # Start conversation
        await conversation.start()
        
        self.logger.info(f"Started conversation {conversation.id}")
        return conversation
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get an active conversation by ID"""
        return self._active_conversations.get(conversation_id)
    
    async def end_conversation(self, conversation_id: str):
        """End a conversation"""
        conversation = self._active_conversations.get(conversation_id)
        if conversation:
            await conversation.end()
            del self._active_conversations[conversation_id]
            self.logger.info(f"Ended conversation {conversation_id}")
    
    async def get_conversation_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum conversations to return
            
        Returns:
            List of conversation summaries
        """
        return await self.conversation_manager.get_user_history(user_id, limit)
    
    async def load_conversation(
        self,
        conversation_id: str,
        resume: bool = True
    ) -> Optional[Conversation]:
        """
        Load a previous conversation.
        
        Args:
            conversation_id: Conversation to load
            resume: Whether to resume the conversation
            
        Returns:
            Conversation instance or None
        """
        conversation = await self.conversation_manager.load_conversation(
            conversation_id,
            voice_engine=self.voice_engine,
            context_engine=self.context_engine,
            coordinator=self.engine_coordinator
        )
        
        if conversation and resume:
            self._active_conversations[conversation.id] = conversation
            await conversation.resume()
        
        return conversation
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics across all conversations"""
        return {
            "active_conversations": len(self._active_conversations),
            "total_conversations": self.conversation_manager.total_conversations,
            "engine_stats": {
                "voice": self.voice_engine.get_metrics() if self.voice_engine else {},
                "context": self.context_engine.get_stats() if self.context_engine else {}
            },
            "coordinator_stats": self.engine_coordinator.get_stats()
        }
    
    async def shutdown(self):
        """Shutdown Voxon and cleanup resources"""
        # End all active conversations
        for conv_id in list(self._active_conversations.keys()):
            await self.end_conversation(conv_id)
        
        # Shutdown components
        await self.conversation_manager.shutdown()
        await self.engine_coordinator.shutdown()
        
        self.is_initialized = False
        self.logger.info("Voxon shutdown complete")
    
    # Context helpers
    
    def inject_context(
        self,
        conversation_id: str,
        information: Optional[Dict[str, Any]] = None,
        strategy: Optional[Dict[str, Any]] = None,
        attention: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Inject context into a specific conversation.
        
        Args:
            conversation_id: Target conversation
            information: What the AI should know
            strategy: How the AI should behave
            attention: What the AI should focus on
            **kwargs: Additional context parameters
        """
        conversation = self._active_conversations.get(conversation_id)
        if conversation:
            conversation.inject_context(
                information=information,
                strategy=strategy,
                attention=attention,
                **kwargs
            )
    
    def update_global_context(
        self,
        information: Optional[Dict[str, Any]] = None,
        strategy: Optional[Dict[str, Any]] = None,
        attention: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Update context for all conversations.
        
        Useful for system-wide updates like:
        - Service outages
        - Policy changes
        - Global events
        """
        for conversation in self._active_conversations.values():
            conversation.inject_context(
                information=information,
                strategy=strategy,
                attention=attention,
                source="global_update",
                **kwargs
            )