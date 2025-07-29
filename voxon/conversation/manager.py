"""
Conversation Manager

Manages multiple conversations and their lifecycle.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
from pathlib import Path

from .conversation import Conversation


class ConversationManager:
    """
    Manages conversation lifecycle and persistence.
    
    Handles:
    - Creating new conversations
    - Loading previous conversations
    - Managing conversation history
    - Cleanup and resource management
    """
    
    def __init__(
        self,
        max_concurrent: int = 100,
        timeout_minutes: int = 30
    ):
        self.max_concurrent = max_concurrent
        self.timeout_minutes = timeout_minutes
        
        # State
        self.total_conversations = 0
        self._active_conversations: Dict[str, Conversation] = {}
        self._conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Persistence (simplified for now)
        self._storage_path = Path.home() / ".voxon" / "conversations"
    
    async def initialize(self, config: Any):
        """Initialize the manager with configuration"""
        # Setup storage
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load any persisted data
        await self._load_history()
    
    async def create_conversation(
        self,
        template: str,
        context: Dict[str, Any],
        user_id: Optional[str],
        voice_engine,
        context_engine,
        coordinator,
        **kwargs
    ) -> Conversation:
        """Create a new conversation"""
        
        # Check concurrent limit
        if len(self._active_conversations) >= self.max_concurrent:
            # Clean up old conversations
            await self._cleanup_stale_conversations()
            
            if len(self._active_conversations) >= self.max_concurrent:
                raise RuntimeError(f"Maximum concurrent conversations ({self.max_concurrent}) reached")
        
        # Create conversation
        conversation = Conversation(
            template=template,
            context=context,
            user_id=user_id,
            voice_engine=voice_engine,
            context_engine=context_engine,
            coordinator=coordinator
        )
        
        # Track it
        self._active_conversations[conversation.id] = conversation
        self.total_conversations += 1
        
        # Add to history
        if user_id:
            if user_id not in self._conversation_history:
                self._conversation_history[user_id] = []
            self._conversation_history[user_id].append({
                "id": conversation.id,
                "started_at": datetime.now().isoformat(),
                "template": template
            })
        
        return conversation
    
    async def load_conversation(
        self,
        conversation_id: str,
        voice_engine,
        context_engine,
        coordinator
    ) -> Optional[Conversation]:
        """Load a previous conversation"""
        
        # Check if already active
        if conversation_id in self._active_conversations:
            return self._active_conversations[conversation_id]
        
        # Try to load from storage
        conv_file = self._storage_path / f"{conversation_id}.json"
        if not conv_file.exists():
            return None
        
        try:
            with open(conv_file, 'r') as f:
                data = json.load(f)
            
            # Recreate conversation
            conversation = Conversation(
                conversation_id=data["id"],
                template=data.get("template"),
                context=data.get("initial_context"),
                user_id=data.get("user_id"),
                voice_engine=voice_engine,
                context_engine=context_engine,
                coordinator=coordinator
            )
            
            # Restore state
            if data.get("started_at"):
                conversation.started_at = datetime.fromisoformat(data["started_at"])
            if data.get("ended_at"):
                conversation.ended_at = datetime.fromisoformat(data["ended_at"])
            conversation.metadata = data.get("metadata", {})
            conversation.summary = data.get("summary")
            
            return conversation
            
        except Exception as e:
            # Log error
            return None
    
    async def save_conversation(self, conversation: Conversation):
        """Save conversation to storage"""
        conv_file = self._storage_path / f"{conversation.id}.json"
        
        try:
            with open(conv_file, 'w') as f:
                json.dump(conversation.to_dict(), f, indent=2)
        except Exception as e:
            # Log error
            pass
    
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        history = self._conversation_history.get(user_id, [])
        
        # Sort by most recent first
        history.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        
        return history[:limit]
    
    async def _cleanup_stale_conversations(self):
        """Remove conversations that have timed out"""
        now = datetime.now()
        timeout_delta = timedelta(minutes=self.timeout_minutes)
        
        stale_ids = []
        for conv_id, conv in self._active_conversations.items():
            if conv.started_at and (now - conv.started_at) > timeout_delta:
                stale_ids.append(conv_id)
        
        for conv_id in stale_ids:
            conv = self._active_conversations[conv_id]
            await conv.end()
            await self.save_conversation(conv)
            del self._active_conversations[conv_id]
    
    async def _load_history(self):
        """Load conversation history from storage"""
        # Simplified - in production would use proper database
        history_file = self._storage_path / "history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self._conversation_history = json.load(f)
            except Exception:
                pass
    
    async def _save_history(self):
        """Save conversation history to storage"""
        history_file = self._storage_path / "history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self._conversation_history, f, indent=2)
        except Exception:
            pass
    
    async def shutdown(self):
        """Shutdown the manager"""
        # Save all active conversations
        for conv in self._active_conversations.values():
            await conv.end()
            await self.save_conversation(conv)
        
        # Save history
        await self._save_history()
        
        self._active_conversations.clear()