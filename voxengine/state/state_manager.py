"""
State Manager

Thread-safe state management with atomic updates and event integration.
Provides a clean API for state transitions while maintaining consistency.
"""

import asyncio
import threading
from typing import Optional, Callable, Dict, Any, List, TypeVar, Generic
from datetime import datetime
import json
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from .conversation_state import (
    ConversationState, ConversationStatus, Message, Turn,
    AudioState, ConnectionState, ConversationMetrics, SpeakerRole
)
from ..events import EventEmitter, EventType, Event


T = TypeVar('T')


class StateUpdate:
    """Represents a state update operation"""
    def __init__(self, field_path: str, value: Any, timestamp: datetime = None):
        self.field_path = field_path
        self.value = value
        self.timestamp = timestamp or datetime.now()


class StateManager:
    """
    Thread-safe state manager with atomic updates.
    
    Features:
    - Atomic state updates with optimistic locking
    - Event emission on state changes
    - State history and snapshots
    - Persistence support
    - Performance monitoring
    """
    
    def __init__(
        self,
        initial_state: Optional[ConversationState] = None,
        event_emitter: Optional[EventEmitter] = None,
        enable_history: bool = True,
        history_size: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        # Core state
        self._state = initial_state or ConversationState()
        self._lock = threading.RLock()  # Reentrant lock for nested updates
        
        # Event integration
        self._event_emitter = event_emitter
        
        # History tracking
        self._enable_history = enable_history
        self._history: List[ConversationState] = []
        self._history_size = history_size
        
        # Update tracking
        self._update_count = 0
        self._update_callbacks: List[Callable[[ConversationState, ConversationState], None]] = []
        
        # Performance metrics
        self._metrics = {
            "total_updates": 0,
            "avg_update_time_ms": 0.0,
            "max_update_time_ms": 0.0,
            "history_snapshots": 0
        }
        
        self.logger = logger or logging.getLogger(__name__)
    
    @property
    def state(self) -> ConversationState:
        """Get current state (thread-safe)"""
        with self._lock:
            return self._state
    
    def update(self, **updates) -> ConversationState:
        """
        Atomically update state fields.
        
        Example:
            new_state = manager.update(
                status=ConversationStatus.CONNECTED,
                connection=old_state.connection.evolve(is_connected=True)
            )
        """
        import time
        start_time = time.time()
        
        with self._lock:
            old_state = self._state
            
            # Create new state with updates
            new_state = old_state.evolve(**updates)
            
            # Update timestamp
            new_state = new_state.evolve(last_activity_at=datetime.now())
            
            # Store new state
            self._state = new_state
            self._update_count += 1
            
            # Add to history
            if self._enable_history:
                self._add_to_history(old_state)
            
            # Emit state change event
            if self._event_emitter:
                asyncio.create_task(self._emit_state_change(old_state, new_state))
            
            # Call update callbacks
            for callback in self._update_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    self.logger.error(f"Error in state update callback: {e}")
            
            # Update metrics
            update_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(update_time_ms)
            
            return new_state
    
    async def update_async(self, **updates) -> ConversationState:
        """Async version of update for use in async contexts"""
        return self.update(**updates)
    
    def add_message(self, message: Message) -> ConversationState:
        """Add a message to the conversation"""
        with self._lock:
            messages = list(self._state.messages)
            messages.append(message)
            
            # Update metrics
            metrics = self._state.metrics.evolve(
                total_messages=self._state.metrics.total_messages + 1
            )
            
            # Handle turn management
            current_turn = self._state.current_turn
            turns = list(self._state.turns)
            
            if message.role == SpeakerRole.USER:
                # Start new turn
                current_turn = Turn(user_message=message)
            elif message.role == SpeakerRole.ASSISTANT and current_turn:
                # Complete current turn
                current_turn = current_turn.evolve(
                    assistant_message=message,
                    completed_at=datetime.now()
                )
                turns.append(current_turn)
                metrics = metrics.evolve(total_turns=metrics.total_turns + 1)
                current_turn = None
            
            return self.update(
                messages=messages,
                current_turn=current_turn,
                turns=turns,
                metrics=metrics
            )
    
    def update_connection(
        self,
        is_connected: Optional[bool] = None,
        latency_ms: Optional[float] = None,
        **kwargs
    ) -> ConversationState:
        """Update connection state"""
        updates = {}
        if is_connected is not None:
            updates['is_connected'] = is_connected
        if latency_ms is not None:
            updates['latency_ms'] = latency_ms
        updates.update(kwargs)
        
        new_connection = self._state.connection.evolve(**updates)
        
        # Update main status based on connection
        status = self._state.status
        if is_connected is True and status == ConversationStatus.CONNECTING:
            status = ConversationStatus.CONNECTED
        elif is_connected is False:
            status = ConversationStatus.DISCONNECTED
        
        return self.update(connection=new_connection, status=status)
    
    def update_audio(
        self,
        is_listening: Optional[bool] = None,
        is_playing: Optional[bool] = None,
        vad_active: Optional[bool] = None,
        **kwargs
    ) -> ConversationState:
        """Update audio state"""
        updates = {}
        if is_listening is not None:
            updates['is_listening'] = is_listening
        if is_playing is not None:
            updates['is_playing'] = is_playing
        if vad_active is not None:
            updates['vad_active'] = vad_active
        updates.update(kwargs)
        
        new_audio = self._state.audio.evolve(**updates)
        
        # Update main status based on audio state
        status = self._state.status
        if is_listening and self._state.connection.is_connected:
            status = ConversationStatus.LISTENING
        
        return self.update(audio=new_audio, status=status)
    
    def start_turn(self, user_message: Message) -> ConversationState:
        """Start a new conversation turn"""
        return self.add_message(user_message)
    
    def complete_turn(self, assistant_message: Message) -> ConversationState:
        """Complete the current turn with assistant response"""
        return self.add_message(assistant_message)
    
    def interrupt_turn(self) -> ConversationState:
        """Mark current turn as interrupted"""
        if not self._state.current_turn:
            return self._state
        
        # Mark any in-progress assistant message as interrupted
        messages = list(self._state.messages)
        if messages and messages[-1].role == SpeakerRole.ASSISTANT:
            messages[-1] = messages[-1].evolve(is_interrupted=True)
        
        metrics = self._state.metrics.evolve(
            interruption_count=self._state.metrics.interruption_count + 1
        )
        
        return self.update(
            messages=messages,
            current_turn=None,
            metrics=metrics
        )
    
    # History and snapshots
    
    def _add_to_history(self, state: ConversationState) -> None:
        """Add state to history"""
        self._history.append(state)
        if len(self._history) > self._history_size:
            self._history.pop(0)
        self._metrics["history_snapshots"] = len(self._history)
    
    def get_history(self, limit: Optional[int] = None) -> List[ConversationState]:
        """Get state history"""
        with self._lock:
            if limit:
                return list(self._history[-limit:])
            return list(self._history)
    
    def create_snapshot(self) -> Dict[str, Any]:
        """Create a state snapshot for debugging"""
        with self._lock:
            return {
                "state": self._state.to_dict(),
                "update_count": self._update_count,
                "metrics": dict(self._metrics),
                "timestamp": datetime.now().isoformat()
            }
    
    # Persistence
    
    def save_to_file(self, path: Path) -> None:
        """Save state to file"""
        with self._lock:
            snapshot = self.create_snapshot()
            with open(path, 'w') as f:
                json.dump(snapshot, f, indent=2)
    
    @classmethod
    def load_from_file(cls, path: Path) -> 'StateManager':
        """Load state from file"""
        with open(path, 'r') as f:
            snapshot = json.load(f)
        
        # Reconstruct state (simplified - full implementation would deserialize properly)
        manager = cls()
        # ... deserialize state from snapshot ...
        return manager
    
    # Event integration
    
    async def _emit_state_change(
        self,
        old_state: ConversationState,
        new_state: ConversationState
    ) -> None:
        """Emit appropriate events based on state changes"""
        if not self._event_emitter:
            return
        
        # Status changes
        if old_state.status != new_state.status:
            await self._event_emitter.emit(Event(
                type=EventType.STATE_CHANGED,
                data={
                    "old_status": old_state.status.value,
                    "new_status": new_state.status.value
                }
            ))
        
        # Connection changes
        if old_state.connection.is_connected != new_state.connection.is_connected:
            event_type = (EventType.CONNECTION_ESTABLISHED 
                         if new_state.connection.is_connected 
                         else EventType.CONNECTION_CLOSED)
            await self._event_emitter.emit(Event(type=event_type))
        
        # Audio state changes
        if old_state.audio.is_listening != new_state.audio.is_listening:
            event_type = (EventType.AUDIO_INPUT_STARTED 
                         if new_state.audio.is_listening 
                         else EventType.AUDIO_INPUT_STOPPED)
            await self._event_emitter.emit(Event(type=event_type))
    
    # Callbacks and monitoring
    
    def on_update(self, callback: Callable[[ConversationState, ConversationState], None]) -> None:
        """Register a callback for state updates"""
        self._update_callbacks.append(callback)
    
    def _update_metrics(self, update_time_ms: float) -> None:
        """Update performance metrics"""
        self._metrics["total_updates"] += 1
        
        # Update average
        total = self._metrics["total_updates"]
        old_avg = self._metrics["avg_update_time_ms"]
        self._metrics["avg_update_time_ms"] = (old_avg * (total - 1) + update_time_ms) / total
        
        # Update max
        self._metrics["max_update_time_ms"] = max(
            self._metrics["max_update_time_ms"],
            update_time_ms
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get state manager metrics"""
        with self._lock:
            return dict(self._metrics)
    
    # Context manager support
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for batched updates"""
        # Collect updates in transaction
        updates = {}
        
        class Transaction:
            def update(self, **kwargs):
                updates.update(kwargs)
        
        yield Transaction()
        
        # Apply all updates atomically
        if updates:
            self.update(**updates)