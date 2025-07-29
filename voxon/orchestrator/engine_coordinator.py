"""
Engine Coordinator

Manages the interaction between VoxEngine and ContextWeaver.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from voxengine.events import EventType
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority


class EngineCoordinator:
    """
    Coordinates VoxEngine and ContextWeaver.
    
    Handles:
    - Event routing between engines
    - Context injection timing
    - State synchronization
    - Performance optimization
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Engine references
        self.voice_engine = None
        self.context_engine = None
        
        # State
        self.is_initialized = False
        self._event_handlers = {}
        self._injection_task = None
        
        # Metrics
        self.metrics = {
            "events_processed": 0,
            "contexts_injected": 0,
            "injection_failures": 0,
            "avg_injection_delay_ms": 0.0
        }
    
    def set_voice_engine(self, engine):
        """Set VoxEngine reference"""
        self.voice_engine = engine
    
    def set_context_engine(self, engine):
        """Set ContextWeaver reference"""
        self.context_engine = engine
    
    async def initialize(self):
        """Initialize the coordinator"""
        if self.is_initialized:
            return
        
        if not self.voice_engine or not self.context_engine:
            raise ValueError("Both engines must be set before initialization")
        
        # Setup event handlers
        self._setup_event_routing()
        
        # Start context engine
        await self.context_engine.start()
        
        # Start injection monitoring
        self._injection_task = asyncio.create_task(self._monitor_injection())
        
        self.is_initialized = True
        self.logger.info("Engine coordinator initialized")
    
    def _setup_event_routing(self):
        """Setup event routing between engines"""
        
        # Voice engine events that trigger context updates
        event_mappings = {
            EventType.TEXT_OUTPUT: self._handle_text_output,
            EventType.TEXT_INPUT: self._handle_text_input,
            EventType.CONVERSATION_STARTED: self._handle_conversation_start,
            EventType.CONVERSATION_ENDED: self._handle_conversation_end,
            EventType.STATE_CHANGED: self._handle_state_change,
            EventType.AUDIO_INPUT_STARTED: self._handle_audio_start,
            EventType.AUDIO_INPUT_STOPPED: self._handle_audio_stop,
        }
        
        # Subscribe to events
        for event_type, handler in event_mappings.items():
            handler_id = self.voice_engine.events.on(event_type, handler)
            self._event_handlers[event_type] = handler_id
        
        self.logger.debug(f"Registered {len(event_mappings)} event handlers")
    
    async def _monitor_injection(self):
        """Monitor for context injection opportunities"""
        while self.is_initialized:
            try:
                # Check if we should inject context
                if self.voice_engine and self.voice_engine.is_connected:
                    state = self.voice_engine.conversation_state
                    
                    # Check for injection opportunity
                    context_to_inject = await self.context_engine.check_injection(state)
                    
                    if context_to_inject:
                        await self._inject_context(context_to_inject)
                
                # Small delay to prevent busy loop
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Injection monitor error: {e}")
                await asyncio.sleep(1.0)
    
    async def _inject_context(self, context: ContextToInject):
        """Inject context into the conversation"""
        try:
            start_time = datetime.now()
            
            # Format context for injection
            formatted_context = self._format_context_for_injection(context)
            
            # Inject via voice engine
            # This is a simplified version - actual implementation would
            # depend on how VoxEngine handles context updates
            if hasattr(self.voice_engine, 'update_conversation_context'):
                await self.voice_engine.update_conversation_context(formatted_context)
            else:
                # Fallback: inject as a system message
                system_message = self._context_to_system_message(context)
                if system_message:
                    await self.voice_engine.send_text(system_message)
            
            # Update metrics
            injection_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics["contexts_injected"] += 1
            self.metrics["avg_injection_delay_ms"] = (
                (self.metrics["avg_injection_delay_ms"] * (self.metrics["contexts_injected"] - 1) + 
                 injection_time) / self.metrics["contexts_injected"]
            )
            
            self.logger.info(f"Injected context {context.context_id} in {injection_time:.2f}ms")
            
        except Exception as e:
            self.metrics["injection_failures"] += 1
            self.logger.error(f"Failed to inject context: {e}")
    
    def _format_context_for_injection(self, context: ContextToInject) -> Dict[str, Any]:
        """Format context for voice engine consumption"""
        return {
            "context_id": context.context_id,
            "information": context.information,
            "strategy": context.strategy,
            "attention": context.attention,
            "priority": context.priority_value,
            "source": context.source or "context_engine"
        }
    
    def _context_to_system_message(self, context: ContextToInject) -> Optional[str]:
        """Convert context to a system message if needed"""
        # This is a fallback for engines that don't support direct context injection
        parts = []
        
        if context.information:
            parts.append(f"[Context Update: {', '.join(f'{k}={v}' for k, v in context.information.items())}]")
        
        if context.strategy:
            parts.append(f"[Strategy: {', '.join(f'{k}={v}' for k, v in context.strategy.items())}]")
        
        if context.attention:
            parts.append(f"[Focus: {', '.join(f'{k}={v}' for k, v in context.attention.items())}]")
        
        return " ".join(parts) if parts else None
    
    # Event handlers
    
    def _handle_text_output(self, event):
        """Handle text output from AI"""
        self.metrics["events_processed"] += 1
        
        # Extract relevant information for context engine
        if hasattr(event, 'text'):
            # Could analyze text for topics, sentiment, etc.
            # and update context engine accordingly
            pass
    
    def _handle_text_input(self, event):
        """Handle text input from user"""
        self.metrics["events_processed"] += 1
        
        # Analyze user input for context triggers
        if hasattr(event, 'text'):
            # Quick context injection for immediate needs
            self._check_immediate_context_needs(event.text)
    
    def _handle_conversation_start(self, event):
        """Handle conversation start"""
        self.metrics["events_processed"] += 1
        
        # Initialize context for new conversation
        if hasattr(event, 'conversation_id'):
            # Could load user history, preferences, etc.
            pass
    
    def _handle_conversation_end(self, event):
        """Handle conversation end"""
        self.metrics["events_processed"] += 1
        
        # Clean up context
        # Save conversation summary for future context
        pass
    
    def _handle_state_change(self, event):
        """Handle state changes"""
        self.metrics["events_processed"] += 1
        
        # State changes might trigger context updates
        if hasattr(event, 'data'):
            old_status = event.data.get('old_status')
            new_status = event.data.get('new_status')
            
            # Example: inject context when entering processing state
            if new_status == 'processing':
                # Could add processing-specific context
                pass
    
    def _handle_audio_start(self, event):
        """Handle audio input start"""
        self.metrics["events_processed"] += 1
        
        # Prepare context for audio input
        # Could set up real-time processing context
        pass
    
    def _handle_audio_stop(self, event):
        """Handle audio input stop"""
        self.metrics["events_processed"] += 1
        
        # Finalize audio context
        # Could trigger post-processing context injection
        pass
    
    def _check_immediate_context_needs(self, text: str):
        """Check if immediate context injection is needed"""
        # Simple keyword-based immediate context injection
        immediate_triggers = {
            "help": {"attention": {"provide_assistance": True}},
            "emergency": {"strategy": {"urgency": "high", "tone": "serious"}},
            "confused": {"strategy": {"clarity": "high", "pace": "slow"}}
        }
        
        text_lower = text.lower()
        for trigger, context_data in immediate_triggers.items():
            if trigger in text_lower:
                # Create immediate context
                immediate_context = ContextToInject(
                    information=context_data.get("information", {}),
                    strategy=context_data.get("strategy", {}),
                    attention=context_data.get("attention", {}),
                    timing=InjectionTiming.IMMEDIATE,
                    priority=ContextPriority.HIGH,
                    source="immediate_trigger"
                )
                
                # Add to context engine
                self.context_engine.add_context(immediate_context)
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            **self.metrics,
            "is_initialized": self.is_initialized,
            "has_voice_engine": self.voice_engine is not None,
            "has_context_engine": self.context_engine is not None,
            "registered_handlers": len(self._event_handlers)
        }
    
    async def shutdown(self):
        """Shutdown the coordinator"""
        # Cancel injection task
        if self._injection_task:
            self._injection_task.cancel()
            try:
                await self._injection_task
            except asyncio.CancelledError:
                pass
        
        # Unregister event handlers
        if self.voice_engine:
            for handler_id in self._event_handlers.values():
                self.voice_engine.events.off(handler_id)
        
        # Stop context engine
        if self.context_engine:
            await self.context_engine.stop()
        
        self.is_initialized = False
        self.logger.info("Engine coordinator shutdown")