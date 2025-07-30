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
from .vad_adapter import VADModeAdapter
from .response_controller import ResponseController, ResponseMode
from .injection_window import InjectionWindowManager


class EngineCoordinator:
    """
    Coordinates VoxEngine and ContextWeaver.
    
    Handles:
    - Event routing between engines
    - Context injection timing
    - State synchronization
    - Performance optimization
    - VAD mode awareness and response control
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
        
        # VAD and response control
        self.vad_mode = None  # 'client' or 'server'
        self.auto_response_enabled = True
        self.injection_mode = "adaptive"  # 'immediate', 'controlled', 'adaptive'
        self.vad_adapter = VADModeAdapter(logger=self.logger)
        self.response_controller = ResponseController(logger=self.logger)
        self.injection_window_manager = InjectionWindowManager(logger=self.logger)
        
        # Metrics
        self.metrics = {
            "events_processed": 0,
            "contexts_injected": 0,
            "injection_failures": 0,
            "avg_injection_delay_ms": 0.0,
            "vad_mode_switches": 0
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
        
        # Detect VAD mode and response settings
        self._detect_vad_mode()
        
        # Setup event handlers
        self._setup_event_routing()
        
        # Start context engine
        await self.context_engine.start()
        
        # Start injection monitoring
        self._injection_task = asyncio.create_task(self._monitor_injection())
        
        self.is_initialized = True
        self.logger.info(f"Engine coordinator initialized (VAD: {self.vad_mode}, Auto-response: {self.auto_response_enabled})")
    
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
            EventType.RESPONSE_STARTED: self._handle_response_start,
            EventType.RESPONSE_COMPLETED: self._handle_response_complete,
        }
        
        # Subscribe to events
        for event_type, handler in event_mappings.items():
            handler_id = self.voice_engine.events.on(event_type, handler)
            self._event_handlers[event_type] = handler_id
        
        self.logger.debug(f"Registered {len(event_mappings)} event handlers")
    
    def _detect_vad_mode(self):
        """Detect VAD mode and response settings from VoxEngine"""
        try:
            # Get VAD mode from voice engine config
            if hasattr(self.voice_engine, 'config'):
                self.vad_mode = getattr(self.voice_engine.config, 'vad_type', 'client')
            
            # Get session config for response settings
            if hasattr(self.voice_engine, 'session_config'):
                turn_detection = self.voice_engine.session_config.get('turn_detection', {})
                self.auto_response_enabled = turn_detection.get('create_response', True)
            elif hasattr(self.voice_engine, 'get_session_config'):
                session_config = self.voice_engine.get_session_config()
                if session_config:
                    turn_detection = session_config.get('turn_detection', {})
                    self.auto_response_enabled = turn_detection.get('create_response', True)
            
            # Update VAD adapter
            server_silence_ms = None
            if hasattr(self.voice_engine, 'session_config'):
                turn_detection = self.voice_engine.session_config.get('turn_detection', {})
                server_silence_ms = turn_detection.get('silence_duration_ms', 500)
            
            self.vad_adapter.update_mode(self.vad_mode, self.auto_response_enabled, server_silence_ms)
            
            # Determine injection mode based on VAD and response settings
            if self.vad_mode == "server" and self.auto_response_enabled:
                self.injection_mode = "immediate"  # Must inject quickly
                response_mode = ResponseMode.AUTOMATIC
            elif self.vad_mode == "client" or not self.auto_response_enabled:
                self.injection_mode = "controlled"  # Have more control
                response_mode = ResponseMode.MANUAL
            else:
                self.injection_mode = "adaptive"  # Adapt based on situation
                response_mode = ResponseMode.HYBRID
            
            # Configure response controller
            asyncio.create_task(self.response_controller.set_response_mode(
                response_mode, self.auto_response_enabled
            ))
            
            # Configure injection window manager
            self.injection_window_manager.update_vad_mode(self.vad_mode, self.auto_response_enabled)
                
            self.logger.info(f"VAD mode: {self.vad_mode}, Auto-response: {self.auto_response_enabled}, Injection mode: {self.injection_mode}")
            
        except Exception as e:
            self.logger.warning(f"Could not detect VAD mode: {e}. Using defaults.")
            self.vad_mode = "client"
            self.auto_response_enabled = True
            self.injection_mode = "adaptive"
    
    async def _monitor_injection(self):
        """Monitor for context injection opportunities with VAD awareness"""
        while self.is_initialized:
            try:
                # Check if we should inject context
                if self.voice_engine and self.voice_engine.is_connected:
                    state = self.voice_engine.conversation_state
                    
                    # Add VAD and injection mode info to state
                    if hasattr(state, '__dict__'):
                        state.vad_mode = self.vad_mode
                        state.auto_response = self.auto_response_enabled
                        state.injection_mode = self.injection_mode
                    
                    # Check for injection opportunity
                    context_to_inject = await self.context_engine.check_injection(state)
                    
                    if context_to_inject:
                        # Get injection recommendation from window manager
                        recommendation = self.injection_window_manager.get_injection_recommendation()
                        
                        if recommendation.get('recommended', False):
                            # Get injection window from response controller
                            window = await self.response_controller.get_injection_window()
                            
                            # Use the more restrictive deadline
                            deadline_ms = min(
                                window.get('window_ms', 1000),
                                recommendation.get('time_available_ms', 1000)
                            )
                            
                            # Request injection through response controller
                            injection_requested = await self.response_controller.request_injection(
                                context=self._format_context_string(context_to_inject),
                                priority=context_to_inject.priority_value,
                                deadline_ms=deadline_ms
                            )
                            
                            if injection_requested:
                                # Execute injection with response coordination
                                best_window = self.injection_window_manager.get_best_window()
                                if best_window:
                                    await self.response_controller.execute_injection(
                                        inject_callback=lambda ctx: self._inject_context_string(ctx),
                                        trigger_response_callback=self._trigger_response if hasattr(self.voice_engine, 'trigger_response') else None
                                    )
                                    # Mark window as used
                                    self.injection_window_manager.mark_window_used(best_window)
                        else:
                            self.logger.debug(f"Injection not recommended: {recommendation.get('reason')}")
                
                # Adjust monitoring frequency based on VAD adapter recommendation
                interval = self.vad_adapter.get_monitoring_interval()
                await asyncio.sleep(interval)
                
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
    
    async def _inject_context_immediate(self, context: ContextToInject):
        """Inject context immediately (for server VAD with auto-response)"""
        # Must be fast - server will auto-respond soon
        await self._inject_context(context)
    
    async def _inject_context_controlled(self, context: ContextToInject):
        """Inject context with response control"""
        # We can control when to trigger response
        await self._inject_context(context)
        
        # If we need to trigger response manually
        if not self.auto_response_enabled and hasattr(self.voice_engine, 'trigger_response'):
            # Wait a bit for context to be processed
            await asyncio.sleep(0.1)
            await self.voice_engine.trigger_response()
    
    async def _inject_context_adaptive(self, context: ContextToInject):
        """Adaptively inject context based on situation"""
        # Check current conversation state
        if hasattr(self.voice_engine, 'is_processing'):
            if self.voice_engine.is_processing:
                # Wait for processing to complete
                await asyncio.sleep(0.05)
        
        await self._inject_context(context)
    
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
    
    def _format_context_string(self, context: ContextToInject) -> str:
        """Format context as string for response controller"""
        parts = []
        
        if context.information:
            info_str = ", ".join(f"{k}: {v}" for k, v in context.information.items())
            parts.append(f"[Context: {info_str}]")
        
        if context.strategy:
            strategy_str = ", ".join(f"{k}: {v}" for k, v in context.strategy.items())
            parts.append(f"[Strategy: {strategy_str}]")
        
        if context.attention:
            attention_str = ", ".join(f"{k}: {v}" for k, v in context.attention.items())
            parts.append(f"[Focus: {attention_str}]")
        
        return " ".join(parts)
    
    async def _inject_context_string(self, context_str: str):
        """Inject context string into conversation"""
        # This method is called by response controller
        if hasattr(self.voice_engine, 'send_text'):
            await self.voice_engine.send_text(context_str)
        else:
            self.logger.warning("Voice engine does not support text injection")
    
    async def _trigger_response(self):
        """Trigger response in voice engine"""
        if hasattr(self.voice_engine, 'trigger_response'):
            await self.voice_engine.trigger_response()
        else:
            self.logger.warning("Voice engine does not support manual response triggering")
    
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
            
            # Check for VAD mode changes
            if 'vad_type' in event.data:
                old_vad = self.vad_mode
                self._detect_vad_mode()
                if old_vad != self.vad_mode:
                    self.metrics["vad_mode_switches"] += 1
                    self.logger.info(f"VAD mode changed: {old_vad} → {self.vad_mode}")
            
            # Example: inject context when entering processing state
            if new_status == 'processing':
                # Could add processing-specific context
                pass
    
    def _handle_audio_start(self, event):
        """Handle audio input start"""
        self.metrics["events_processed"] += 1
        
        # Notify injection window manager
        self.injection_window_manager.on_user_input_start()
        
        # Prepare context for audio input
        # Could set up real-time processing context
        pass
    
    def _handle_audio_stop(self, event):
        """Handle audio input stop"""
        self.metrics["events_processed"] += 1
        
        # Notify injection window manager
        self.injection_window_manager.on_user_input_end()
        
        # Finalize audio context
        # Could trigger post-processing context injection
        pass
    
    def _handle_response_start(self, event):
        """Handle AI response start"""
        self.metrics["events_processed"] += 1
        
        # Notify injection window manager
        self.injection_window_manager.on_ai_response_start()
        
        # Mark response as pending in controller
        self.response_controller.mark_response_pending()
    
    def _handle_response_complete(self, event):
        """Handle AI response complete"""
        self.metrics["events_processed"] += 1
        
        # Notify injection window manager
        self.injection_window_manager.on_ai_response_end()
        
        # Mark response as ready in controller
        self.response_controller.mark_response_ready()
    
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
        response_metrics = self.response_controller.get_metrics() if self.response_controller else {}
        window_metrics = self.injection_window_manager.get_metrics() if self.injection_window_manager else {}
        
        return {
            **self.metrics,
            "is_initialized": self.is_initialized,
            "has_voice_engine": self.voice_engine is not None,
            "has_context_engine": self.context_engine is not None,
            "registered_handlers": len(self._event_handlers),
            "vad_mode": self.vad_mode,
            "auto_response_enabled": self.auto_response_enabled,
            "injection_mode": self.injection_mode,
            "response_controller": response_metrics,
            "injection_windows": window_metrics
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