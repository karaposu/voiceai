"""
Unified Voice Engine

Main entry point for the realtime voice API framework.
Provides a clean, simple API for voice interactions.

This module focuses on the public API, delegating implementation
details to BaseEngine.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, Literal, Union, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
import json

from .core.stream_protocol import StreamEvent, StreamEventType, StreamState
AudioBytes = bytes  # Simple type alias for audio data
from .core.provider_protocol import Usage, Cost
from .core.exceptions import EngineError
from .strategies.base_strategy import EngineConfig
import time

# Import the base engine that handles implementation
from .base_engine import BaseEngine

# Import event system
from .events import (
    EventEmitter, EventType, Event,
    AudioEvent, TextEvent, ConnectionEvent,
    ConversationEvent, FunctionCallEvent,
    ErrorEvent, TimingEvent
)


@dataclass
class VoiceEngineConfig:
    """Configuration for Voice Engine"""
    
    # API Configuration
    api_key: str
    provider: str = "openai"
    
    # Mode selection
    mode: Literal["fast", "big", "provider"] = "fast"
    
    # Audio settings
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = 24000
    chunk_duration_ms: int = 100
    
    # Features
    vad_enabled: bool = True
    vad_type: Literal["client", "server"] = "client"
    vad_threshold: float = 0.02
    vad_speech_start_ms: int = 100
    vad_speech_end_ms: int = 500
    
    # Voice settings
    voice: str = "alloy"
    language: Optional[str] = None
    
    # Performance
    latency_mode: Literal["ultra_low", "balanced", "quality"] = "balanced"
    
    # Features (for big lane)
    enable_transcription: bool = False
    enable_functions: bool = False
    enable_multi_provider: bool = False
    
    # Advanced
    log_level: str = "INFO"
    save_audio: bool = False
    audio_save_path: Optional[Path] = None
    
    # Additional provider-specific config
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_engine_config(self) -> EngineConfig:
        """Convert to strategy engine config"""
        return EngineConfig(
            api_key=self.api_key,
            provider=self.provider,
            input_device=self.input_device,
            output_device=self.output_device,
            enable_vad=self.vad_enabled,
            enable_transcription=self.enable_transcription,
            enable_functions=self.enable_functions,
            latency_mode=self.latency_mode,
            metadata={
                **self.metadata,
                "voice": self.voice,
                "language": self.language,
                "vad_type": self.vad_type,
                "vad_threshold": self.vad_threshold,
                "vad_speech_start_ms": self.vad_speech_start_ms,
                "vad_speech_end_ms": self.vad_speech_end_ms,
                "sample_rate": self.sample_rate,
                "chunk_duration_ms": self.chunk_duration_ms,
                "save_audio": self.save_audio,
                "audio_save_path": str(self.audio_save_path) if self.audio_save_path else None
            }
        )


class VoiceEngine:
    """
    Unified Voice Engine - Main entry point for realtime voice API.
    
    Simple, clean API for voice interactions with AI.
    
    Fast Lane (mode="fast"):
        - Ultra-low latency (<50ms)
        - Direct audio path
        - Best for real-time conversations
        
    Big Lane (mode="big"):
        - Advanced features
        - Audio effects and processing
        - Best for complex applications
    
    Example:
        ```python
        # Simple usage with events
        engine = VoiceEngine(api_key="...", mode="fast")
        await engine.connect()
        
        # Option 1: Use event system (recommended)
        engine.events.on(EventType.AUDIO_OUTPUT_CHUNK, 
                        lambda event: play(event.audio_data))
        engine.events.on(EventType.TEXT_OUTPUT,
                        lambda event: print(f"AI: {event.text}"))
        
        # Option 2: Use callbacks (simpler for basic use)
        engine.on_audio_response = lambda audio: play(audio)
        engine.on_text_response = lambda text: print(f"AI: {text}")
        
        # Start conversation
        await engine.start_listening()
        ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[VoiceEngineConfig] = None,
        mode: Literal["fast", "big", "provider"] = "fast",
        **kwargs
    ):
        """
        Initialize Voice Engine.
        
        Args:
            api_key: API key for the provider
            config: Full configuration object
            mode: Choose "fast" or "big" lane
            **kwargs: Additional config parameters
        """
        # Handle configuration
        if config:
            self.config = config
        else:
            if not api_key:
                raise ValueError("API key required")
            
            self.config = VoiceEngineConfig(
                api_key=api_key,
                mode=mode,
                **kwargs
            )
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Store mode for quick access
        self.mode = self.config.mode
        self.name = f"VoiceEngine-{self.mode}"
        self.logger.info(f"Voice Engine initialized in {self.mode} mode")
        
        # Create base engine that handles all implementation
        self._base = BaseEngine(logger=self.logger)
        
        # Create strategy through base engine
        # Use provider-based fast lane if metadata indicates it
        use_provider_fast_lane = self.config.metadata.get("use_provider_fast_lane", False)
        self._base.create_strategy(self.mode, use_provider_fast_lane=use_provider_fast_lane)
        
        # Direct reference to strategy for hot path
        self._strategy = self._base.strategy
        
        # Event emitter
        self.events = EventEmitter(name=f"VoiceEngine-{self.mode}")
        
        # State management
        from voxon.state import StateManager, ConversationState
        self._state_manager = StateManager(
            initial_state=ConversationState(),
            event_emitter=self.events,
            logger=self.logger
        )
        
        # Legacy callbacks (for backward compatibility and convenience)
        self.on_audio_response: Optional[Callable[[AudioBytes], None]] = None
        self.on_text_response: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_function_call: Optional[Callable[[Dict[str, Any]], Any]] = None
        self.on_response_done: Optional[Callable[[], None]] = None
    
    # ============== Properties ==============
    
    @property
    def is_connected(self) -> bool:
        """Check if engine is connected"""
        return self._base.is_connected
    
    @property
    def is_listening(self) -> bool:
        """Check if engine is listening"""
        return self._base.is_listening
    
    def get_state(self) -> StreamState:
        """Get current stream state"""
        return self._base.get_state()
    
    @property
    def conversation_state(self) -> 'ConversationState':
        """Get current conversation state"""
        return self._state_manager.state
    
    @property
    def state_manager(self) -> 'StateManager':
        """Get state manager for advanced state operations"""
        return self._state_manager
    
    # ============== Connection Methods ==============


    # In voice_engine.py, update the connect method:

    async def connect(self, retry_count: int = 3) -> None:
        """
        Connect to the voice API.
        
        Args:
            retry_count: Number of connection attempts
            
        Raises:
            EngineError: If connection fails
        """
        if self.is_connected:
            self.logger.warning("Already connected")
            return
        
        last_error = None
        
        for attempt in range(retry_count):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt + 1}/{retry_count}")
                    await asyncio.sleep(1.0 * attempt)
                
                # Recreate strategy if needed (for reconnection)
                if not self._base.strategy or not hasattr(self._base.strategy, '_is_initialized'):
                    use_provider_fast_lane = self.config.metadata.get("use_provider_fast_lane", False)
                    self._base.create_strategy(self.mode, use_provider_fast_lane=use_provider_fast_lane)
                    self._strategy = self._base.strategy
                
                # Initialize strategy
                await self._base.initialize_strategy(self.config.to_engine_config())
                
                # Setup audio for fast lane
                if self.mode == "fast":
                    await self._base.setup_fast_lane_audio(
                        sample_rate=self.config.sample_rate,
                        chunk_duration_ms=self.config.chunk_duration_ms,
                        input_device=self.config.input_device,
                        output_device=self.config.output_device,
                        vad_enabled=self.config.vad_enabled,
                        vad_threshold=self.config.vad_threshold,
                        vad_speech_start_ms=self.config.vad_speech_start_ms,
                        vad_speech_end_ms=self.config.vad_speech_end_ms
                    )
                
                # Connect
                await self._base.do_connect()
                
                # Setup event handlers
                self._setup_event_handlers()
                
                # Start event processor now that we have a loop
                self.events._start_processor()
                
                # Emit connection established event
                await self.events.emit(ConnectionEvent(
                    type=EventType.CONNECTION_ESTABLISHED,
                    connection_id=getattr(self._strategy, 'session_id', None),
                    source=self.name
                ))
                
                return
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                # Emit connection failed event
                await self.events.emit(ConnectionEvent(
                    type=EventType.CONNECTION_FAILED,
                    error=e,
                    retry_count=attempt + 1,
                    source=self.name
                ))
                
                # IMPORTANT: Cleanup on failure to prevent segmentation fault
                try:
                    await self._base.cleanup()
                except Exception as cleanup_error:
                    self.logger.error(f"Cleanup error after failed connection: {cleanup_error}")
        
        # Final cleanup before raising error
        try:
            await self._base.cleanup()
        except Exception as cleanup_error:
            self.logger.error(f"Final cleanup error: {cleanup_error}")
        
        error = EngineError(f"Failed to connect after {retry_count} attempts")
        
        # Emit final connection error
        await self.events.emit(ErrorEvent(
            type=EventType.ERROR_NETWORK,
            error=error,
            error_message=str(error),
            recoverable=False,
            source=self.name
        ))
        
        raise error from last_error

        
    
    
    
    

    async def disconnect(self) -> None:
        """Disconnect from voice API"""
        try:
            # Update state
            from voxon.state import ConversationStatus
            self._state_manager.update_connection(is_connected=False)
            self._state_manager.update(status=ConversationStatus.DISCONNECTED)
            
            # Emit disconnection event
            await self.events.emit(ConnectionEvent(
                type=EventType.CONNECTION_CLOSED,
                source=self.name
            ))
            
            await self._base.cleanup()
            await self.events.close()
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
            await self.events.emit(ErrorEvent(
                type=EventType.ERROR_GENERAL,
                error=e,
                error_message=f"Disconnect error: {e}",
                source=self.name
            ))
    
    # ============== Audio Control ==============
    
    async def start_listening(self) -> None:
        """Start listening for audio input"""
        self._ensure_connected()
        await self._base.start_audio_processing()
        
        # Update state
        self._state_manager.update_audio(is_listening=True)
        
        await self.events.emit(Event(
            type=EventType.AUDIO_INPUT_STARTED,
            source=self.name
        ))
    
    async def stop_listening(self) -> None:
        """Stop listening for audio input"""
        await self._base.stop_audio_processing()
        
        # Update state
        self._state_manager.update_audio(is_listening=False)
        
        await self.events.emit(Event(
            type=EventType.AUDIO_INPUT_STOPPED,
            source=self.name
        ))
    
    # ============== Communication Methods ==============
    
    async def send_audio(self, audio_data: AudioBytes) -> None:
        """
        Send audio data to the AI for real-time streaming.
        
        Args:
            audio_data: Raw audio bytes
        """
        self._ensure_connected()
        # Direct call to strategy for zero overhead
        await self._strategy.send_audio(audio_data)
    
    async def send_recorded_audio(self, audio_data: AudioBytes, 
                                auto_respond: bool = True) -> None:
        """
        Send complete audio recording and optionally trigger response.
        
        Use this when you have a complete audio recording (e.g., from push-to-talk,
        file upload, or pre-recorded audio) rather than real-time streaming.
        
        Args:
            audio_data: Complete audio recording
            auto_respond: Whether to automatically trigger AI response (default: True)
        """
        self._ensure_connected()
        # Check if strategy supports this method
        if hasattr(self._strategy, 'send_recorded_audio'):
            await self._strategy.send_recorded_audio(audio_data, auto_respond)
        else:
            # Fallback: send as regular audio and manually trigger response
            await self._strategy.send_audio(audio_data)
            if auto_respond and hasattr(self._strategy, 'trigger_response'):
                await self._strategy.trigger_response()
    
    async def send_text(self, text: str) -> None:
        """
        Send text message to the AI.
        
        Args:
            text: Text message
        """
        self._ensure_connected()
        
        # Add message to state
        from voxon.state import Message, SpeakerRole
        message = Message(
            role=SpeakerRole.USER,
            content=text
        )
        self._state_manager.add_message(message)
        
        # Emit text input event
        await self.events.emit(TextEvent(
            type=EventType.TEXT_INPUT,
            text=text,
            source=self.name
        ))
        
        # Direct call to strategy for zero overhead
        await self._strategy.send_text(text)
        self.logger.debug(f"Sent text: {text}")
    
    async def interrupt(self) -> None:
        """Interrupt the current AI response"""
        self._ensure_connected()
        
        # Emit conversation interrupted event
        await self.events.emit(ConversationEvent(
            type=EventType.CONVERSATION_INTERRUPTED,
            source=self.name
        ))
        
        await self._strategy.interrupt()
    
    # ============== Convenience Methods ==============


    async def text_2_audio_response(self, text: str, timeout: float = 30.0) -> AudioBytes:
        """
        Convert text to speech with real-time playback.
        """
        self._ensure_connected()
        
        audio_future: asyncio.Future[bytes] = asyncio.Future()
        audio_chunks: List[bytes] = []
        first_chunk_played = False
        
        # Handler IDs for cleanup
        audio_handler_id = None
        done_handler_id = None
        
        def collect_audio(event: AudioEvent):
            nonlocal first_chunk_played
            if event.audio_data:
                audio_chunks.append(event.audio_data)
                
                # Play immediately for real-time experience
                if self._base.components.audio_engine:
                    if not first_chunk_played:
                        # Tiny delay only for first chunk to ensure stream is ready
                        time.sleep(0.005)  # 5ms - imperceptible
                        first_chunk_played = True
                    self._base.components.audio_engine.play_audio(event.audio_data)
        
        def on_done(event: Event):
            if not audio_future.done():
                if audio_chunks:
                    audio_future.set_result(b"".join(audio_chunks))
                else:
                    audio_future.set_exception(EngineError("No audio received"))
        
        try:
            # Register temporary handlers
            audio_handler_id = self.events.on(EventType.AUDIO_OUTPUT_CHUNK, collect_audio)
            done_handler_id = self.events.on(EventType.RESPONSE_COMPLETED, on_done)
            
            # Send text
            await self.send_text(text)
            
            # Wait for response
            return await asyncio.wait_for(audio_future, timeout=timeout)
            
        except asyncio.TimeoutError:
            raise EngineError(f"Response timeout after {timeout} seconds")
        finally:
            # Remove handlers
            if audio_handler_id:
                self.events.off(audio_handler_id)
            if done_handler_id:
                self.events.off(done_handler_id)
    
    

            
    
    async def transcribe_audio_file(self, file_path: Union[str, Path]) -> str:
        """
        Transcribe an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcribed text
            
        Note:
            Requires big lane mode (not yet implemented)
        """
        if self.mode != "big":
            raise EngineError("File transcription requires big lane mode")
        raise NotImplementedError("File transcription not yet implemented")
    
    # ============== Metrics and Usage ==============
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self._base.get_metrics()
        metrics["mode"] = self.mode
        return metrics
    
    async def get_usage(self) -> Usage:
        """Get usage statistics"""
        return await self._base.get_usage()
    
    async def estimate_cost(self) -> Cost:
        """Estimate session cost"""
        return await self._base.estimate_cost()
    
    # ============== Advanced API ==============
    
    async def event_stream(self) -> AsyncIterator[StreamEvent]:
        """
        Stream all events for advanced usage.
        
        Yields:
            StreamEvent objects as they occur
        """
        self._ensure_connected()
        
        event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        
        def queue_event(event: StreamEvent):
            try:
                event_queue.put_nowait(event)
            except asyncio.QueueFull:
                self.logger.warning("Event queue full, dropping event")
        
        # Subscribe to all events
        for event_type in StreamEventType:
            self._base.set_event_handler(event_type, queue_event)
        
        try:
            while self.is_connected:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    yield event
                except asyncio.TimeoutError:
                    continue
        finally:
            # Clear handlers
            self._base._event_handlers.clear()
    
    # ============== Context Manager ==============
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    # ============== Factory Methods ==============
    
    @classmethod
    def create_simple(cls, api_key: str, voice: str = "alloy") -> 'VoiceEngine':
        """
        Create a simple voice engine with fast lane.
        
        Args:
            api_key: API key
            voice: Voice to use
            
        Returns:
            Configured VoiceEngine instance
        """
        config = VoiceEngineConfig(
            api_key=api_key,
            voice=voice,
            mode="fast",
            latency_mode="ultra_low"
        )
        return cls(config=config)
    
    @classmethod
    def create_advanced(
        cls,
        api_key: str,
        enable_transcription: bool = True,
        enable_functions: bool = True,
        **kwargs
    ) -> 'VoiceEngine':
        """
        Create an advanced voice engine with big lane.
        
        Args:
            api_key: API key
            enable_transcription: Enable transcription features
            enable_functions: Enable function calling
            **kwargs: Additional configuration
            
        Returns:
            Configured VoiceEngine instance
            
        Note:
            Big lane is not yet implemented
        """
        config = VoiceEngineConfig(
            api_key=api_key,
            mode="big",
            enable_transcription=enable_transcription,
            enable_functions=enable_functions,
            latency_mode="balanced",
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'VoiceEngine':
        """
        Create voice engine from JSON config file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configured VoiceEngine instance
        """
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = VoiceEngineConfig(**config_data)
        return cls(config=config)
    
    # ============== Private Methods ==============
    
    def _ensure_connected(self):
        """Ensure engine is connected"""
        if not self.is_connected:
            raise EngineError("Not connected. Call connect() first.")
    
    def _setup_event_handlers(self):
        """Setup event handlers"""
        # Create handler mapping
        handlers = {
            StreamEventType.AUDIO_OUTPUT_CHUNK: self._handle_audio_output,
            StreamEventType.TEXT_OUTPUT_CHUNK: self._handle_text_output,
            StreamEventType.STREAM_ERROR: self._handle_error,
            StreamEventType.STREAM_ENDED: self._handle_stream_ended,
        }
        
        # Add function call handler if available
        if hasattr(StreamEventType, 'FUNCTION_CALL'):
            handlers[StreamEventType.FUNCTION_CALL] = self._handle_function_call
        
        # Pass to base engine
        self._base.setup_event_handlers(handlers)
    
    def _handle_audio_output(self, event: StreamEvent):
        """Handle audio output"""
        if event.data:
            audio_data = event.data.get("audio")
            if audio_data:
                # Emit audio event (sync safe)
                self.events.emit_sync(AudioEvent(
                    type=EventType.AUDIO_OUTPUT_CHUNK,
                    audio_data=audio_data,
                    source=self.name,
                    metadata=event.metadata or {}
                ))
                
                # Call legacy callback if set
                if self.on_audio_response:
                    self.on_audio_response(audio_data)
                
                # Auto-play if available
                self._base.play_audio(audio_data)
    
    def _handle_text_output(self, event: StreamEvent):
        """Handle text output"""
        if event.data:
            text = event.data.get("text")
            if text:
                # Add assistant message to state (only if not partial)
                if not event.data.get("is_partial", False):
                    from voxon.state import Message, SpeakerRole
                    message = Message(
                        role=SpeakerRole.ASSISTANT,
                        content=text
                    )
                    self._state_manager.add_message(message)
                
                # Emit text event (sync safe)
                self.events.emit_sync(TextEvent(
                    type=EventType.TEXT_OUTPUT,
                    text=text,
                    is_partial=event.data.get("is_partial", False),
                    source=self.name,
                    metadata=event.metadata or {}
                ))
                
                # Call legacy callback if set
                if self.on_text_response:
                    self.on_text_response(text)
    
    def _handle_error(self, event: StreamEvent):
        """Handle errors"""
        if event.data:
            error_msg = event.data.get("error", "Unknown error")
            error = EngineError(str(error_msg))
            
            # Emit error event (sync safe)
            self.events.emit_sync(ErrorEvent(
                type=EventType.ERROR_GENERAL,
                error=error,
                error_message=str(error_msg),
                source=self.name,
                metadata=event.metadata or {}
            ))
            
            # Call legacy callback if set
            if self.on_error:
                self.on_error(error)
    
    def _handle_function_call(self, event: StreamEvent):
        """Handle function calls"""
        if event.data:
            function_data = event.data.get("function")
            if function_data:
                # Call legacy callback first to get result if set
                result = None
                if self.on_function_call:
                    result = self.on_function_call(function_data)
                
                # Emit function call event with result (sync safe)
                self.events.emit_sync(FunctionCallEvent(
                    type=EventType.FUNCTION_CALL_INVOKED,
                    function_name=function_data.get("name", ""),
                    arguments=function_data.get("arguments", {}),
                    call_id=function_data.get("call_id"),
                    result=result,
                    source=self.name,
                    metadata=event.metadata or {}
                ))
    
    def _handle_stream_ended(self, event: StreamEvent):
        """Handle stream ended"""
        # Emit response completed event (sync safe)
        self.events.emit_sync(Event(
            type=EventType.RESPONSE_COMPLETED,
            source=self.name,
            metadata=event.metadata or {}
        ))
        
        # Call legacy callback if set
        if self.on_response_done:
            self.on_response_done()


# ============== Convenience Functions ==============

async def create_voice_session(
    api_key: str,
    mode: Literal["fast", "big"] = "fast",
    **kwargs
) -> VoiceEngine:
    """
    Create and connect a voice session.
    
    Quick setup for simple use cases.
    
    Args:
        api_key: API key
        mode: Engine mode
        **kwargs: Additional configuration
        
    Returns:
        Connected VoiceEngine instance
    """
    engine = VoiceEngine(api_key=api_key, mode=mode, **kwargs)
    await engine.connect()
    return engine


def run_voice_engine(
    api_key: str,
    mode: Literal["fast", "big"] = "fast",
    on_audio: Optional[Callable[[AudioBytes], None]] = None,
    on_text: Optional[Callable[[str], None]] = None,
    **kwargs
):
    """
    Run voice engine in a simple event loop.
    
    Good for testing and simple applications.
    
    Args:
        api_key: API key
        mode: Engine mode
        on_audio: Audio response callback
        on_text: Text response callback
        **kwargs: Additional configuration
    """
    async def main():
        engine = VoiceEngine(api_key=api_key, mode=mode, **kwargs)
        
        # Set callbacks (simpler API for basic use)
        if on_audio:
            engine.on_audio_response = on_audio
        if on_text:
            engine.on_text_response = on_text
        
        # Connect and start
        await engine.connect()
        await engine.start_listening()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await engine.disconnect()
    
    asyncio.run(main())