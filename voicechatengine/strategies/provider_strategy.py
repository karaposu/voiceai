"""
Provider-based Strategy

A new strategy that uses the provider abstraction for voice interactions.
This demonstrates how to integrate the new provider system with VoiceEngine.
"""

import asyncio
import time
from typing import Optional, Dict, Any, AsyncIterator, Callable, List
import logging

from .base_strategy import BaseStrategy, EngineConfig
from ..core.stream_protocol import StreamEvent, StreamEventType, StreamState
from ..core.provider_protocol import Usage, Cost, ProviderConfig as CoreProviderConfig
from ..providers import get_registry, BaseProvider, ProviderEvent
from ..providers.provider_adapter import ProviderSessionAdapter
from voxstream import VoxStream
from voxstream.config.types import StreamConfig as AudioConfig, VADConfig, ProcessingMode, VADType


class ProviderStrategy(BaseStrategy):
    """
    Strategy that uses the new provider abstraction.
    
    This strategy demonstrates how to use the BaseProvider interface
    for voice interactions instead of direct WebSocket connections.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config: Optional[EngineConfig] = None
        self._provider: Optional[BaseProvider] = None
        self._session: Optional[ProviderSessionAdapter] = None
        self._audio_engine: Optional[VoxStream] = None
        self._event_task: Optional[asyncio.Task] = None
        self._state = StreamState.IDLE
        self._event_handlers: Dict[StreamEventType, List[Callable]] = {}
        self._is_initialized = False
        
    async def initialize(self, config: EngineConfig) -> None:
        """Initialize the provider-based strategy."""
        self.logger.info("ProviderStrategy.initialize() called")
        
        if self._is_initialized:
            raise RuntimeError("Already initialized")
            
        self.config = config
        
        try:
            self._state = StreamState.STARTING
            self.logger.info(f"Initializing provider strategy with provider: {config.provider}")
            
            # Get provider from registry
            registry = get_registry()
            provider_adapter = registry.get(self.config.provider)
            
            # Create provider config
            provider_config = CoreProviderConfig(
                api_key=self.config.api_key,
                endpoint=self.config.metadata.get("endpoint"),
                timeout=30.0,
                max_retries=3,
                metadata={
                    "model": self.config.metadata.get("model", "default"),
                    "voice": self.config.metadata.get("voice", "alloy"),
                    "language": self.config.metadata.get("language", "en"),
                    "temperature": self.config.metadata.get("temperature", 0.7)
                }
            )
            
            # Create stream config
            from ..core.stream_protocol import StreamConfig, AudioFormat as CoreAudioFormat
            try:
                stream_config = StreamConfig(
                    provider=self.config.provider,
                    mode="both",  # Support both audio and text
                    audio_format=CoreAudioFormat(
                        sample_rate=self.config.metadata.get("sample_rate", 24000),
                        channels=1,
                        bit_depth=16
                    ),
                    buffer_size_ms=self.config.metadata.get("chunk_duration_ms", 100),
                    enable_vad=self.config.enable_vad,
                    vad_threshold=self.config.metadata.get("vad_threshold", 0.5),
                    metadata={
                        "enable_transcription": self.config.enable_transcription,
                        "enable_functions": self.config.enable_functions
                    }
                )
                self.logger.info("Successfully created StreamConfig")
            except Exception as e:
                self.logger.error(f"Error creating StreamConfig: {e}")
                raise
            
            # Create session
            self.logger.info("Creating session with provider adapter...")
            try:
                self._session = await provider_adapter.create_session(provider_config, stream_config)
                self.logger.info("Session created successfully")
            except Exception as e:
                self.logger.error(f"Error creating session: {e}")
                raise
            
            # Initialize audio engine using unified interface
            vad_config = None
            if self.config.enable_vad:
                vad_config = VADConfig(
                    type=VADType.ENERGY_BASED,
                    energy_threshold=self.config.metadata.get("vad_threshold", 0.02),
                    speech_start_ms=self.config.metadata.get("vad_speech_start_ms", 100),
                    speech_end_ms=self.config.metadata.get("vad_speech_end_ms", 500)
                )
            
            # Create audio config for VoxStream
            audio_config = AudioConfig(
                sample_rate=stream_config.audio_format.sample_rate,
                channels=stream_config.audio_format.channels,
                chunk_duration_ms=self.config.metadata.get("chunk_duration_ms", 100)
            )
            
            self._audio_engine = VoxStream(
                config=audio_config,
                mode=ProcessingMode.REALTIME
            )
            
            # Configure devices and VAD
            self._audio_engine.configure_devices(
                input_device=self.config.input_device,
                output_device=self.config.output_device
            )
            
            if vad_config:
                self._audio_engine.configure_vad(vad_config)
            
            # Start event processing
            self._event_task = asyncio.create_task(self._process_provider_events())
            
            self._state = StreamState.IDLE
            self._is_initialized = True
            self.logger.info("Provider strategy initialized successfully")
            
        except Exception as e:
            self._state = StreamState.ERROR
            self.logger.error(f"Failed to initialize provider strategy: {e}")
            raise RuntimeError(f"Failed to initialize provider strategy: {e}") from e
            
    async def start_stream(self, **options) -> None:
        """Start the voice stream."""
        if self._state != StreamState.IDLE:
            raise RuntimeError("Strategy not ready to start stream")
            
        self._state = StreamState.ACTIVE
        
        # Start audio processing
        if self.config.input_device is not None:
            await self._audio_engine.start_stream(
                input_device=self.config.input_device,
                output_device=self.config.output_device
            )
            
            # Start audio forwarding
            asyncio.create_task(self._forward_audio_to_provider())
            
        self.logger.info("Stream started")
        
    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio to the provider."""
        if self._session and self._session.is_active:
            await self._session.send_audio(audio_data)
            
    async def send_text(self, text: str) -> None:
        """Send text to the provider."""
        if self._session and self._session.is_active:
            await self._session.send_text(text)
            # For OpenAI, we need to explicitly request a response
            if hasattr(self._session, 'create_response'):
                await self._session.create_response()
            
    async def interrupt(self) -> None:
        """Interrupt the current response."""
        if self._session and self._session.is_active:
            await self._session.interrupt()
            
    async def get_events(self) -> AsyncIterator[StreamEvent]:
        """Get events from the provider."""
        if self._session:
            async for event in self._session.get_event_stream():
                yield event
                
    async def stop_stream(self) -> None:
        """Stop the stream."""
        self._state = StreamState.ENDED
        
        # Stop audio
        if self._audio_engine:
            await self._audio_engine.stop_capture_stream()
            
        # Cancel event processing
        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("Stream stopped")
        
    async def cleanup(self) -> None:
        """Clean up resources."""
        # End session
        if self._session:
            usage = await self._session.end_session()
            self.logger.info(f"Session ended. Usage: {usage}")
            
        # Clean up audio
        if self._audio_engine:
            await self._audio_engine.cleanup_async()
            
        self._state = StreamState.ENDED
        self.logger.info("Provider strategy cleaned up")
        
    def get_state(self) -> StreamState:
        """Get current stream state."""
        return self._state
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get strategy metrics."""
        metrics = {
            "state": self._state.value,
            "provider": self.config.provider
        }
        
        if self._session:
            usage = self._session.get_usage()
            metrics["usage"] = {
                "audio_input_seconds": usage.audio_input_seconds,
                "audio_output_seconds": usage.audio_output_seconds,
                "text_input_tokens": usage.text_input_tokens,
                "text_output_tokens": usage.text_output_tokens
            }
            
        if self._audio_engine:
            audio_metrics = self._audio_engine.get_metrics()
            if audio_metrics:
                metrics["audio"] = audio_metrics
                
        return metrics
        
    def get_usage(self) -> Usage:
        """Get current usage stats."""
        if self._session:
            return self._session.get_usage()
        return Usage()
        
    def estimate_cost(self) -> Cost:
        """Estimate current cost."""
        usage = self.get_usage()
        registry = get_registry()
        provider = registry.get(self.config.provider)
        return provider.estimate_cost(usage)
        
    # Private methods
    
    async def _process_provider_events(self):
        """Process events from the provider."""
        try:
            async for event in self.get_events():
                # Handle audio playback
                if event.type == StreamEventType.AUDIO_OUTPUT_CHUNK:
                    audio_data = event.data.get("audio")
                    if audio_data and self._audio_engine:
                        # Convert from hex if needed
                        if isinstance(audio_data, str):
                            audio_bytes = bytes.fromhex(audio_data)
                        else:
                            audio_bytes = audio_data
                        self._audio_engine.queue_audio(audio_bytes)
                        
                # Emit event to engine
                await self._emit_event(event)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error processing provider events: {e}")
            
    async def _forward_audio_to_provider(self):
        """Forward captured audio to the provider."""
        if not self._audio_engine:
            return
            
        try:
            # Start audio capture
            audio_queue = await self._audio_engine.start_capture_stream()
            
            # Forward audio chunks
            while True:
                try:
                    audio_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    if audio_chunk:
                        await self.send_audio(audio_chunk)
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error forwarding audio: {e}")
            
    # Implementation of remaining abstract methods
    
    async def connect(self) -> None:
        """Establish connection to provider."""
        # For provider strategy, connection happens during initialize
        # But we need to ensure we're properly initialized
        if not self._is_initialized:
            raise RuntimeError("Strategy not initialized. Call initialize() first.")
        if not self._session or not self._session.is_active:
            raise RuntimeError("Provider session not active")
        
    async def disconnect(self) -> None:
        """Disconnect from provider - handled in cleanup."""
        # Disconnection is handled in cleanup
        await self.stop_stream()
        
    async def start_audio_input(self) -> None:
        """Start capturing audio input."""
        if self._audio_engine and self.config.input_device is not None:
            # Audio streaming already started in start_stream
            pass
            
    async def stop_audio_input(self) -> None:
        """Stop capturing audio input."""
        # Handled by stop_stream
        pass
        
    async def get_response_stream(self) -> AsyncIterator[StreamEvent]:
        """Get stream of response events."""
        async for event in self.get_events():
            yield event
            
    def set_event_handler(
        self,
        event_type: StreamEventType,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Set handler for specific event type."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        
    async def _emit_event(self, event: StreamEvent) -> None:
        """Emit event to registered handlers."""
        handlers = self._event_handlers.get(event.type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")
                
    async def estimate_cost(self) -> Cost:
        """Estimate cost of current session."""
        usage = self.get_usage()
        registry = get_registry()
        provider = registry.get(self.config.provider)
        return provider.estimate_cost(usage)