"""
Fast Stream Manager V2 - Provider-based Implementation

Refactored version that uses the provider abstraction while maintaining
the same fast performance characteristics.
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import logging

from typing import List
from voicechatengine.core.stream_protocol import (
    IStreamManager, StreamState, StreamEvent, StreamEventType,
    StreamMetrics, AudioFormat
)
from voicechatengine.providers import get_registry, ProviderEvent, MessageType, ConnectionState
from voicechatengine.providers.base import BaseProvider
from voicechatengine.core.exceptions import StreamError
from voxstream.voice.vad import VoiceState as VADState

AudioBytes = bytes  # Simple type alias for audio data


@dataclass
class FastStreamConfig:
    """Configuration for provider-based fast stream"""
    provider: str = "openai"  # Default to OpenAI for backward compatibility
    api_key: str = ""
    voice: str = "alloy"
    audio_format: AudioFormat = field(default_factory=lambda: AudioFormat(
        sample_rate=24000,
        channels=1,
        bit_depth=16
    ))
    
    # Performance settings
    send_immediately: bool = True  # No buffering
    event_callbacks: bool = True   # Direct callbacks vs queue
    
    # Provider-specific settings
    model: str = "gpt-4o-realtime-preview"
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class FastStreamManagerV2(IStreamManager):
    """
    Provider-based fast stream manager.
    
    Uses the provider abstraction while maintaining the same
    performance characteristics as the original fast lane.
    """
    
    def __init__(
        self,
        config: FastStreamConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Provider instance
        self._provider: Optional[BaseProvider] = None
        self._stream_id = f"fast_{int(time.time() * 1000)}"
        self._state = StreamState.IDLE
        
        # Direct callbacks (no event queue)
        self._audio_callback: Optional[Callable[[AudioBytes], None]] = None
        self._text_callback: Optional[Callable[[str], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._response_done_callback: Optional[Callable[[], None]] = None
        
        # Event processing task
        self._event_task: Optional[asyncio.Task] = None
        
        # Minimal metrics
        self._start_time = 0.0
        self._audio_bytes_sent = 0
        self._audio_bytes_received = 0
        
    # ============== IStreamManager Implementation ==============
    
    @property
    def stream_id(self) -> str:
        return self._stream_id
    
    @property
    def state(self) -> StreamState:
        return self._state
    
    async def start(self) -> None:
        """Start the stream using provider abstraction"""
        if self._state != StreamState.IDLE:
            raise StreamError(f"Cannot start in state {self._state}")
        
        self._state = StreamState.STARTING
        self._start_time = time.time()
        
        try:
            # Get provider from registry
            registry = get_registry()
            provider_adapter = registry.get(self.config.provider)
            
            # Create provider config
            from voicechatengine.providers.openai_provider import OpenAIConfig
            provider_config = OpenAIConfig(
                api_key=self.config.api_key,
                model=self.config.model,
                voice=self.config.voice,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Create provider instance
            self._provider = provider_adapter._provider_class(provider_config)
            
            # Connect to provider
            if not await self._provider.connect():
                raise StreamError("Failed to connect to provider")
            
            # Configure session for fast lane (client-side VAD)
            await self._provider.update_session(
                voice=self.config.voice,
                modalities=["text", "audio"],
                instructions="You are a helpful voice assistant.",
                turn_detection={
                    "type": "client",  # Client-side VAD for fast lane
                    "create_response": False  # We'll manually trigger responses
                },
                input_audio_format="pcm16",
                output_audio_format="pcm16"
            )
            
            # Start event processing
            self._event_task = asyncio.create_task(self._process_provider_events())
            
            self._state = StreamState.ACTIVE
            
        except Exception as e:
            self._state = StreamState.ERROR
            if self._error_callback:
                self._error_callback(e)
            raise
    
    async def stop(self) -> None:
        """Stop the stream"""
        if self._state not in [StreamState.ACTIVE, StreamState.ERROR]:
            return
        
        self._state = StreamState.ENDING
        
        # Cancel event processing
        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect provider
        if self._provider:
            await self._provider.disconnect()
            self._provider = None
        
        self._state = StreamState.ENDED
    
    async def send_audio(self, audio_data: AudioBytes) -> None:
        """Send audio with zero buffering"""
        if self._state != StreamState.ACTIVE:
            raise StreamError(f"Cannot send audio in state {self._state}")
        
        if not self._provider:
            raise StreamError("Provider not initialized")
        
        # Send audio directly to provider
        await self._provider.send_audio(audio_data)
        
        # Update metrics
        self._audio_bytes_sent += len(audio_data)
    
    async def send_text(self, text: str) -> None:
        """Send text message"""
        if self._state != StreamState.ACTIVE:
            raise StreamError(f"Cannot send text in state {self._state}")
        
        if not self._provider:
            raise StreamError("Provider not initialized")
        
        # Send text to provider
        await self._provider.send_text(text)
        
        # For client-side VAD, manually trigger response
        await self._provider.create_response()
    
    async def commit_audio_and_respond(self) -> None:
        """
        Commit audio buffer and trigger response.
        Used when we have a complete audio recording.
        """
        if self._state != StreamState.ACTIVE:
            raise StreamError(f"Cannot commit audio in state {self._state}")
        
        if not self._provider:
            raise StreamError("Provider not initialized")
        
        # Commit audio buffer
        await self._provider.send_message(
            MessageType.AUDIO_BUFFER_COMMIT,
            {}
        )
        
        # Trigger response
        await self._provider.create_response()
    
    def subscribe_events(
        self,
        event_types: List[StreamEventType],
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Subscribe to events with direct callbacks"""
        # Map to direct callbacks for performance
        if StreamEventType.AUDIO_OUTPUT_CHUNK in event_types:
            self._audio_callback = lambda audio: handler(
                StreamEvent(
                    type=StreamEventType.AUDIO_OUTPUT_CHUNK,
                    stream_id=self._stream_id,
                    timestamp=time.time(),
                    data={"audio": audio}
                )
            )
        
        if StreamEventType.TEXT_OUTPUT_CHUNK in event_types:
            self._text_callback = lambda text: handler(
                StreamEvent(
                    type=StreamEventType.TEXT_OUTPUT_CHUNK,
                    stream_id=self._stream_id,
                    timestamp=time.time(),
                    data={"text": text}
                )
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get minimal metrics"""
        uptime = time.time() - self._start_time if self._start_time else 0
        
        return {
            "state": self._state.value,
            "uptime_seconds": uptime,
            "audio_bytes_sent": self._audio_bytes_sent,
            "audio_bytes_received": self._audio_bytes_received,
            "throughput_bps": (
                (self._audio_bytes_sent + self._audio_bytes_received) / uptime
                if uptime > 0 else 0
            )
        }
    
    # ============== Fast Direct Callbacks ==============
    
    def set_audio_callback(self, callback: Callable[[AudioBytes], None]):
        """Set direct audio callback (fastest)"""
        self._audio_callback = callback
    
    def set_text_callback(self, callback: Callable[[str], None]):
        """Set direct text callback"""
        self._text_callback = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]):
        """Set error callback"""
        self._error_callback = callback
    
    def set_response_done_callback(self, callback: Callable[[], None]):
        """Set response done callback"""
        self._response_done_callback = callback
    
    # ============== Private Methods ==============
    
    async def _process_provider_events(self):
        """
        Process events from the provider.
        Optimized for minimal overhead.
        """
        if not self._provider:
            return
        
        try:
            async for event in self._provider.events():
                # Fast path for audio
                if event.type == "audio_chunk":
                    if self._audio_callback and event.data:
                        audio_hex = event.data.get("audio", "")
                        if audio_hex:
                            # Convert hex to bytes
                            audio_bytes = bytes.fromhex(audio_hex)
                            self._audio_bytes_received += len(audio_bytes)
                            self._audio_callback(audio_bytes)
                
                # Text response
                elif event.type == "text_chunk":
                    if self._text_callback and event.data:
                        text = event.data.get("text", "")
                        if text:
                            self._text_callback(text)
                
                # Response done
                elif event.type == "response.completed":
                    if self._response_done_callback:
                        self._response_done_callback()
                
                # Error
                elif event.type == "error":
                    if self._error_callback and event.data:
                        error_msg = event.data.get("message", "Unknown error")
                        self._error_callback(StreamError(error_msg))
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error processing provider events: {e}")
            if self._error_callback:
                self._error_callback(e)


class FastVADStreamManagerV2(FastStreamManagerV2):
    """
    Specialized fast stream manager with integrated VAD.
    
    Combines provider-based stream management with VAD for ultimate performance.
    """
    
    def __init__(
        self,
        config: FastStreamConfig,
        vad_detector: 'FastVADDetector',
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        self.vad_detector = vad_detector
        self._current_vad_state: Optional[VADState] = None
        
    async def send_audio_with_vad(
        self,
        audio_data: AudioBytes,
        vad_state: VADState
    ) -> None:
        """
        Send audio with VAD state optimization.
        Only send when speech is detected.
        """
        # Only send if speech detected
        if vad_state == VADState.SPEECH:
            await self.send_audio(audio_data)
            
        # Handle state transitions
        if self._current_vad_state != vad_state:
            if vad_state == VADState.SPEECH_END:
                # Speech ended, trigger response
                await self.commit_audio_and_respond()
            
            self._current_vad_state = vad_state