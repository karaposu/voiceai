"""
Fast Lane Strategy V2 - Provider-based Implementation

Refactored to use provider abstraction while maintaining
the same performance characteristics.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, AsyncIterator, Callable
import logging

from .base_strategy import BaseStrategy, EngineConfig
from ..core.stream_protocol import StreamEvent, StreamEventType, StreamState
from voxstream.config.types import StreamConfig as AudioConfig, VADConfig, ProcessingMode, VADType
from voxstream.voice.vad import VoiceState
AudioBytes = bytes  # Simple type alias for audio data
from ..core.provider_protocol import Usage, Cost
from ..core.exceptions import EngineError

# Import the new provider-based stream manager
from ..fast_lane.fast_stream_manager_v2 import FastStreamManagerV2, FastStreamConfig

# Import VoxStream VAD detector
from voxstream.voice.vad import VADetector as FastVADDetector

# Temporary placeholder for DirectAudioCapture
class DirectAudioCapture:
    def __init__(self, *args, **kwargs):
        pass
    def start(self):
        pass
    def stop(self):
        pass
    def get_metrics(self):
        return {}
    async def start_async_capture(self):
        return asyncio.Queue()

# Import VoiceState from voxstream
from voxstream.voice.vad import VoiceState as VADState


class FastLaneStrategyV2(BaseStrategy):
    """
    Provider-based fast lane implementation.
    
    Uses the provider abstraction while maintaining:
    - Client-side VAD only
    - Direct audio path
    - Minimal latency
    - Direct callbacks instead of events
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config: Optional[EngineConfig] = None
        
        # Core components
        self.audio_capture: Optional[DirectAudioCapture] = None
        self.vad_detector: Optional[FastVADDetector] = None
        self.stream_manager: Optional[FastStreamManagerV2] = None
        self._audio_engine = None  # For unified audio interface

        self._audio_queue: Optional[asyncio.Queue] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._error_callback: Optional[Callable] = None
        
        # Direct callbacks
        self._event_handlers: Dict[StreamEventType, Callable] = {}
        
        # State
        self._is_initialized = False
        self._is_connected = False
        self._is_capturing = False
        
        # Metrics
        self._start_time = 0
        self._audio_chunks_processed = 0
        self._total_latency_ms = 0
        self._vad_transitions = 0
    
    # ============== Strategy Implementation ==============
    
    async def initialize(self, config: EngineConfig) -> None:
        """Initialize fast lane with provider-based components"""
        if self._is_initialized:
            raise EngineError("Already initialized")
        
        self.config = config
        
        # Create audio config
        audio_config = AudioConfig(
            sample_rate=config.metadata.get("sample_rate", 24000),
            channels=1,
            chunk_duration_ms=config.metadata.get("chunk_duration_ms", 100)
        )
        
        # Initialize VAD if enabled
        if config.enable_vad:
            vad_config = VADConfig(
                type=VADType.ENERGY_BASED,
                energy_threshold=config.metadata.get("vad_threshold", 0.02),
                speech_start_ms=config.metadata.get("vad_speech_start_ms", 100),
                speech_end_ms=config.metadata.get("vad_speech_end_ms", 500)
            )
            
            self.vad_detector = FastVADDetector(
                config=vad_config,
                audio_config=audio_config
            )
        
        # Initialize audio capture
        self.audio_capture = DirectAudioCapture(
            device=config.input_device,
            config=audio_config,
        )
        
        # Initialize provider-based stream manager
        stream_config = FastStreamConfig(
            provider=config.provider,  # Use provider from config
            api_key=config.api_key,
            voice=config.metadata.get("voice", "alloy"),
            model=config.metadata.get("model", "gpt-4o-realtime-preview"),
            temperature=config.metadata.get("temperature", 0.7),
            send_immediately=True
        )
        
        self.stream_manager = FastStreamManagerV2(
            config=stream_config,
            logger=self.logger
        )
        
        # Wire up callbacks
        self._setup_callbacks()
        
        self._is_initialized = True
        self.logger.info("Fast lane strategy v2 initialized with provider")
    
    async def connect(self) -> None:
        """Connect to provider using fast stream manager"""
        if not self._is_initialized:
            raise EngineError("Not initialized")
        
        if self._is_connected:
            return
        
        self._start_time = time.time()
        
        # Start stream manager (connects to provider)
        await self.stream_manager.start()
        
        self._is_connected = True
        self.logger.info("Fast lane connected via provider")
    
    async def disconnect(self) -> None:
        """Disconnect from provider"""
        if not self._is_connected:
            return
        
        # Stop audio capture first
        if self._is_capturing:
            await self.stop_audio_input()
        
        # Stop stream
        if self.stream_manager:
            await self.stream_manager.stop()
        
        self._is_connected = False
        self.logger.info("Fast lane disconnected")
    
    async def start_audio_input(self) -> None:
        """Start audio input capture"""
        if not self.audio_capture:
            # Create audio capture
            audio_config = AudioConfig(
                sample_rate=24000,
                channels=1,
                chunk_duration_ms=100
            )
            
            self.audio_capture = DirectAudioCapture(
                device=self.config.input_device,
                config=audio_config,
                logger=self.logger
            )
        
        # Start capture and get the queue
        self._audio_queue = await self.audio_capture.start_async_capture()
        self._audio_task = asyncio.create_task(self._process_audio_queue())
        self._is_capturing = True
    
    async def _process_audio_queue(self):
        """Process audio queue in background"""
        try:
            while self._is_connected:
                try:
                    # Get audio chunk from queue
                    audio_chunk = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.1
                    )
                    
                    # Send to stream
                    await self.send_audio(audio_chunk)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Audio processing error: {e}")
                    if self._error_callback:
                        self._error_callback(e)
                        
        except asyncio.CancelledError:
            pass
    
    async def stop_audio_input(self) -> None:
        """Stop audio input capture"""
        if not self._is_capturing:
            return
        
        # Cancel audio task
        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
        
        # Stop capture
        if self.audio_capture:
            self.audio_capture.stop()
        
        self._is_capturing = False
    
    async def send_audio(self, audio_data: AudioBytes) -> None:
        """Send audio through provider"""
        if not self._is_connected or not self.stream_manager:
            return
        
        # Process through VAD if enabled
        if self.vad_detector:
            vad_state = self.vad_detector.process_chunk(audio_data)
            
            # Only send if speech detected
            if vad_state in [VADState.SPEECH_STARTING, VADState.SPEECH]:
                await self.stream_manager.send_audio(audio_data)
                self._audio_chunks_processed += 1
            
            # Trigger response on speech end
            elif vad_state == VADState.SPEECH_ENDING:
                await self.stream_manager.commit_audio_and_respond()
                self._vad_transitions += 1
        else:
            # No VAD, send all audio
            await self.stream_manager.send_audio(audio_data)
            self._audio_chunks_processed += 1
    
    async def send_text(self, text: str) -> None:
        """Send text through provider"""
        if not self._is_connected or not self.stream_manager:
            raise EngineError("Not connected")
        
        await self.stream_manager.send_text(text)
    
    async def interrupt(self) -> None:
        """Interrupt current response"""
        if not self._is_connected:
            return
        
        # Provider handles interruption
        if self.stream_manager and self.stream_manager._provider:
            await self.stream_manager._provider.interrupt()
    
    def get_state(self) -> StreamState:
        """Get current state"""
        if not self.stream_manager:
            return StreamState.IDLE
        return self.stream_manager.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fast lane metrics"""
        base_metrics = {}
        
        # Add stream metrics
        if self.stream_manager:
            base_metrics.update(self.stream_manager.get_metrics())
        
        # Add fast lane specific metrics
        base_metrics.update({
            "audio_chunks_processed": self._audio_chunks_processed,
            "vad_transitions": self._vad_transitions,
            "is_capturing": self._is_capturing,
            "provider": self.config.provider if self.config else "unknown"
        })
        
        # Add VAD metrics
        if self.vad_detector:
            base_metrics["vad_metrics"] = self.vad_detector.get_metrics()
        
        return base_metrics
    
    def get_usage(self) -> Usage:
        """Get usage stats"""
        # TODO: Get from provider
        return Usage()
    
    async def estimate_cost(self) -> Cost:
        """Estimate cost"""
        # TODO: Get from provider
        return Cost()
    
    # ============== Event Handling ==============
    
    def set_event_handler(
        self,
        event_type: StreamEventType,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Set event handler"""
        self._event_handlers[event_type] = handler
        
        # Map to stream manager callbacks
        if self.stream_manager:
            if event_type == StreamEventType.AUDIO_OUTPUT_CHUNK:
                self.stream_manager.set_audio_callback(
                    lambda audio: self._emit_audio_event(audio)
                )
            elif event_type == StreamEventType.TEXT_OUTPUT_CHUNK:
                self.stream_manager.set_text_callback(
                    lambda text: self._emit_text_event(text)
                )
            elif event_type == StreamEventType.STREAM_ERROR:
                self.stream_manager.set_error_callback(
                    lambda error: self._emit_error_event(error)
                )
    
    def _setup_callbacks(self):
        """Setup direct callbacks for performance"""
        if not self.stream_manager:
            return
        
        # Set up callbacks on stream manager
        self.stream_manager.set_audio_callback(self._on_audio_received)
        self.stream_manager.set_text_callback(self._on_text_received)
        self.stream_manager.set_error_callback(self._on_error)
        self.stream_manager.set_response_done_callback(self._on_response_done)
    
    def _emit_audio_event(self, audio: AudioBytes):
        """Emit audio event - compatibility method"""
        self._on_audio_received(audio)
    
    def _emit_text_event(self, text: str):
        """Emit text event - compatibility method"""
        self._on_text_received(text)
    
    def _emit_error_event(self, error: Exception):
        """Emit error event - compatibility method"""
        self._on_error(error)
    
    def _on_audio_received(self, audio: AudioBytes):
        """Direct audio callback"""
        handler = self._event_handlers.get(StreamEventType.AUDIO_OUTPUT_CHUNK)
        if handler:
            event = StreamEvent(
                type=StreamEventType.AUDIO_OUTPUT_CHUNK,
                stream_id=self.stream_manager.stream_id,
                timestamp=time.time(),
                data={"audio": audio}
            )
            handler(event)
    
    def _on_text_received(self, text: str):
        """Direct text callback"""
        handler = self._event_handlers.get(StreamEventType.TEXT_OUTPUT_CHUNK)
        if handler:
            event = StreamEvent(
                type=StreamEventType.TEXT_OUTPUT_CHUNK,
                stream_id=self.stream_manager.stream_id,
                timestamp=time.time(),
                data={"text": text}
            )
            handler(event)
    
    def _on_error(self, error: Exception):
        """Direct error callback"""
        handler = self._event_handlers.get(StreamEventType.STREAM_ERROR)
        if handler:
            event = StreamEvent(
                type=StreamEventType.STREAM_ERROR,
                stream_id=self.stream_manager.stream_id,
                timestamp=time.time(),
                data={"error": str(error)}
            )
            handler(event)
    
    def _on_response_done(self):
        """Response done callback"""
        handler = self._event_handlers.get(StreamEventType.STREAM_ENDED)
        if handler:
            event = StreamEvent(
                type=StreamEventType.STREAM_ENDED,
                stream_id=self.stream_manager.stream_id,
                timestamp=time.time(),
                data={}
            )
            handler(event)
    
    # ============== Required Abstract Methods ==============
    
    async def start_stream(self, **options) -> None:
        """Start streaming"""
        await self.start_audio_input()
    
    async def stop_stream(self) -> None:
        """Stop streaming"""
        await self.stop_audio_input()
    
    async def get_events(self) -> AsyncIterator[StreamEvent]:
        """Get event stream - not used in fast lane (uses callbacks)"""
        yield  # Empty generator
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.disconnect()
    
    async def get_response_stream(self) -> AsyncIterator[StreamEvent]:
        """Get response stream - not used in fast lane (uses callbacks)"""
        yield  # Empty generator