"""
Provider Adapter

Adapts the new BaseProvider interface to work with the existing provider protocol.
This allows seamless integration of v2 providers with the current VoiceEngine.
"""

import asyncio
import time
from typing import Dict, Any, Optional, AsyncIterator, Tuple, List
from dataclasses import dataclass

from ..core.provider_protocol import (
    IVoiceProvider, IProviderSession, ProviderCapabilities, ProviderFeature,
    CostModel, CostUnit, Usage, Cost, ProviderConfig as CoreProviderConfig,
    VoiceConfig, TranscriptionConfig, FunctionDefinition, FunctionCall,
    QualityPreset, QualitySettings
)
from ..core.stream_protocol import StreamEvent, StreamEventType, StreamConfig, AudioFormat as CoreAudioFormat

from .base import BaseProvider, ProviderEvent, MessageType, AudioFormat


class ProviderAdapter(IVoiceProvider):
    """
    Adapts BaseProvider implementations to IVoiceProvider interface.
    """
    
    def __init__(self, provider_class: type[BaseProvider], name: str):
        self._provider_class = provider_class
        self._name = name
        
    @property
    def name(self) -> str:
        return self._name
        
    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities from the base provider."""
        # Create config class based on provider type
        if self._name == "openai":
            from .openai_provider import OpenAIConfig
            temp_config = OpenAIConfig(api_key="temp")
        elif self._name == "mock":
            from .mock_provider import MockConfig
            temp_config = MockConfig(api_key="temp")
        else:
            from .base import ProviderConfig
            temp_config = ProviderConfig(api_key="temp")
            
        temp_provider = self._provider_class(temp_config)
        caps = temp_provider.get_capabilities()
        
        # Map capabilities to ProviderCapabilities
        features = []
        if caps.get("features", {}).get("streaming", False):
            features.append(ProviderFeature.STREAMING_TEXT)
        if "audio" in caps.get("modalities", []):
            features.append(ProviderFeature.REALTIME_VOICE)
        if caps.get("features", {}).get("function_calling", False):
            features.append(ProviderFeature.FUNCTION_CALLING)
        if caps.get("features", {}).get("vad", False):
            features.append(ProviderFeature.SERVER_VAD)
        if caps.get("features", {}).get("interruption", False):
            features.append(ProviderFeature.INTERRUPTION)
            
        # Get audio format
        audio_format = temp_provider.get_audio_format()
        
        return ProviderCapabilities(
            provider_name=self._name,
            features=features,
            supported_audio_formats=[CoreAudioFormat(
                sample_rate=audio_format.sample_rate,
                channels=audio_format.channels,
                bit_depth=16 if audio_format.encoding == "pcm16" else 16
            )],
            supported_sample_rates=caps.get("sample_rates", [audio_format.sample_rate]),
            max_audio_duration_ms=300000,  # 5 minutes default
            min_audio_chunk_ms=100,
            available_voices=caps.get("voices", ["alloy"]),
            supports_voice_config=True,
            supported_languages=["en"],  # Default to English
            max_concurrent_streams=1
        )
        
    def get_cost_model(self) -> CostModel:
        """Get cost model for the provider."""
        # Default cost model - providers can override
        return CostModel(
            audio_input_cost=0.06,  # $0.06 per minute
            audio_input_unit=CostUnit.PER_MINUTE,
            audio_output_cost=0.24,  # $0.24 per minute
            audio_output_unit=CostUnit.PER_MINUTE,
            text_input_cost=0.00001,  # $0.01 per 1k tokens
            text_input_unit=CostUnit.PER_TOKEN,
            text_output_cost=0.00003,  # $0.03 per 1k tokens
            text_output_unit=CostUnit.PER_TOKEN
        )
        
    async def create_session(
        self,
        config: CoreProviderConfig,
        stream_config: StreamConfig
    ) -> 'ProviderSessionAdapter':
        """Create a new session with the provider."""
        # Create config class based on provider type
        if self._name == "openai":
            from .openai_provider import OpenAIConfig
            provider_config = OpenAIConfig(
                api_key=config.api_key,
                model=config.metadata.get("model", "default"),
                url=config.endpoint,
                voice=config.metadata.get("voice", "alloy"),
                language=config.metadata.get("language", "en"),
                temperature=config.metadata.get("temperature", 0.7),
                max_tokens=config.metadata.get("max_tokens"),
                timeout=int(config.timeout),
                retry_attempts=config.max_retries
            )
        elif self._name == "mock":
            from .mock_provider import MockConfig
            provider_config = MockConfig(
                api_key=config.api_key,
                model=config.metadata.get("model", "default"),
                url=config.endpoint,
                voice=config.metadata.get("voice", "alloy"),
                language=config.metadata.get("language", "en"),
                temperature=config.metadata.get("temperature", 0.7),
                max_tokens=config.metadata.get("max_tokens"),
                timeout=int(config.timeout),
                retry_attempts=config.max_retries
            )
        else:
            from .base import ProviderConfig
            provider_config = ProviderConfig(
                api_key=config.api_key,
                model=config.metadata.get("model", "default"),
                url=config.endpoint,
                voice=config.metadata.get("voice", "alloy"),
                language=config.metadata.get("language", "en"),
                temperature=config.metadata.get("temperature", 0.7),
                max_tokens=config.metadata.get("max_tokens"),
                timeout=int(config.timeout),
                retry_attempts=config.max_retries
            )
        
        # Create provider instance
        provider = self._provider_class(provider_config)
        
        # Create session adapter
        session = ProviderSessionAdapter(provider, stream_config)
        
        # Connect
        if await provider.connect():
            return session
        else:
            raise ConnectionError(f"Failed to connect to {self._name}")
            
    async def validate_config(self, config: CoreProviderConfig) -> Tuple[bool, Optional[str]]:
        """Validate provider configuration."""
        if not config.api_key:
            return False, "API key is required"
            
        # Basic validation passed
        return True, None
        
    def estimate_cost(self, usage: Usage) -> Cost:
        """Estimate cost for given usage."""
        cost_model = self.get_cost_model()
        
        audio_cost = (
            (usage.audio_input_seconds / 60) * cost_model.audio_input_cost +
            (usage.audio_output_seconds / 60) * cost_model.audio_output_cost
        )
        
        text_cost = (
            (usage.text_input_tokens / 1000) * cost_model.text_input_cost +
            (usage.text_output_tokens / 1000) * cost_model.text_output_cost
        )
        
        session_cost = usage.session_count * cost_model.session_cost
        function_cost = usage.function_calls * cost_model.function_call_cost
        
        return Cost(
            audio_cost=audio_cost,
            text_cost=text_cost,
            session_cost=session_cost,
            function_cost=function_cost
        )


class ProviderSessionAdapter(IProviderSession):
    """
    Adapts a BaseProvider instance to IProviderSession interface.
    """
    
    def __init__(self, provider: BaseProvider, stream_config: StreamConfig):
        self._provider = provider
        self._stream_config = stream_config
        self._usage = Usage()
        self._start_time = time.time()
        self._audio_input_bytes = 0
        self._audio_output_bytes = 0
        
    @property
    def session_id(self) -> str:
        return self._provider.session_id or "unknown"
        
    @property
    def is_active(self) -> bool:
        return self._provider.is_connected()
        
    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio to provider."""
        await self._provider.send_audio(audio_data)
        self._audio_input_bytes += len(audio_data)
        
        # Update usage (assuming 16-bit PCM at provider's sample rate)
        audio_format = self._provider.get_audio_format()
        samples = len(audio_data) / 2  # 16-bit = 2 bytes per sample
        seconds = samples / audio_format.sample_rate
        self._usage.audio_input_seconds += seconds
        
    async def send_text(self, text: str) -> None:
        """Send text to provider."""
        await self._provider.send_text(text)
        # Rough token estimation
        self._usage.text_input_tokens += len(text.split()) * 1.3
        
    async def send_function_result(self, call_id: str, result: Any) -> None:
        """Send function call result."""
        # Convert to provider message format
        await self._provider.send_message(
            MessageType.CONVERSATION_ITEM_CREATE,
            {
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result
                }
            }
        )
        
    def get_event_stream(self) -> AsyncIterator[StreamEvent]:
        """Get stream of events from provider."""
        return self._convert_events()
        
    async def _convert_events(self) -> AsyncIterator[StreamEvent]:
        """Convert provider events to stream events."""
        async for event in self._provider.events():
            stream_event = self._convert_provider_event(event)
            if stream_event:
                yield stream_event
                
    def _convert_provider_event(self, event: ProviderEvent) -> Optional[StreamEvent]:
        """Convert a provider event to a stream event."""
        # Map provider event types to stream event types
        type_mapping = {
            "session.created": StreamEventType.STREAM_STARTED,
            "session.updated": StreamEventType.STREAM_READY,
            "audio_chunk": StreamEventType.AUDIO_OUTPUT_CHUNK,
            "text_chunk": StreamEventType.TEXT_OUTPUT_CHUNK,
            "response.started": StreamEventType.RESPONSE_STARTED,
            "response.completed": StreamEventType.RESPONSE_COMPLETED,
            "error": StreamEventType.STREAM_ERROR,
            "function_call": StreamEventType.FUNCTION_CALL_STARTED
        }
        
        stream_type = type_mapping.get(event.type)
        if not stream_type:
            # Use generic event type for unmapped events
            stream_type = StreamEventType.STREAM_READY
            
        # Update usage for output events
        if event.type == "audio_chunk":
            audio_hex = event.data.get("audio", "")
            audio_bytes = len(bytes.fromhex(audio_hex))
            self._audio_output_bytes += audio_bytes
            
            # Update usage
            audio_format = self._provider.get_audio_format()
            samples = audio_bytes / 2
            seconds = samples / audio_format.sample_rate
            self._usage.audio_output_seconds += seconds
            
        elif event.type == "text_chunk":
            text = event.data.get("text", "")
            self._usage.text_output_tokens += len(text.split()) * 1.3
            
        return StreamEvent(
            type=stream_type,
            stream_id=self.session_id,
            timestamp=event.timestamp,
            data=event.data,
            metadata={
                "provider": event.provider,
                "original_type": event.type,
                "session_id": event.session_id
            }
        )
        
    async def interrupt(self) -> None:
        """Interrupt current response."""
        await self._provider.interrupt()
        
    async def end_session(self) -> Usage:
        """End session and return usage stats."""
        await self._provider.disconnect()
        
        # Finalize usage
        self._usage.session_count = 1
        session_duration = time.time() - self._start_time
        
        return self._usage
        
    def get_usage(self) -> Usage:
        """Get current usage stats."""
        return Usage(
            audio_input_seconds=self._usage.audio_input_seconds,
            audio_output_seconds=self._usage.audio_output_seconds,
            text_input_tokens=self._usage.text_input_tokens,
            text_output_tokens=self._usage.text_output_tokens,
            function_calls=self._usage.function_calls,
            session_count=0  # Not counted until session ends
        )