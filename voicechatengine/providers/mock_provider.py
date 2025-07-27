"""
Mock Provider Implementation for Testing

This module implements a mock provider that simulates AI provider behavior
without making actual API calls. Useful for testing and development.
"""

import asyncio
import json
import time
import logging
import random
from typing import Dict, Any, Optional, AsyncIterator, List
from collections import deque

from .base import (
    BaseProvider, ProviderConfig, ProviderEvent, AudioFormat,
    MessageType, ConnectionState, ConnectionError, MessageError
)


class MockConfig(ProviderConfig):
    """Mock provider configuration"""
    def __init__(self, **kwargs):
        # Extract mock-specific settings before passing to parent
        self.simulate_latency = kwargs.pop('simulate_latency', True)
        self.error_rate = kwargs.pop('error_rate', 0.0)  # 0-1 probability
        self.response_delay = kwargs.pop('response_delay', 0.5)  # seconds
        self.audio_chunk_size = kwargs.pop('audio_chunk_size', 1024)
        
        # Initialize parent with remaining kwargs
        super().__init__(**kwargs)


class MockProvider(BaseProvider):
    """
    Mock provider for testing VoiceEngine without real API calls.
    
    Features:
    - Simulates connection lifecycle
    - Generates mock responses
    - Simulates audio streaming
    - Configurable latency and errors
    - Records all interactions for testing
    """
    
    def __init__(self, config: MockConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self._event_queue: asyncio.Queue[ProviderEvent] = asyncio.Queue()
        self._message_history: List[Dict[str, Any]] = []
        self._audio_buffer: deque = deque()
        self._response_task: Optional[asyncio.Task] = None
        self._is_responding = False
        self._mock_responses = [
            "I understand. Let me help you with that.",
            "That's an interesting question. Here's what I think...",
            "Based on what you've told me, I would suggest...",
            "Let me process that information for you.",
            "I'm here to assist. What would you like to know?"
        ]
        
    async def connect(self) -> bool:
        """Simulate connection to provider."""
        try:
            self.state = ConnectionState.CONNECTING
            self.logger.info("Mock provider connecting...")
            
            # Simulate connection delay
            if self.config.simulate_latency:
                await asyncio.sleep(0.5)
            
            # Simulate connection error
            if random.random() < self.config.error_rate:
                self.state = ConnectionState.ERROR
                self.logger.error("Mock connection failed (simulated)")
                return False
            
            # Generate session ID
            self.session_id = f"mock_session_{int(time.time())}"
            self.state = ConnectionState.CONNECTED
            
            # Queue session created event
            await self._queue_event(ProviderEvent(
                type="session.created",
                data={"session": {"id": self.session_id}},
                timestamp=time.time(),
                provider="mock",
                session_id=self.session_id
            ))
            
            self.logger.info(f"Mock provider connected: {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Mock connection error: {e}")
            self.state = ConnectionState.ERROR
            return False
            
    async def disconnect(self) -> None:
        """Simulate disconnection."""
        self.state = ConnectionState.DISCONNECTED
        
        if self._response_task:
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
                
        self.session_id = None
        self._message_history.clear()
        self._audio_buffer.clear()
        self.logger.info("Mock provider disconnected")
        
    async def send_message(self, message_type: MessageType, data: Dict[str, Any]) -> None:
        """Record message and simulate processing."""
        if not self.is_connected():
            raise ConnectionError("Mock provider not connected")
            
        # Record message
        message = {
            "type": message_type.value,
            "data": data,
            "timestamp": time.time()
        }
        self._message_history.append(message)
        
        if self.config.enable_logging:
            self.logger.debug(f"Mock received: {message}")
            
        # Simulate message processing
        if self.config.simulate_latency:
            await asyncio.sleep(0.05)
            
        # Handle specific message types
        if message_type == MessageType.RESPONSE_CREATE:
            await self._simulate_response(data.get("modalities", ["text"]))
        elif message_type == MessageType.RESPONSE_CANCEL:
            await self._cancel_response()
            
    async def send_audio(self, audio_data: bytes) -> None:
        """Buffer audio data."""
        self._audio_buffer.append(audio_data)
        await self.send_message(
            MessageType.AUDIO_BUFFER_APPEND,
            {"audio": len(audio_data)}  # Just track size for mock
        )
        
    async def send_text(self, text: str, role: str = "user") -> None:
        """Process text message."""
        await self.send_message(
            MessageType.CONVERSATION_ITEM_CREATE,
            {
                "item": {
                    "type": "message",
                    "role": role,
                    "content": text
                }
            }
        )
        
        # Auto-respond if user message
        if role == "user" and self.config.simulate_latency:
            await asyncio.sleep(0.1)
            await self.create_response()
            
    async def interrupt(self) -> None:
        """Interrupt current response."""
        await self._cancel_response()
        await self.send_message(MessageType.RESPONSE_CANCEL, {})
        
    async def create_response(self, modalities: List[str] = None) -> None:
        """Generate mock response."""
        if modalities is None:
            modalities = ["text"]
            
        await self.send_message(
            MessageType.RESPONSE_CREATE,
            {"modalities": modalities}
        )
        
    def get_audio_format(self) -> AudioFormat:
        """Return mock audio format."""
        return AudioFormat(
            encoding="pcm16",
            sample_rate=16000,
            channels=1
        )
        
    async def events(self) -> AsyncIterator[ProviderEvent]:
        """Yield mock events."""
        while self.is_connected():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1
                )
                yield event
            except asyncio.TimeoutError:
                continue
                
    async def update_session(self, **kwargs) -> None:
        """Update mock session."""
        await self.send_message(
            MessageType.SESSION_UPDATE,
            {"session": kwargs}
        )
        
        # Queue confirmation event
        await self._queue_event(ProviderEvent(
            type="session.updated",
            data={"session": kwargs},
            timestamp=time.time(),
            provider="mock",
            session_id=self.session_id
        ))
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Return mock capabilities."""
        return {
            "provider": "mock",
            "models": ["mock-gpt"],
            "modalities": ["text", "audio"],
            "voices": ["mock-voice-1", "mock-voice-2"],
            "audio_formats": ["pcm16"],
            "sample_rates": [16000, 24000],
            "features": {
                "streaming": True,
                "interruption": True,
                "function_calling": False,
                "turn_detection": True,
                "vad": True
            }
        }
        
    # Mock-specific methods
    
    async def _simulate_response(self, modalities: List[str]) -> None:
        """Simulate AI response generation."""
        if self._is_responding:
            return
            
        self._is_responding = True
        self._response_task = asyncio.create_task(
            self._generate_response(modalities)
        )
        
    async def _generate_response(self, modalities: List[str]) -> None:
        """Generate mock response content."""
        try:
            # Simulate initial delay
            await asyncio.sleep(self.config.response_delay)
            
            # Start response
            await self._queue_event(ProviderEvent(
                type="response.started",
                data={"modalities": modalities},
                timestamp=time.time(),
                provider="mock",
                session_id=self.session_id
            ))
            
            if "text" in modalities:
                # Generate text response
                response_text = random.choice(self._mock_responses)
                
                # Stream text chunks
                words = response_text.split()
                for i, word in enumerate(words):
                    if not self._is_responding:
                        break
                        
                    chunk = word + (" " if i < len(words) - 1 else "")
                    await self._queue_event(ProviderEvent(
                        type="text_chunk",
                        data={"text": chunk},
                        timestamp=time.time(),
                        provider="mock",
                        session_id=self.session_id
                    ))
                    
                    # Simulate typing delay
                    if self.config.simulate_latency:
                        await asyncio.sleep(0.05)
                        
            if "audio" in modalities:
                # Generate mock audio chunks
                num_chunks = random.randint(5, 15)
                for i in range(num_chunks):
                    if not self._is_responding:
                        break
                        
                    # Create mock audio data
                    audio_data = bytes(self.config.audio_chunk_size)
                    
                    await self._queue_event(ProviderEvent(
                        type="audio_chunk",
                        data={"audio": audio_data.hex()},
                        timestamp=time.time(),
                        provider="mock",
                        session_id=self.session_id
                    ))
                    
                    if self.config.simulate_latency:
                        await asyncio.sleep(0.1)
                        
            # Complete response
            await self._queue_event(ProviderEvent(
                type="response.completed",
                data={},
                timestamp=time.time(),
                provider="mock",
                session_id=self.session_id
            ))
            
        except asyncio.CancelledError:
            self.logger.info("Mock response cancelled")
        except Exception as e:
            self.logger.error(f"Mock response error: {e}")
        finally:
            self._is_responding = False
            
    async def _cancel_response(self) -> None:
        """Cancel ongoing response."""
        self._is_responding = False
        if self._response_task:
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
                
    async def _queue_event(self, event: ProviderEvent) -> None:
        """Queue event and emit to handlers."""
        await self._event_queue.put(event)
        await self._emit_event(event)
        
    # Testing utilities
    
    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get recorded message history for testing."""
        return self._message_history.copy()
        
    def clear_message_history(self) -> None:
        """Clear message history."""
        self._message_history.clear()
        
    def get_audio_buffer_size(self) -> int:
        """Get current audio buffer size."""
        return sum(len(chunk) for chunk in self._audio_buffer)
        
    def set_mock_responses(self, responses: List[str]) -> None:
        """Set custom mock responses."""
        self._mock_responses = responses
        
    async def simulate_error(self, error_type: str, error_message: str) -> None:
        """Simulate an error event."""
        await self._queue_event(ProviderEvent(
            type="error",
            data={"type": error_type, "message": error_message},
            timestamp=time.time(),
            provider="mock",
            session_id=self.session_id,
            error=error_message
        ))