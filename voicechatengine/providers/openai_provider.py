# here is voicechatengine/providers/openai_provider.py

"""
OpenAI Realtime API Provider Implementation

This module implements the BaseProvider interface for OpenAI's Realtime API.
It handles all OpenAI-specific protocol details while exposing a standard interface.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, AsyncIterator, List
import websockets
from websockets.client import WebSocketClientProtocol

from .base import (
    BaseProvider, ProviderConfig, ProviderEvent, AudioFormat,
    MessageType, ConnectionState, ConnectionError, MessageError
)


class OpenAIConfig(ProviderConfig):
    """OpenAI-specific configuration"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.url:
            self.url = "wss://api.openai.com/v1/realtime"
        if self.model == "default":
            self.model = "gpt-4o-realtime-preview"


class OpenAIProvider(BaseProvider):
    """
    OpenAI Realtime API provider implementation.
    
    This provider handles the WebSocket connection and protocol
    specific to OpenAI's Realtime API.
    """
    
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.ws: Optional[WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._event_queue: asyncio.Queue[ProviderEvent] = asyncio.Queue()
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to OpenAI."""
        try:
            self.state = ConnectionState.CONNECTING
            
            # Build connection URL with model parameter
            url = f"{self.config.url}?model={self.config.model}"
            
            # Set up headers
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            # Connect
            self.ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            # Wait for session creation
            session_created = await self._wait_for_session()
            if session_created:
                self.state = ConnectionState.CONNECTED
                self.logger.info(f"Connected to OpenAI: {self.session_id}")
                return True
            else:
                await self.disconnect()
                return False
                
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.state = ConnectionState.ERROR
            return False
            
    async def disconnect(self) -> None:
        """Disconnect from OpenAI."""
        self.state = ConnectionState.DISCONNECTED
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
                
        if self.ws:
            await self.ws.close()
            self.ws = None
            
        self.session_id = None
        self.logger.info("Disconnected from OpenAI")
        
    async def send_message(self, message_type: MessageType, data: Dict[str, Any]) -> None:
        """Send a message to OpenAI."""
        if not self.is_connected():
            raise ConnectionError("Not connected to OpenAI")
            
        # Map generic message types to OpenAI-specific types
        openai_type = self._map_message_type(message_type)
        
        message = {
            "type": openai_type,
            **data
        }
        
        if self.config.enable_logging:
            self.logger.debug(f"Sending: {message}")
            
        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            raise MessageError(f"Failed to send message: {e}")
            
    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data to OpenAI."""
        await self.send_message(
            MessageType.AUDIO_BUFFER_APPEND,
            {"audio": audio_data.hex()}  # OpenAI expects hex-encoded audio
        )
        
    async def send_text(self, text: str, role: str = "user") -> None:
        """Send text message to OpenAI."""
        await self.send_message(
            MessageType.CONVERSATION_ITEM_CREATE,
            {
                "item": {
                    "type": "message",
                    "role": role,
                    "content": [
                        {"type": "text", "text": text}
                    ]
                }
            }
        )
        
    async def interrupt(self) -> None:
        """Interrupt the current response."""
        await self.send_message(MessageType.RESPONSE_CANCEL, {})
        
    async def create_response(self, modalities: List[str] = None) -> None:
        """Request a response from OpenAI."""
        if modalities is None:
            modalities = ["text", "audio"]
            
        await self.send_message(
            MessageType.RESPONSE_CREATE,
            {"modalities": modalities}
        )
        
    def get_audio_format(self) -> AudioFormat:
        """Get OpenAI's required audio format."""
        return AudioFormat(
            encoding="pcm16",
            sample_rate=24000,
            channels=1
        )
        
    async def events(self) -> AsyncIterator[ProviderEvent]:
        """Async iterator for OpenAI events."""
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
        """Update OpenAI session configuration."""
        session_update = {
            "session": {}
        }
        
        # Map common parameters to OpenAI format
        if "voice" in kwargs:
            session_update["session"]["voice"] = kwargs["voice"]
        if "instructions" in kwargs:
            session_update["session"]["instructions"] = kwargs["instructions"]
        if "temperature" in kwargs:
            session_update["session"]["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            session_update["session"]["max_response_output_tokens"] = kwargs["max_tokens"]
            
        await self.send_message(MessageType.SESSION_UPDATE, session_update)
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get OpenAI provider capabilities."""
        return {
            "provider": "openai",
            "models": ["gpt-4o-realtime-preview"],
            "modalities": ["text", "audio"],
            "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            "audio_formats": ["pcm16"],
            "sample_rates": [24000],
            "features": {
                "streaming": True,
                "interruption": True,
                "function_calling": True,
                "turn_detection": True,
                "vad": True
            }
        }
        
    # Private methods
    
    def _map_message_type(self, message_type: MessageType) -> str:
        """Map generic message types to OpenAI-specific types."""
        mapping = {
            MessageType.SESSION_CREATE: "session.create",
            MessageType.SESSION_UPDATE: "session.update",
            MessageType.CONVERSATION_ITEM_CREATE: "conversation.item.create",
            MessageType.CONVERSATION_ITEM_DELETE: "conversation.item.delete",
            MessageType.AUDIO_BUFFER_APPEND: "input_audio_buffer.append",
            MessageType.AUDIO_BUFFER_COMMIT: "input_audio_buffer.commit",
            MessageType.AUDIO_BUFFER_CLEAR: "input_audio_buffer.clear",
            MessageType.RESPONSE_CREATE: "response.create",
            MessageType.RESPONSE_CANCEL: "response.cancel"
        }
        return mapping.get(message_type, message_type.value)
        
    async def _receive_loop(self) -> None:
        """Receive messages from OpenAI WebSocket."""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    event = self._convert_to_provider_event(data)
                    
                    # Put in queue for events() iterator
                    await self._event_queue.put(event)
                    
                    # Emit to handlers
                    await self._emit_event(event)
                    
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
            self.state = ConnectionState.DISCONNECTED
        except Exception as e:
            self.logger.error(f"Receive loop error: {e}")
            self.state = ConnectionState.ERROR
            
    def _convert_to_provider_event(self, data: Dict[str, Any]) -> ProviderEvent:
        """Convert OpenAI message to standard ProviderEvent."""
        event_type = data.get("type", "unknown")
        
        # Handle session events
        if event_type == "session.created":
            self.session_id = data.get("session", {}).get("id")
            
        # Map OpenAI events to standard events
        if event_type == "response.audio.delta":
            event_type = "audio_chunk"
        elif event_type == "response.text.delta":
            event_type = "text_chunk"
        elif event_type == "response.function_call_arguments.done":
            event_type = "function_call"
            
        return ProviderEvent(
            type=event_type,
            data=data,
            timestamp=time.time(),
            provider="openai",
            session_id=self.session_id,
            error=data.get("error")
        )
        
    async def _wait_for_session(self, timeout: float = 5.0) -> bool:
        """Wait for session creation confirmation."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.session_id:
                return True
            await asyncio.sleep(0.1)
            
        return False