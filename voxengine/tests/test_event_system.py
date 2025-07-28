"""
Tests for the Voice Engine Event System
"""

import asyncio
import pytest
from typing import List, Dict, Any
import time

from voxengine.events import (
    EventEmitter, EventType, Event,
    AudioEvent, TextEvent, ConnectionEvent,
    ErrorEvent, FunctionCallEvent
)


class TestEventEmitter:
    """Test the EventEmitter class"""
    
    @pytest.mark.asyncio
    async def test_basic_event_emission(self):
        """Test basic event emission and handling"""
        emitter = EventEmitter()
        events_received = []
        
        # Register handler
        def handler(event: Event):
            events_received.append(event)
        
        emitter.on(EventType.TEXT_OUTPUT, handler)
        
        # Emit event
        event = TextEvent(type=EventType.TEXT_OUTPUT, text="Hello")
        await emitter.emit(event, sync=True)
        
        assert len(events_received) == 1
        assert events_received[0].text == "Hello"
    
    @pytest.mark.asyncio
    async def test_async_handler(self):
        """Test async event handlers"""
        emitter = EventEmitter()
        events_received = []
        
        # Async handler
        async def async_handler(event: Event):
            await asyncio.sleep(0.01)  # Simulate async work
            events_received.append(event)
        
        emitter.on(EventType.AUDIO_OUTPUT_CHUNK, async_handler)
        
        # Emit event
        event = AudioEvent(type=EventType.AUDIO_OUTPUT_CHUNK, audio_data=b"test")
        await emitter.emit(event, sync=True)
        
        assert len(events_received) == 1
        assert events_received[0].audio_data == b"test"
    
    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        """Test multiple handlers for same event"""
        emitter = EventEmitter()
        handler1_called = False
        handler2_called = False
        
        def handler1(event: Event):
            nonlocal handler1_called
            handler1_called = True
        
        def handler2(event: Event):
            nonlocal handler2_called
            handler2_called = True
        
        emitter.on(EventType.CONNECTION_ESTABLISHED, handler1)
        emitter.on(EventType.CONNECTION_ESTABLISHED, handler2)
        
        event = ConnectionEvent(type=EventType.CONNECTION_ESTABLISHED)
        await emitter.emit(event, sync=True)
        
        assert handler1_called
        assert handler2_called
    
    @pytest.mark.asyncio
    async def test_handler_priority(self):
        """Test handler execution order by priority"""
        emitter = EventEmitter()
        execution_order = []
        
        def low_priority(event: Event):
            execution_order.append("low")
        
        def high_priority(event: Event):
            execution_order.append("high")
        
        def medium_priority(event: Event):
            execution_order.append("medium")
        
        # Register in random order with different priorities
        emitter.on(EventType.RESPONSE_STARTED, low_priority, priority=0)
        emitter.on(EventType.RESPONSE_STARTED, high_priority, priority=10)
        emitter.on(EventType.RESPONSE_STARTED, medium_priority, priority=5)
        
        event = Event(type=EventType.RESPONSE_STARTED)
        await emitter.emit(event, sync=True)
        
        assert execution_order == ["high", "medium", "low"]
    
    @pytest.mark.asyncio
    async def test_once_handler(self):
        """Test one-time handlers"""
        emitter = EventEmitter()
        call_count = 0
        
        def handler(event: Event):
            nonlocal call_count
            call_count += 1
        
        emitter.once(EventType.RESPONSE_COMPLETED, handler)
        
        # Emit twice
        event = Event(type=EventType.RESPONSE_COMPLETED)
        await emitter.emit(event, sync=True)
        await emitter.emit(event, sync=True)
        
        # Should only be called once
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_event_filter(self):
        """Test event filtering"""
        emitter = EventEmitter()
        events_received = []
        
        def handler(event: TextEvent):
            events_received.append(event.text)
        
        # Only handle events with specific text
        def filter_func(event: TextEvent) -> bool:
            return "important" in event.text.lower()
        
        emitter.on(EventType.TEXT_OUTPUT, handler, filter=filter_func)
        
        # Emit various events
        await emitter.emit(TextEvent(type=EventType.TEXT_OUTPUT, text="Hello"), sync=True)
        await emitter.emit(TextEvent(type=EventType.TEXT_OUTPUT, text="Important message"), sync=True)
        await emitter.emit(TextEvent(type=EventType.TEXT_OUTPUT, text="Another important note"), sync=True)
        
        assert events_received == ["Important message", "Another important note"]
    
    @pytest.mark.asyncio
    async def test_global_handler(self):
        """Test global handlers that receive all events"""
        emitter = EventEmitter()
        all_events = []
        
        def global_handler(event: Event):
            all_events.append(event.type)
        
        # Register for all events
        emitter.on("*", global_handler)
        
        # Emit various event types
        await emitter.emit(Event(type=EventType.CONNECTION_ESTABLISHED), sync=True)
        await emitter.emit(TextEvent(type=EventType.TEXT_OUTPUT, text="test"), sync=True)
        await emitter.emit(AudioEvent(type=EventType.AUDIO_OUTPUT_CHUNK), sync=True)
        
        assert len(all_events) == 3
        assert EventType.CONNECTION_ESTABLISHED in all_events
        assert EventType.TEXT_OUTPUT in all_events
        assert EventType.AUDIO_OUTPUT_CHUNK in all_events
    
    @pytest.mark.asyncio
    async def test_handler_removal(self):
        """Test removing handlers"""
        emitter = EventEmitter()
        handler_called = False
        
        def handler(event: Event):
            nonlocal handler_called
            handler_called = True
        
        # Register and get handler ID
        handler_id = emitter.on(EventType.ERROR_GENERAL, handler)
        
        # Remove handler
        removed = emitter.off(handler_id)
        assert removed
        
        # Emit event - handler should not be called
        await emitter.emit(ErrorEvent(type=EventType.ERROR_GENERAL), sync=True)
        assert not handler_called
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in event handlers"""
        emitter = EventEmitter()
        error_handler_called = False
        
        def failing_handler(event: Event):
            raise ValueError("Handler error")
        
        def error_handler(error: Exception, event: Event):
            nonlocal error_handler_called
            error_handler_called = True
            assert isinstance(error, ValueError)
            assert str(error) == "Handler error"
        
        emitter.on(EventType.TEXT_OUTPUT, failing_handler)
        emitter.on_error(error_handler)
        
        # Should not raise
        await emitter.emit(TextEvent(type=EventType.TEXT_OUTPUT, text="test"), sync=True)
        
        assert error_handler_called
    
    @pytest.mark.asyncio
    async def test_event_history(self):
        """Test event history tracking"""
        emitter = EventEmitter(enable_history=True, history_size=5)
        
        # Emit several events
        for i in range(10):
            await emitter.emit(Event(type=EventType.METRICS_UPDATED, data={"index": i}))
        
        # Get history
        history = emitter.get_history()
        assert len(history) == 5  # Limited by history_size
        
        # Should have the last 5 events
        for i, event in enumerate(history):
            assert event.data["index"] == i + 5
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test temporary handler with context manager"""
        emitter = EventEmitter()
        events_received = []
        
        def handler(event: Event):
            events_received.append(event)
        
        # Use context manager for temporary handler
        async with emitter.listen(EventType.CONNECTION_CLOSED, handler) as handler_id:
            # Handler is active
            await emitter.emit(Event(type=EventType.CONNECTION_CLOSED), sync=True)
            assert len(events_received) == 1
        
        # Handler should be removed after context
        await emitter.emit(Event(type=EventType.CONNECTION_CLOSED), sync=True)
        assert len(events_received) == 1  # No new events
    
    @pytest.mark.asyncio
    async def test_metrics(self):
        """Test emitter metrics"""
        emitter = EventEmitter()
        
        # Register some handlers
        emitter.on(EventType.TEXT_OUTPUT, lambda e: None)
        emitter.on(EventType.AUDIO_OUTPUT_CHUNK, lambda e: None)
        
        # Emit some events
        await emitter.emit(TextEvent(type=EventType.TEXT_OUTPUT, text="test"), sync=True)
        await emitter.emit(AudioEvent(type=EventType.AUDIO_OUTPUT_CHUNK), sync=True)
        
        metrics = emitter.get_metrics()
        assert metrics["events_emitted"] == 2
        assert metrics["events_handled"] == 2
        assert metrics["handler_count"] == 2
    
    @pytest.mark.asyncio
    async def test_multiple_event_types(self):
        """Test handler for multiple event types"""
        emitter = EventEmitter()
        events_received = []
        
        def handler(event: Event):
            events_received.append(event.type)
        
        # Register for multiple event types
        emitter.on([EventType.AUDIO_INPUT_STARTED, EventType.AUDIO_INPUT_STOPPED], handler)
        
        # Emit events
        await emitter.emit(Event(type=EventType.AUDIO_INPUT_STARTED), sync=True)
        await emitter.emit(Event(type=EventType.AUDIO_INPUT_STOPPED), sync=True)
        await emitter.emit(Event(type=EventType.AUDIO_OUTPUT_STARTED), sync=True)  # Not registered
        
        assert len(events_received) == 2
        assert EventType.AUDIO_INPUT_STARTED in events_received
        assert EventType.AUDIO_INPUT_STOPPED in events_received


class TestEventTypes:
    """Test specific event type classes"""
    
    def test_audio_event(self):
        """Test AudioEvent creation"""
        event = AudioEvent(
            type=EventType.AUDIO_OUTPUT_CHUNK,
            audio_data=b"test audio",
            sample_rate=24000,
            channels=1,
            duration_ms=100
        )
        
        assert event.audio_data == b"test audio"
        assert event.sample_rate == 24000
        assert event.channels == 1
        assert event.duration_ms == 100
    
    def test_text_event(self):
        """Test TextEvent creation"""
        event = TextEvent(
            type=EventType.TEXT_OUTPUT,
            text="Hello world",
            is_partial=True,
            language="en"
        )
        
        assert event.text == "Hello world"
        assert event.is_partial is True
        assert event.language == "en"
    
    def test_connection_event(self):
        """Test ConnectionEvent creation"""
        event = ConnectionEvent(
            type=EventType.CONNECTION_ESTABLISHED,
            connection_id="conn123",
            retry_count=2,
            latency_ms=45.5
        )
        
        assert event.connection_id == "conn123"
        assert event.retry_count == 2
        assert event.latency_ms == 45.5
    
    def test_function_call_event(self):
        """Test FunctionCallEvent creation"""
        event = FunctionCallEvent(
            type=EventType.FUNCTION_CALL_INVOKED,
            function_name="get_weather",
            arguments={"city": "London"},
            call_id="call123"
        )
        
        assert event.function_name == "get_weather"
        assert event.arguments == {"city": "London"}
        assert event.call_id == "call123"
    
    def test_error_event(self):
        """Test ErrorEvent creation"""
        error = ValueError("Test error")
        event = ErrorEvent(
            type=EventType.ERROR_GENERAL,
            error=error,
            error_code="E001",
            recoverable=True,
            retry_after_seconds=5.0
        )
        
        assert event.error == error
        assert event.error_code == "E001"
        assert event.error_message == "Test error"
        assert event.recoverable is True
        assert event.retry_after_seconds == 5.0
    
    def test_event_serialization(self):
        """Test event serialization to dict"""
        event = TextEvent(
            type=EventType.TEXT_OUTPUT,
            text="Test message",
            source="test_source"
        )
        
        event_dict = event.to_dict()
        assert event_dict["type"] == EventType.TEXT_OUTPUT.value
        assert event_dict["source"] == "test_source"
        assert "timestamp" in event_dict
        assert "data" in event_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])