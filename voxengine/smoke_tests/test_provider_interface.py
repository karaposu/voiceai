#!/usr/bin/env python3
"""
Provider Interface Smoke Tests

This script tests the basic functionality of the provider abstraction
without using any testing frameworks. Run directly with Python.

python -m voxengine.smoke_tests.test_provider_interface
"""

import asyncio
import sys
import os
import time
from typing import List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from voxengine.providers.mock_provider import MockProvider, MockConfig
from voxengine.providers.base import MessageType, ConnectionState, ProviderEvent


def print_test(test_name: str, passed: bool, error: str = None):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if error:
        print(f"     Error: {error}")


async def test_provider_lifecycle():
    """Test provider connection lifecycle."""
    print("\n=== Testing Provider Lifecycle ===")
    
    # Test 1: Create provider
    try:
        config = MockConfig(api_key="test_key", model="mock-gpt")
        provider = MockProvider(config)
        assert provider.state == ConnectionState.DISCONNECTED
        print_test("Provider creation", True)
    except Exception as e:
        print_test("Provider creation", False, str(e))
        return False
        
    # Test 2: Connect
    try:
        connected = await provider.connect()
        assert connected == True
        assert provider.state == ConnectionState.CONNECTED
        assert provider.session_id is not None
        print_test("Provider connection", True)
    except Exception as e:
        print_test("Provider connection", False, str(e))
        return False
        
    # Test 3: Check connected state
    try:
        assert provider.is_connected() == True
        print_test("Connected state check", True)
    except Exception as e:
        print_test("Connected state check", False, str(e))
        
    # Test 4: Disconnect
    try:
        await provider.disconnect()
        assert provider.state == ConnectionState.DISCONNECTED
        assert provider.session_id is None
        print_test("Provider disconnection", True)
    except Exception as e:
        print_test("Provider disconnection", False, str(e))
        return False
        
    return True


async def test_message_sending():
    """Test sending messages to provider."""
    print("\n=== Testing Message Sending ===")
    
    config = MockConfig(api_key="test_key")
    provider = MockProvider(config)
    await provider.connect()
    
    # Test 1: Send text message
    try:
        await provider.send_text("Hello, world!", role="user")
        history = provider.get_message_history()
        assert len(history) > 0
        assert any(msg["data"]["item"]["content"] == "Hello, world!" for msg in history)
        print_test("Send text message", True)
    except Exception as e:
        print_test("Send text message", False, str(e))
        
    # Test 2: Send audio
    try:
        audio_data = b"mock_audio_data_12345"
        await provider.send_audio(audio_data)
        buffer_size = provider.get_audio_buffer_size()
        assert buffer_size > 0
        print_test("Send audio data", True)
    except Exception as e:
        print_test("Send audio data", False, str(e))
        
    # Test 3: Create response
    try:
        await provider.create_response(modalities=["text", "audio"])
        await asyncio.sleep(0.1)  # Let response start
        print_test("Create response", True)
    except Exception as e:
        print_test("Create response", False, str(e))
        
    # Test 4: Interrupt response
    try:
        await provider.interrupt()
        print_test("Interrupt response", True)
    except Exception as e:
        print_test("Interrupt response", False, str(e))
        
    await provider.disconnect()
    return True


async def test_event_streaming():
    """Test event streaming from provider."""
    print("\n=== Testing Event Streaming ===")
    
    config = MockConfig(api_key="test_key", simulate_latency=False)
    provider = MockProvider(config)
    
    events_received = []
    
    # Test 1: Collect events during connection and message
    try:
        # Start event collection task
        async def collect_events():
            async for event in provider.events():
                events_received.append(event)
                
        event_task = asyncio.create_task(collect_events())
        
        # Connect (should generate session.created event)
        await provider.connect()
        
        # Send a message to trigger more events
        await provider.send_text("Test message", role="user")
        
        # Wait a bit for events to be collected
        await asyncio.sleep(0.5)
        
        # Cancel event collection
        event_task.cancel()
        try:
            await event_task
        except asyncio.CancelledError:
            pass
                
        assert len(events_received) > 0
        assert any(e.type == "session.created" for e in events_received)
        print_test("Event streaming", True)
    except Exception as e:
        print_test("Event streaming", False, str(e))
        
    # Test 2: Event handlers
    try:
        handler_called = {"count": 0}
        
        def test_handler(event: ProviderEvent):
            handler_called["count"] += 1
            
        provider.on_event("test_event", test_handler)
        
        # Emit test event
        await provider._queue_event(ProviderEvent(
            type="test_event",
            data={"test": True},
            timestamp=time.time(),
            provider="mock",
            session_id=provider.session_id
        ))
        
        await asyncio.sleep(0.1)
        assert handler_called["count"] > 0
        print_test("Event handlers", True)
    except Exception as e:
        print_test("Event handlers", False, str(e))
        
    await provider.disconnect()
    return True


async def test_error_handling():
    """Test error handling in provider."""
    print("\n=== Testing Error Handling ===")
    
    # Test 1: Connection error
    try:
        config = MockConfig(api_key="test_key", error_rate=1.0)  # Always fail
        provider = MockProvider(config)
        connected = await provider.connect()
        assert connected == False
        assert provider.state == ConnectionState.ERROR
        print_test("Connection error handling", True)
    except Exception as e:
        print_test("Connection error handling", False, str(e))
        
    # Test 2: Send without connection
    try:
        config = MockConfig(api_key="test_key")
        provider = MockProvider(config)
        error_caught = False
        
        try:
            await provider.send_text("Test")
        except Exception:
            error_caught = True
            
        assert error_caught
        print_test("Send without connection error", True)
    except Exception as e:
        print_test("Send without connection error", False, str(e))
        
    return True


async def test_provider_capabilities():
    """Test provider capability reporting."""
    print("\n=== Testing Provider Capabilities ===")
    
    config = MockConfig(api_key="test_key")
    provider = MockProvider(config)
    
    try:
        caps = provider.get_capabilities()
        assert "provider" in caps
        assert "models" in caps
        assert "modalities" in caps
        assert "features" in caps
        assert caps["provider"] == "mock"
        print_test("Get capabilities", True)
    except Exception as e:
        print_test("Get capabilities", False, str(e))
        
    return True


async def test_audio_format():
    """Test audio format specification."""
    print("\n=== Testing Audio Format ===")
    
    config = MockConfig(api_key="test_key")
    provider = MockProvider(config)
    
    try:
        audio_format = provider.get_audio_format()
        assert audio_format.encoding == "pcm16"
        assert audio_format.sample_rate == 16000
        assert audio_format.channels == 1
        print_test("Audio format", True)
    except Exception as e:
        print_test("Audio format", False, str(e))
        
    return True


async def test_session_update():
    """Test session configuration updates."""
    print("\n=== Testing Session Updates ===")
    
    config = MockConfig(api_key="test_key")
    provider = MockProvider(config)
    await provider.connect()
    
    try:
        await provider.update_session(
            voice="new-voice",
            temperature=0.9,
            instructions="Be helpful"
        )
        
        # Check message history
        history = provider.get_message_history()
        update_msg = next((m for m in history if m["type"] == "session.update"), None)
        assert update_msg is not None
        assert update_msg["data"]["session"]["voice"] == "new-voice"
        print_test("Session update", True)
    except Exception as e:
        print_test("Session update", False, str(e))
        
    await provider.disconnect()
    return True


async def run_all_tests():
    """Run all smoke tests."""
    print("=" * 50)
    print("PROVIDER INTERFACE SMOKE TESTS")
    print("=" * 50)
    
    tests = [
        ("Lifecycle", test_provider_lifecycle),
        ("Message Sending", test_message_sending),
        ("Event Streaming", test_event_streaming),
        ("Error Handling", test_error_handling),
        ("Capabilities", test_provider_capabilities),
        ("Audio Format", test_audio_format),
        ("Session Updates", test_session_update)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result != False:
                passed += len([1 for _ in range(1)])  # Count as 1 test suite
        except Exception as e:
            print(f"\n❌ Test suite '{test_name}' crashed: {e}")
            failed += 1
            
    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed} test suites passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)