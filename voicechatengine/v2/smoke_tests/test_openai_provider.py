#!/usr/bin/env python3
"""
OpenAI Provider Smoke Tests

This script tests the OpenAI provider implementation.
Requires OPENAI_API_KEY environment variable to be set.
"""

import asyncio
import sys
import os
import time
from typing import List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from voicechatengine.v2.providers.openai_provider import OpenAIProvider, OpenAIConfig
from voicechatengine.v2.providers.base import MessageType, ConnectionState, ProviderEvent


def print_test(test_name: str, passed: bool, error: str = None):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if error:
        print(f"     Error: {error}")


async def test_openai_connection():
    """Test OpenAI provider connection."""
    print("\n=== Testing OpenAI Connection ===")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  SKIPPED - No OPENAI_API_KEY environment variable")
        return None
        
    # Test connection
    try:
        config = OpenAIConfig(api_key=api_key)
        provider = OpenAIProvider(config)
        
        # Check initial state
        assert provider.state == ConnectionState.DISCONNECTED
        print_test("Initial state", True)
        
        # Connect
        connected = await provider.connect()
        assert connected == True
        assert provider.state == ConnectionState.CONNECTED
        assert provider.session_id is not None
        print_test("Connection established", True)
        
        # Test capabilities
        caps = provider.get_capabilities()
        assert caps["provider"] == "openai"
        assert "gpt-4o-realtime-preview" in caps["models"]
        print_test("Capabilities check", True)
        
        # Disconnect
        await provider.disconnect()
        assert provider.state == ConnectionState.DISCONNECTED
        print_test("Disconnection", True)
        
        return True
        
    except Exception as e:
        print_test("OpenAI connection test", False, str(e))
        return False


async def test_openai_messages():
    """Test sending messages to OpenAI."""
    print("\n=== Testing OpenAI Messages ===")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  SKIPPED - No OPENAI_API_KEY environment variable")
        return None
        
    try:
        config = OpenAIConfig(api_key=api_key)
        provider = OpenAIProvider(config)
        await provider.connect()
        
        # Test text message
        await provider.send_text("Hello from smoke test!", role="user")
        print_test("Send text message", True)
        
        # Test session update
        await provider.update_session(
            voice="nova",
            temperature=0.7,
            instructions="You are a helpful assistant."
        )
        print_test("Update session", True)
        
        # Test creating response
        await provider.create_response(modalities=["text"])
        print_test("Create response", True)
        
        # Wait briefly for response
        await asyncio.sleep(2)
        
        # Test interruption
        await provider.interrupt()
        print_test("Interrupt response", True)
        
        await provider.disconnect()
        return True
        
    except Exception as e:
        print_test("OpenAI message test", False, str(e))
        if 'provider' in locals():
            await provider.disconnect()
        return False


async def test_openai_events():
    """Test receiving events from OpenAI."""
    print("\n=== Testing OpenAI Events ===")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  SKIPPED - No OPENAI_API_KEY environment variable")
        return None
        
    try:
        config = OpenAIConfig(api_key=api_key)
        provider = OpenAIProvider(config)
        await provider.connect()
        
        events_received = []
        
        # Collect initial events
        start_time = time.time()
        async for event in provider.events():
            events_received.append(event)
            if len(events_received) >= 1 or time.time() - start_time > 2:
                break
                
        assert len(events_received) > 0
        print_test("Event reception", True)
        
        # Test event handler
        handler_called = {"count": 0}
        
        async def test_handler(event: ProviderEvent):
            handler_called["count"] += 1
            
        provider.on_event("session.created", test_handler)
        
        # Should have already received session.created
        assert handler_called["count"] > 0
        print_test("Event handler", True)
        
        await provider.disconnect()
        return True
        
    except Exception as e:
        print_test("OpenAI event test", False, str(e))
        if 'provider' in locals():
            await provider.disconnect()
        return False


async def test_openai_audio_format():
    """Test OpenAI audio format requirements."""
    print("\n=== Testing OpenAI Audio Format ===")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  SKIPPED - No OPENAI_API_KEY environment variable")
        return None
        
    try:
        config = OpenAIConfig(api_key=api_key)
        provider = OpenAIProvider(config)
        
        audio_format = provider.get_audio_format()
        assert audio_format.encoding == "pcm16"
        assert audio_format.sample_rate == 24000
        assert audio_format.channels == 1
        print_test("Audio format check", True)
        
        return True
        
    except Exception as e:
        print_test("Audio format test", False, str(e))
        return False


async def run_all_tests():
    """Run all OpenAI provider tests."""
    print("=" * 50)
    print("OPENAI PROVIDER SMOKE TESTS")
    print("=" * 50)
    
    tests = [
        ("Connection", test_openai_connection),
        ("Messages", test_openai_messages),
        ("Events", test_openai_events),
        ("Audio Format", test_openai_audio_format)
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result is None:
                skipped += 1
            elif result == True:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test suite '{test_name}' crashed: {e}")
            failed += 1
            
    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 50)
    
    if skipped > 0:
        print("\n⚠️  Some tests were skipped. Set OPENAI_API_KEY to run all tests.")
    
    return failed == 0


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)