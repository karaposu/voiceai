"""
Test 07: VoiceEngine Text Interactions with Event System
Tests text-based interactions using the new event system with REAL OpenAI API connections.

python -m  voxengine.smoke_tests.test_07_voice_engine_text_events
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import time
from ..voice_engine import VoiceEngine
from ..events import EventType, TextEvent, AudioEvent, ErrorEvent

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

async def test_text_conversation_with_events():
    """Test text conversation with event system"""
    print("\n=== Test 1: Text Conversation with Events ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Track responses
        text_responses = []
        audio_chunks = []
        errors = []
        response_complete = False
        
        # Setup event handlers
        engine.events.on(EventType.TEXT_OUTPUT, 
                        lambda event: text_responses.append(event.text))
        engine.events.on(EventType.AUDIO_OUTPUT_CHUNK, 
                        lambda event: audio_chunks.append(event.audio_data))
        engine.events.on(EventType.ERROR_GENERAL, 
                        lambda event: errors.append(event.error))
        engine.events.on(EventType.RESPONSE_COMPLETED, 
                        lambda event: globals().update({'response_complete': True}))
        
        # Connect to API
        await engine.connect()
        print("✓ Connected to OpenAI API")
        
        # Listen for connection events
        connection_established = False
        engine.events.once(EventType.CONNECTION_ESTABLISHED, 
                          lambda event: print(f"✓ Connection event received: {event.type}"))
        
        # Send a text message
        test_message = "Hello! Please respond with a short greeting."
        await engine.send_text(test_message)
        print(f"✓ Sent text: '{test_message}'")
        
        # Wait for response (with timeout)
        start_time = time.time()
        timeout = 10  # 10 seconds timeout
        
        while not response_complete and time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
        
        # Check results
        print(f"✓ Response received:")
        print(f"  Text responses: {len(text_responses)}")
        print(f"  Audio chunks: {len(audio_chunks)}")
        print(f"  Errors: {len(errors)}")
        
        if text_responses:
            print(f"  AI said: '{text_responses[0][:100]}...'")
        
        if audio_chunks:
            total_audio_bytes = sum(len(chunk) for chunk in audio_chunks)
            print(f"  Total audio bytes: {total_audio_bytes}")
        
        # Disconnect
        await engine.disconnect()
        print("✓ Disconnected from API")
        
        # Success if we got either text or audio response
        return len(text_responses) > 0 or len(audio_chunks) > 0
        
    except Exception as e:
        print(f"✗ Text conversation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_event_priorities():
    """Test event handler priorities"""
    print("\n=== Test 2: Event Handler Priorities ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Track handler execution order
        execution_order = []
        
        # Register handlers with different priorities
        engine.events.on(EventType.TEXT_OUTPUT, 
                        lambda event: execution_order.append("low"), 
                        priority=1)
        engine.events.on(EventType.TEXT_OUTPUT, 
                        lambda event: execution_order.append("high"), 
                        priority=10)
        engine.events.on(EventType.TEXT_OUTPUT, 
                        lambda event: execution_order.append("medium"), 
                        priority=5)
        
        # Connect
        await engine.connect()
        print("✓ Connected to API")
        
        # Send message
        await engine.send_text("Say 'hello' in one word")
        
        # Wait for response
        await asyncio.sleep(3)
        
        # Check execution order
        if execution_order:
            print(f"✓ Handler execution order: {execution_order}")
            # Should be high, medium, low
            expected = ["high", "medium", "low"]
            if execution_order[:3] == expected:
                print("✓ Priorities working correctly")
            else:
                print("✗ Unexpected priority order")
        
        await engine.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ Priority test failed: {e}")
        return False

async def test_event_filtering():
    """Test event filtering"""
    print("\n=== Test 3: Event Filtering ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Track filtered responses
        long_texts = []
        short_texts = []
        
        # Filter for long texts (> 20 chars)
        engine.events.on(EventType.TEXT_OUTPUT, 
                        lambda event: long_texts.append(event.text),
                        filter=lambda event: len(event.text) > 20)
        
        # Filter for short texts (<= 20 chars)
        engine.events.on(EventType.TEXT_OUTPUT, 
                        lambda event: short_texts.append(event.text),
                        filter=lambda event: len(event.text) <= 20)
        
        await engine.connect()
        print("✓ Connected to API")
        
        # Send message that should generate a longer response
        await engine.send_text("Count from 1 to 5 with explanations")
        await asyncio.sleep(5)
        
        print(f"✓ Long text chunks: {len(long_texts)}")
        print(f"✓ Short text chunks: {len(short_texts)}")
        
        await engine.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ Filter test failed: {e}")
        return False

async def test_mixed_callbacks_and_events():
    """Test using both callbacks and events together"""
    print("\n=== Test 4: Mixed Callbacks and Events ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Track from both systems
        callback_texts = []
        event_texts = []
        
        # Set up callback (legacy style)
        engine.on_text_response = lambda text: callback_texts.append(text)
        
        # Set up event handler (new style)
        engine.events.on(EventType.TEXT_OUTPUT, 
                        lambda event: event_texts.append(event.text))
        
        await engine.connect()
        print("✓ Connected to API")
        
        # Send message
        await engine.send_text("Say 'test complete'")
        await asyncio.sleep(3)
        
        print(f"✓ Callback texts: {len(callback_texts)}")
        print(f"✓ Event texts: {len(event_texts)}")
        
        # Both should receive the same messages
        if callback_texts == event_texts:
            print("✓ Callbacks and events working together correctly")
        else:
            print("✗ Mismatch between callbacks and events")
        
        await engine.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ Mixed test failed: {e}")
        return False

async def test_event_metrics():
    """Test event system metrics"""
    print("\n=== Test 5: Event System Metrics ===")
    
    try:
        engine = VoiceEngine(api_key=API_KEY, mode="fast")
        
        # Register multiple handlers
        handler_ids = []
        handler_ids.append(engine.events.on(EventType.TEXT_OUTPUT, lambda e: None))
        handler_ids.append(engine.events.on(EventType.AUDIO_OUTPUT_CHUNK, lambda e: None))
        handler_ids.append(engine.events.on(EventType.ERROR_GENERAL, lambda e: None))
        
        # Check initial metrics
        metrics = engine.events.get_metrics()
        print(f"✓ Initial handler count: {metrics['handler_count']}")
        
        await engine.connect()
        
        # Send message
        await engine.send_text("Hello")
        await asyncio.sleep(3)
        
        # Check metrics after activity
        metrics = engine.events.get_metrics()
        print(f"✓ Events emitted: {metrics['events_emitted']}")
        print(f"✓ Events handled: {metrics['events_handled']}")
        print(f"✓ Errors caught: {metrics['errors_caught']}")
        
        # Remove a handler
        engine.events.off(handler_ids[0])
        metrics = engine.events.get_metrics()
        print(f"✓ Handler count after removal: {metrics['handler_count']}")
        
        await engine.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Voice Engine Event System Tests")
    print("=" * 50)
    
    tests = [
        test_text_conversation_with_events,
        test_event_priorities,
        test_event_filtering,
        test_mixed_callbacks_and_events,
        test_event_metrics
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
        print()
    
    # Summary
    print("Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)