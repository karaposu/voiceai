#!/usr/bin/env python3
"""
Fast Lane Provider Integration Test

Tests the new provider-based fast lane implementation with real OpenAI API.

python -m voxengine.smoke_tests.test_fast_lane_provider
"""

import asyncio
import sys
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from voxengine.voice_engine import VoiceEngine, VoiceEngineConfig
from voxengine.base_engine import BaseEngine
from voxengine.strategies.fast_lane_strategy_v2 import FastLaneStrategyV2
from voxengine.strategies.base_strategy import EngineConfig
from voxengine.providers import get_registry
from voxengine.core.stream_protocol import StreamEventType

# Get API key from environment
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)


async def test_fast_lane_v2_direct():
    """Test FastLaneStrategyV2 directly with real OpenAI."""
    print("\n=== Testing FastLaneStrategyV2 Direct ===")
    
    engine = BaseEngine()
    
    # Create FastLaneStrategyV2 using create_strategy
    engine.create_strategy("fast")
    
    # Track events
    events_received = []
    text_responses = []
    audio_chunks = []
    
    def track_event(event_type):
        def handler(event):
            events_received.append((event_type, event))
            if event_type == StreamEventType.AUDIO_OUTPUT_CHUNK:
                audio_chunks.append(event.data.get("audio", b""))
            elif event_type == StreamEventType.TEXT_OUTPUT_CHUNK:
                text_responses.append(event.data.get("text", ""))
        return handler
    
    try:
        # Initialize with real OpenAI first
        config = EngineConfig(
            api_key=API_KEY,
            provider="openai"
        )
        await engine.initialize_strategy(config)
        
        # Setup handlers after initialization
        engine.setup_event_handlers({
            StreamEventType.STREAM_STARTED: track_event(StreamEventType.STREAM_STARTED),
            StreamEventType.AUDIO_OUTPUT_CHUNK: track_event(StreamEventType.AUDIO_OUTPUT_CHUNK),
            StreamEventType.TEXT_OUTPUT_CHUNK: track_event(StreamEventType.TEXT_OUTPUT_CHUNK),
            StreamEventType.STREAM_ENDED: track_event(StreamEventType.STREAM_ENDED),
            StreamEventType.STREAM_ERROR: track_event(StreamEventType.STREAM_ERROR)
        })
        print("✅ Strategy initialized with OpenAI provider")
        
        # Connect
        await engine.connect()
        print("✅ Connected to OpenAI")
        
        # Send text
        await engine.send_text("Hello! Please say 'Hi there!' back to me.")
        print("✅ Sent text message")
        
        # Wait for response
        await asyncio.sleep(5)
        
        # Check results
        print(f"\n📊 Results:")
        print(f"  Events received: {len(events_received)}")
        print(f"  Text responses: {len(text_responses)}")
        print(f"  Audio chunks: {len(audio_chunks)}")
        
        if text_responses:
            print(f"  AI said: '{' '.join(text_responses)}'")
        
        if audio_chunks:
            total_audio = sum(len(chunk) for chunk in audio_chunks)
            print(f"  Total audio bytes: {total_audio}")
        
        # Disconnect
        await engine.disconnect()
        print("✅ Disconnected")
        
        return len(audio_chunks) > 0 or len(text_responses) > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_voice_engine_provider_mode():
    """Test VoiceEngine with provider mode using real OpenAI."""
    print("\n=== Testing VoiceEngine Provider Mode ===")
    
    # Create config for provider mode
    config = VoiceEngineConfig(
        api_key=API_KEY,
        provider="openai",  # Use real OpenAI
        mode="provider",    # Use provider mode
        vad_enabled=True,
        voice="alloy"
    )
    
    engine = VoiceEngine(config=config)
    
    # Track responses
    text_responses = []
    audio_chunks = []
    errors = []
    response_complete = False
    
    def on_text(text):
        text_responses.append(text)
        print(f"  Text: {text}")
    
    def on_audio(audio):
        audio_chunks.append(audio)
    
    def on_error(error):
        errors.append(error)
        print(f"  Error: {error}")
    
    def on_done():
        nonlocal response_complete
        response_complete = True
    
    # Setup handlers
    engine.on_text_response = on_text
    engine.on_audio_response = on_audio
    engine.on_error = on_error
    engine.on_response_done = on_done
    
    try:
        # Connect
        print("Connecting to OpenAI...")
        await engine.connect()
        print("✅ Connected successfully")
        
        # Send text
        test_message = "Count from 1 to 3."
        print(f"\nSending: '{test_message}'")
        await engine.send_text(test_message)
        
        # Wait for response
        start_time = time.time()
        timeout = 10
        while not response_complete and time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
        
        # Check results
        print(f"\n📊 Results:")
        print(f"  Text responses: {len(text_responses)}")
        print(f"  Audio chunks: {len(audio_chunks)}")
        print(f"  Errors: {len(errors)}")
        
        if audio_chunks:
            total_audio = sum(len(chunk) for chunk in audio_chunks)
            print(f"  Total audio bytes: {total_audio}")
        
        # Get metrics
        metrics = engine.get_metrics()
        print(f"\n📈 Metrics:")
        print(f"  Total interactions: {metrics.get('total_interactions', 0)}")
        print(f"  Audio chunks sent: {metrics.get('audio_chunks_sent', 0)}")
        print(f"  Audio chunks received: {metrics.get('audio_chunks_received', 0)}")
        
        # Disconnect
        await engine.disconnect()
        print("\n✅ Disconnected")
        
        return len(text_responses) > 0 or len(audio_chunks) > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fast_lane_interruption():
    """Test interruption with FastLaneStrategyV2."""
    print("\n=== Testing Fast Lane Interruption ===")
    
    engine = BaseEngine()
    engine.create_strategy("fast")
    
    # Track state
    response_started = False
    chunks_before_interrupt = 0
    chunks_after_interrupt = 0
    interrupted = False
    
    def on_audio(event):
        nonlocal response_started, chunks_before_interrupt, chunks_after_interrupt
        response_started = True
        if not interrupted:
            chunks_before_interrupt += 1
        else:
            chunks_after_interrupt += 1
    
    try:
        # Initialize first
        config = EngineConfig(api_key=API_KEY, provider="openai")
        await engine.initialize_strategy(config)
        
        # Setup handlers after initialization
        engine.setup_event_handlers({
            StreamEventType.AUDIO_OUTPUT_CHUNK: on_audio
        })
        await engine.connect()
        print("✅ Connected")
        
        # Send message for long response
        await engine.send_text("Count slowly from 1 to 20, pausing between each number.")
        print("✅ Sent message for long response")
        
        # Wait for response to start
        await asyncio.sleep(2)
        
        if response_started:
            # Interrupt
            await engine.interrupt()
            interrupted = True
            print(f"✅ Interrupted after {chunks_before_interrupt} chunks")
            
            # Wait a bit to ensure no more chunks
            await asyncio.sleep(2)
            print(f"  Chunks after interrupt: {chunks_after_interrupt}")
        else:
            print("⚠️  Response didn't start in time")
        
        # Disconnect
        await engine.disconnect()
        
        return chunks_after_interrupt == 0  # Success if no chunks after interrupt
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_provider_capabilities():
    """Test provider capabilities for fast lane."""
    print("\n=== Testing Provider Capabilities ===")
    
    registry = get_registry()
    
    # Check OpenAI capabilities
    all_caps = registry.get_all_capabilities()
    if "openai" in all_caps:
        caps = all_caps["openai"]
        print("OpenAI Provider Capabilities:")
        print(f"  Features: {[f.value for f in caps.features]}")
        print(f"  Sample rates: {caps.supported_sample_rates}")
        print(f"  Voices: {caps.available_voices[:3]}...")  # First 3 voices
    
    # Get provider adapter
    provider_adapter = registry.get("openai")
    print(f"\n✅ Provider adapter type: {type(provider_adapter).__name__}")
    
    # Test config validation
    from voxengine.core.provider_protocol import ProviderConfig as CoreProviderConfig
    
    config = CoreProviderConfig(
        api_key=API_KEY,
        metadata={"voice": "alloy"}
    )
    
    valid, msg = await provider_adapter.validate_config(config)
    print(f"✅ Config validation: {valid} - {msg}")
    
    return valid


async def test_fast_vs_provider_mode():
    """Compare fast mode vs provider mode."""
    print("\n=== Testing Fast vs Provider Mode ===")
    
    results = {}
    
    # Test fast mode (original)
    print("\n1️⃣ Testing FAST mode:")
    fast_engine = VoiceEngine(api_key=API_KEY, mode="fast")
    
    try:
        await fast_engine.connect()
        
        start_time = time.time()
        await fast_engine.send_text("Say 'test'")
        await asyncio.sleep(3)  # Wait for response
        fast_time = time.time() - start_time
        
        await fast_engine.disconnect()
        results['fast'] = {'success': True, 'time': fast_time}
        print(f"✅ Fast mode completed in {fast_time:.2f}s")
    except Exception as e:
        results['fast'] = {'success': False, 'error': str(e)}
        print(f"❌ Fast mode error: {e}")
    
    # Test provider mode
    print("\n2️⃣ Testing PROVIDER mode:")
    provider_config = VoiceEngineConfig(
        api_key=API_KEY,
        provider="openai",
        mode="provider"
    )
    provider_engine = VoiceEngine(config=provider_config)
    
    try:
        await provider_engine.connect()
        
        start_time = time.time()
        await provider_engine.send_text("Say 'test'")
        await asyncio.sleep(3)  # Wait for response
        provider_time = time.time() - start_time
        
        await provider_engine.disconnect()
        results['provider'] = {'success': True, 'time': provider_time}
        print(f"✅ Provider mode completed in {provider_time:.2f}s")
    except Exception as e:
        results['provider'] = {'success': False, 'error': str(e)}
        print(f"❌ Provider mode error: {e}")
    
    # Compare results
    print("\n📊 Comparison:")
    for mode, result in results.items():
        if result['success']:
            print(f"  {mode}: ✅ Success ({result['time']:.2f}s)")
        else:
            print(f"  {mode}: ❌ Failed - {result['error']}")
    
    return all(r['success'] for r in results.values())


async def run_all_tests():
    """Run all fast lane provider tests."""
    print("=" * 60)
    print("FAST LANE PROVIDER INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Provider Capabilities", test_provider_capabilities),
        ("FastLaneStrategyV2 Direct", test_fast_lane_v2_direct),
        ("VoiceEngine Provider Mode", test_voice_engine_provider_mode),
        ("Fast Lane Interruption", test_fast_lane_interruption),
        ("Fast vs Provider Mode", test_fast_vs_provider_mode)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} tests passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)