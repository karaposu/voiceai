#!/usr/bin/env python3
"""
Provider Integration Test

Tests the integration of the new provider abstraction with VoiceEngine.


python -m voicechatengine.smoke_tests.test_provider_integration
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from voicechatengine.voice_engine import VoiceEngine, VoiceEngineConfig


async def test_provider_mode():
    """Test VoiceEngine with provider mode."""
    print("\n=== Testing VoiceEngine with Provider Mode ===")
    
    # Create config for provider mode
    config = VoiceEngineConfig(
        api_key="test-key",
        provider="mock",  # Use mock provider for testing
        mode="provider",  # Use new provider mode
        vad_enabled=True,
        voice="mock-voice-1"
    )
    
    # Create engine with the config (not just api_key)
    engine = VoiceEngine(config=config)
    
    try:
        # Connect
        print("Connecting...")
        await engine.connect()
        print("✅ Connected successfully")
        
        # Check state
        state = engine.get_state()
        print(f"✅ Engine state: {state}")
        
        # Send text
        print("Sending text message...")
        await engine.send_text("Hello from provider mode!")
        print("✅ Text sent successfully")
        
        # Wait a bit for response
        await asyncio.sleep(1)
        
        # Get metrics
        metrics = engine.get_metrics()
        print(f"✅ Metrics: {metrics}")
        
        # Disconnect
        print("Disconnecting...")
        await engine.disconnect()
        print("✅ Disconnected successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_provider_registry():
    """Test provider registry."""
    print("\n=== Testing Provider Registry ===")
    
    from voicechatengine.providers import get_registry
    
    registry = get_registry()
    
    # List providers
    providers = registry.list_providers()
    print(f"Available providers: {providers}")
    assert "openai" in providers
    assert "mock" in providers
    print("✅ Registry contains expected providers")
    
    # Get capabilities
    all_caps = registry.get_all_capabilities()
    for name, caps in all_caps.items():
        print(f"\n{name} capabilities:")
        print(f"  - Features: {[f.value for f in caps.features]}")
        print(f"  - Voices: {caps.available_voices}")
        print(f"  - Sample rates: {caps.supported_sample_rates}")
    
    print("\n✅ Provider registry working correctly")
    return True


async def test_provider_adapter():
    """Test provider adapter with existing protocol."""
    print("\n=== Testing Provider Adapter ===")
    
    from voicechatengine.providers import get_registry, MockConfig
    from voicechatengine.core.provider_protocol import ProviderConfig as CoreProviderConfig
    from voicechatengine.core.stream_protocol import StreamConfig, AudioFormat
    
    registry = get_registry()
    provider_adapter = registry.get("mock")
    
    # Validate config
    config = CoreProviderConfig(
        api_key="test-key",
        metadata={"voice": "test-voice"}
    )
    
    valid, msg = await provider_adapter.validate_config(config)
    assert valid
    print("✅ Config validation passed")
    
    # Create session
    stream_config = StreamConfig(
        provider="mock",
        mode="both",
        audio_format=AudioFormat(
            sample_rate=16000,
            channels=1,
            bit_depth=16
        ),
        enable_vad=True
    )
    
    session = await provider_adapter.create_session(config, stream_config)
    assert session.is_active
    print("✅ Session created successfully")
    
    # Send text
    await session.send_text("Test message")
    print("✅ Text sent through adapter")
    
    # Get usage
    usage = session.get_usage()
    print(f"✅ Usage tracked: {usage}")
    
    # End session
    final_usage = await session.end_session()
    print(f"✅ Session ended. Final usage: {final_usage}")
    
    return True


async def run_all_tests():
    """Run all integration tests."""
    print("=" * 50)
    print("PROVIDER INTEGRATION TESTS")
    print("=" * 50)
    
    tests = [
        test_provider_registry,
        test_provider_adapter,
        test_provider_mode
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test '{test_func.__name__}' crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed} tests passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)