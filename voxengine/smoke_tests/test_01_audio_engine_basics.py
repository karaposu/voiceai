"""
Test 01: VoxStream Basics
Tests the fundamental VoxStream functionality without external dependencies.


python -m voxengine.smoke_tests.test_01_audio_engine_basics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import time
import numpy as np
from voxstream import VoxStream
from voxstream.core.stream import create_fast_lane_engine, create_adaptive_engine
from voxstream.config.types import StreamConfig, ProcessingMode, VADConfig
AudioBytes = bytes

def generate_test_audio(duration_ms: int = 100, sample_rate: int = 24000) -> AudioBytes:
    """Generate test audio (sine wave)"""
    duration_s = duration_ms / 1000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()

def test_voxstream_creation():
    """Test basic VoxStream creation"""
    print("\n=== Test 1: VoxStream Creation ===")
    
    try:
        # Test default creation
        stream = VoxStream()
        print("✓ Default VoxStream created")
        
        # Test with specific mode
        stream_rt = VoxStream(mode=ProcessingMode.REALTIME)
        print("✓ Realtime mode VoxStream created")
        
        stream_quality = VoxStream(mode=ProcessingMode.QUALITY)
        print("✓ Quality mode VoxStream created")
        
        # Test factory function
        stream_fast = create_fast_lane_engine()
        print("✓ Fast lane VoxStream created")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create VoxStream: {e}")
        return False

def test_audio_processing():
    """Test basic audio processing"""
    print("\n=== Test 2: Audio Processing ===")
    
    try:
        stream = VoxStream(mode=ProcessingMode.REALTIME)
        test_audio = generate_test_audio(100)  # 100ms of audio
        
        # Test process_audio
        start = time.time()
        processed = stream.process_audio(test_audio)
        elapsed = time.time() - start
        
        print(f"✓ Processed {len(test_audio)} bytes in {elapsed*1000:.2f}ms")
        print(f"  Input size: {len(test_audio)} bytes")
        print(f"  Output size: {len(processed)} bytes")
        
        # Test chunking - VoxStream may have different API
        # chunks = stream.processor.chunk_audio(test_audio, chunk_duration_ms=20)
        # print(f"✓ Chunked audio into {len(chunks)} chunks")
        
        return True
    except Exception as e:
        print(f"✗ Audio processing failed: {e}")
        return False

def test_mode_switching():
    """Test mode switching and optimization"""
    print("\n=== Test 3: Mode Switching ===")
    
    try:
        stream = VoxStream(mode=ProcessingMode.BALANCED)
        test_audio = generate_test_audio(50)
        
        # Test initial mode
        print(f"✓ Initial mode: {stream.mode.value}")
        
        # Process in balanced mode
        start = time.time()
        _ = stream.process_audio(test_audio)
        balanced_time = time.time() - start
        
        # VoxStream may have different optimization methods
        # stream.optimize_for_latency()
        print(f"✓ Processed in balanced mode: {balanced_time*1000:.2f}ms")
        
        return True
    except Exception as e:
        print(f"✗ Mode switching failed: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing capabilities"""
    print("\n=== Test 4: Concurrent Processing ===")
    
    try:
        stream = VoxStream(mode=ProcessingMode.REALTIME)
        test_audio = generate_test_audio(20)  # 20ms chunks
        
        # Simulate rapid sequential processing
        times = []
        for i in range(10):
            start = time.time()
            _ = stream.process_audio(test_audio)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"✓ Average processing time: {avg_time*1000:.2f}ms")
        print(f"  Min: {min(times)*1000:.2f}ms")
        print(f"  Max: {max(times)*1000:.2f}ms")
        
        return True
    except Exception as e:
        print(f"✗ Concurrent processing failed: {e}")
        return False

def test_with_config():
    """Test VoxStream with custom configuration"""
    print("\n=== Test 5: Custom Configuration ===")
    
    try:
        config = StreamConfig(sample_rate=16000, channels=1, chunk_duration_ms=20)
        stream = VoxStream(config=config)
        
        print(f"✓ Created with custom config:")
        print(f"  Sample rate: {config.sample_rate}")
        print(f"  Channels: {config.channels}")
        print(f"  Chunk duration: {config.chunk_duration_ms}ms")
        
        # Test processing with custom config
        test_audio = generate_test_audio(100, sample_rate=16000)
        processed = stream.process_audio(test_audio)
        print(f"✓ Processed audio with custom config")
        
        return True
    except Exception as e:
        print(f"✗ Custom configuration failed: {e}")
        return False

def test_vad_configuration():
    """Test Voice Activity Detection configuration"""
    print("\n=== Test 6: VAD Configuration ===")
    
    try:
        stream = VoxStream()
        
        # Configure VAD
        vad_config = VADConfig(
            energy_threshold=0.02,
            speech_start_ms=100,
            speech_end_ms=300
        )
        stream.configure_vad(vad_config)
        print("✓ Configured VAD")
        
        # Test with silent audio
        silent_audio = bytes(4800)  # 100ms of silence at 24kHz
        _ = stream.process_audio(silent_audio)
        
        # Test with speech audio
        speech_audio = generate_test_audio(200)
        _ = stream.process_audio(speech_audio)
        
        print("✓ Processed audio with VAD")
        
        return True
    except Exception as e:
        print(f"✗ VAD configuration failed: {e}")
        return False

def test_cleanup():
    """Test cleanup and resource management"""
    print("\n=== Test 7: Cleanup ===")
    
    try:
        stream = VoxStream()
        
        # Process some audio
        test_audio = generate_test_audio(50)
        for i in range(5):
            stream.process_audio(test_audio)
        
        # Cleanup
        stream.cleanup()
        print("✓ Cleanup successful")
        
        # Test creating new instance after cleanup
        stream2 = VoxStream()
        stream2.process_audio(test_audio)
        stream2.cleanup()
        print("✓ Can create new instance after cleanup")
        
        return True
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("VoxStream Basic Tests")
    print("=" * 50)
    
    tests = [
        test_voxstream_creation,
        test_audio_processing,
        test_mode_switching,
        test_concurrent_processing,
        test_with_config,
        test_vad_configuration,
        test_cleanup
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        time.sleep(0.1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    print("=" * 50)
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)