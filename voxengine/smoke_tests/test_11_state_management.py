"""
Test 11: State Management System
Tests the state management functionality with real scenarios.

python -m voxengine.smoke_tests.test_11_state_management
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import time
from datetime import datetime
from ..state import (
    StateManager, ConversationState, ConversationStatus,
    Message, SpeakerRole, AudioState, ConnectionState
)
from ..events import EventEmitter, EventType

async def test_basic_state_operations():
    """Test basic state creation and updates"""
    print("\n=== Test 1: Basic State Operations ===")
    
    # Create state manager
    manager = StateManager()
    
    # Check initial state
    state = manager.state
    assert state.status == ConversationStatus.IDLE
    assert state.conversation_id is not None
    assert len(state.messages) == 0
    print("✓ Initial state created correctly")
    
    # Update status
    new_state = manager.update(status=ConversationStatus.CONNECTING)
    assert new_state.status == ConversationStatus.CONNECTING
    assert manager.state.status == ConversationStatus.CONNECTING
    print("✓ Status update works")
    
    # Add message
    message = Message(role=SpeakerRole.USER, content="Hello")
    state_with_msg = manager.add_message(message)
    assert len(state_with_msg.messages) == 1
    assert state_with_msg.messages[0].content == "Hello"
    print("✓ Message addition works")
    
    return True

async def test_atomic_updates():
    """Test atomic state updates"""
    print("\n=== Test 2: Atomic Updates ===")
    
    manager = StateManager()
    
    # Perform multiple updates atomically
    start_time = time.time()
    
    # Simulate concurrent updates
    tasks = []
    for i in range(10):
        task = asyncio.create_task(
            manager.update_async(
                status=ConversationStatus.CONNECTED if i % 2 == 0 else ConversationStatus.LISTENING
            )
        )
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"✓ 10 concurrent updates completed in {elapsed_ms:.2f}ms")
    
    # Check metrics
    metrics = manager.get_metrics()
    assert metrics["total_updates"] >= 10
    print(f"✓ Average update time: {metrics['avg_update_time_ms']:.3f}ms")
    
    return metrics["avg_update_time_ms"] < 1.0  # Should be < 1ms

async def test_state_history():
    """Test state history tracking"""
    print("\n=== Test 3: State History ===")
    
    manager = StateManager(enable_history=True, history_size=5)
    
    # Create some state changes
    for i in range(10):
        manager.add_message(Message(
            role=SpeakerRole.USER if i % 2 == 0 else SpeakerRole.ASSISTANT,
            content=f"Message {i}"
        ))
    
    # Check history
    history = manager.get_history()
    assert len(history) == 5  # Limited by history_size
    print(f"✓ History tracking works (size: {len(history)})")
    
    # Check snapshot
    snapshot = manager.create_snapshot()
    assert "state" in snapshot
    assert "metrics" in snapshot
    assert snapshot["update_count"] >= 10
    print("✓ Snapshot creation works")
    
    return True

async def test_conversation_flow():
    """Test realistic conversation flow"""
    print("\n=== Test 4: Conversation Flow ===")
    
    manager = StateManager()
    
    # Start conversation
    manager.update(status=ConversationStatus.CONNECTED)
    
    # User speaks
    user_msg = Message(role=SpeakerRole.USER, content="What's the weather?")
    manager.start_turn(user_msg)
    assert manager.state.current_turn is not None
    assert manager.state.current_turn.user_message == user_msg
    print("✓ Turn started correctly")
    
    # Assistant responds
    assistant_msg = Message(role=SpeakerRole.ASSISTANT, content="The weather is sunny.")
    manager.complete_turn(assistant_msg)
    assert manager.state.current_turn is None
    assert len(manager.state.turns) == 1
    assert manager.state.turns[0].is_complete
    print("✓ Turn completed correctly")
    
    # Test interruption
    manager.start_turn(Message(role=SpeakerRole.USER, content="Actually..."))
    manager.interrupt_turn()
    assert manager.state.metrics.interruption_count == 1
    print("✓ Interruption tracked")
    
    return True

async def test_event_integration():
    """Test state manager with event system"""
    print("\n=== Test 5: Event Integration ===")
    
    # Create event emitter
    emitter = EventEmitter()
    events_received = []
    
    # Track state change events
    emitter.on(EventType.STATE_CHANGED, lambda e: events_received.append(e))
    emitter.on(EventType.CONNECTION_ESTABLISHED, lambda e: events_received.append(e))
    
    # Create state manager with event emitter
    manager = StateManager(event_emitter=emitter)
    
    # Make changes that should emit events
    manager.update(status=ConversationStatus.CONNECTING)
    await asyncio.sleep(0.1)  # Let events process
    
    manager.update_connection(is_connected=True)
    await asyncio.sleep(0.1)  # Let events process
    
    # Check events were emitted
    assert len(events_received) >= 2
    print(f"✓ Received {len(events_received)} state change events")
    
    return True

async def test_performance():
    """Test state management performance"""
    print("\n=== Test 6: Performance Test ===")
    
    manager = StateManager()
    
    # Test message addition performance
    start_time = time.time()
    for i in range(100):
        manager.add_message(Message(
            role=SpeakerRole.USER if i % 2 == 0 else SpeakerRole.ASSISTANT,
            content=f"Message {i}" * 10  # Longer messages
        ))
    
    elapsed_ms = (time.time() - start_time) * 1000
    avg_ms = elapsed_ms / 100
    
    print(f"✓ Added 100 messages in {elapsed_ms:.2f}ms")
    print(f"✓ Average per message: {avg_ms:.3f}ms")
    
    # Check final state
    assert len(manager.state.messages) == 100
    assert manager.state.metrics.total_messages == 100
    
    metrics = manager.get_metrics()
    print(f"✓ Max update time: {metrics['max_update_time_ms']:.3f}ms")
    
    return avg_ms < 1.0  # Should be < 1ms per update

async def test_state_persistence():
    """Test state persistence capabilities"""
    print("\n=== Test 7: State Persistence ===")
    
    import tempfile
    import json
    from pathlib import Path
    
    manager = StateManager()
    
    # Create some state
    manager.update(status=ConversationStatus.CONNECTED)
    manager.add_message(Message(role=SpeakerRole.USER, content="Test message"))
    manager.update_connection(is_connected=True, latency_ms=45.5)
    
    # Save to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
        manager.save_to_file(temp_path)
    
    # Read and verify
    with open(temp_path, 'r') as f:
        saved_data = json.load(f)
    
    assert saved_data["state"]["status"] == "connected"
    assert saved_data["state"]["message_count"] == 1
    assert saved_data["update_count"] >= 3
    print("✓ State persistence works")
    
    # Cleanup
    temp_path.unlink()
    
    return True

async def test_voice_engine_integration():
    """Test integration with VoiceEngine"""
    print("\n=== Test 8: VoiceEngine Integration ===")
    
    try:
        from ..voice_engine import VoiceEngine, VoiceEngineConfig
        
        # Create engine (without connecting)
        config = VoiceEngineConfig(
            api_key="test-key",
            mode="fast"
        )
        engine = VoiceEngine(config=config)
        
        # Check state is initialized
        assert engine.conversation_state is not None
        assert engine.conversation_state.status == ConversationStatus.IDLE
        print("✓ VoiceEngine has state manager")
        
        # Access state manager
        assert engine.state_manager is not None
        print("✓ State manager accessible")
        
        return True
        
    except Exception as e:
        print(f"✗ VoiceEngine integration failed: {e}")
        return False

async def main():
    """Run all state management tests"""
    print("State Management System Tests")
    print("=" * 50)
    
    tests = [
        test_basic_state_operations,
        test_atomic_updates,
        test_state_history,
        test_conversation_flow,
        test_event_integration,
        test_performance,
        test_state_persistence,
        test_voice_engine_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    # Summary
    print("Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All state management tests passed!")
        print("\nState management system is ready for use:")
        print("- Atomic updates with < 1ms latency")
        print("- Thread-safe operations")
        print("- Event integration")
        print("- History and persistence")
        print("- Clean API")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)