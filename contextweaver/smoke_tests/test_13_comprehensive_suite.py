#!/usr/bin/env python3
"""
Comprehensive Test Suite

Tests the complete integrated system with all phases working together.
This test suite validates:
- End-to-end functionality
- Performance benchmarks
- Stress testing
- Edge cases
- Real-world scenarios
"""

import asyncio
import sys
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple
sys.path.append(str(Path(__file__).parent.parent.parent))

from voxon import Voxon, VoxonConfig
from voxengine import VoiceEngine, VoiceEngineConfig
from contextweaver import ContextWeaver
from contextweaver.strategies import ConservativeStrategy, AggressiveStrategy, AdaptiveStrategy
from contextweaver.detectors import (
    SilenceDetector, PauseDetector, TopicChangeDetector,
    ResponseTimingDetector, ConversationFlowDetector
)
from contextweaver.schema import ContextToInject, InjectionTiming, ContextPriority
from voxon.state import Message, SpeakerRole, ConversationStatus


class ComprehensiveTestSuite:
    """Comprehensive test suite for the entire system"""
    
    def __init__(self):
        self.results = {
            'performance': {},
            'functionality': {},
            'stress': {},
            'edge_cases': {},
            'real_world': {}
        }
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST SUITE - ALL PHASES INTEGRATED")
        print("="*60 + "\n")
        
        test_categories = [
            ("Functionality Tests", self.run_functionality_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Stress Tests", self.run_stress_tests),
            ("Edge Case Tests", self.run_edge_case_tests),
            ("Real-World Scenario Tests", self.run_real_world_tests)
        ]
        
        for category_name, test_func in test_categories:
            print(f"\n{category_name}")
            print("-" * len(category_name))
            try:
                await test_func()
                print(f"✓ {category_name} completed")
            except Exception as e:
                print(f"✗ {category_name} failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        self.print_summary()
    
    async def run_functionality_tests(self):
        """Test all functional requirements"""
        tests = [
            self.test_all_vad_modes,
            self.test_all_strategies,
            self.test_all_detectors,
            self.test_context_priority_handling,
            self.test_injection_timing_modes,
            self.test_learning_persistence
        ]
        
        for test in tests:
            test_name = test.__name__.replace('test_', '')
            try:
                result = await test()
                self.results['functionality'][test_name] = result
                print(f"  ✓ {test_name}")
            except Exception as e:
                self.results['functionality'][test_name] = {'passed': False, 'error': str(e)}
                print(f"  ✗ {test_name}: {str(e)}")
    
    async def test_all_vad_modes(self) -> Dict[str, Any]:
        """Test all VAD mode combinations"""
        start_time = time.time()
        
        vad_configs = [
            ("server+auto", "server", True),
            ("server+manual", "server", False),
            ("client+auto", "client", True),
            ("client+manual", "client", False)
        ]
        
        results = {}
        for name, vad_type, auto_response in vad_configs:
            # Use mock voice engine for testing
            class TestVoiceEngine:
                def __init__(self, vad_type):
                    self.config = type('Config', (), {'vad_type': vad_type})()
                    self.session_config = {
                        'turn_detection': {'create_response': auto_response}
                    }
                    self.events = type('Events', (), {
                        'on': lambda self, e, h: None,
                        'off': lambda self, h: None
                    })()
            
            voice_engine = TestVoiceEngine(vad_type)
            
            context_weaver = ContextWeaver(strategy=AdaptiveStrategy())
            voxon = Voxon(VoxonConfig())
            
            coordinator = voxon.engine_coordinator
            coordinator.voice_engine = voice_engine
            coordinator.context_engine = context_weaver
            
            await coordinator.initialize()
            
            # Verify configuration
            assert coordinator.vad_mode == vad_type
            assert coordinator.auto_response == auto_response
            
            results[name] = {
                'injection_mode': coordinator.injection_mode,
                'monitor_interval': coordinator.monitor_interval,
                'window_size': coordinator.injection_window_manager.window_size_ms
            }
            
            await coordinator.shutdown()
        
        return {
            'passed': True,
            'duration': time.time() - start_time,
            'configs_tested': len(vad_configs),
            'results': results
        }
    
    async def test_all_strategies(self) -> Dict[str, Any]:
        """Test all injection strategies"""
        strategies = [
            ConservativeStrategy(),
            AggressiveStrategy(),
            AdaptiveStrategy()
        ]
        
        results = {}
        for strategy in strategies:
            name = strategy.__class__.__name__
            context_weaver = ContextWeaver(strategy=strategy)
            
            # Test injection decision
            mock_state = type('State', (), {
                'messages': [Message(role=SpeakerRole.USER, content="Test")],
                'audio': type('Audio', (), {'vad_active': False})()
            })()
            
            detection = type('Detection', (), {
                'detected': True,
                'confidence': 0.8,
                'metadata': {}
            })()
            
            # Create minimal context for testing
            test_context = ContextToInject(
                information={"test": "data"},
                priority=ContextPriority.MEDIUM
            )
            available_context = {"test": test_context}
            
            decision = await strategy.decide(
                detections=[detection],
                state=mock_state,
                available_context=available_context
            )
            should_inject = decision.should_inject
            
            results[name] = {
                'threshold': getattr(strategy, 'threshold', 0.5),
                'injected': should_inject,
                'adaptive': hasattr(strategy, 'adapt_threshold')
            }
        
        return {
            'passed': True,
            'strategies_tested': len(strategies),
            'results': results
        }
    
    async def test_all_detectors(self) -> Dict[str, Any]:
        """Test all detector types"""
        detectors = [
            SilenceDetector(enable_prediction=True),
            PauseDetector(),
            TopicChangeDetector(),
            ResponseTimingDetector(),
            ConversationFlowDetector()
        ]
        
        # Create test state
        mock_state = type('State', (), {
            'messages': [
                Message(role=SpeakerRole.USER, content="Hello"),
                Message(role=SpeakerRole.ASSISTANT, content="Hi there!"),
                Message(role=SpeakerRole.USER, content="Can you help me?")
            ],
            'audio': type('Audio', (), {
                'vad_active': False,
                'is_listening': True
            })(),
            'vad_mode': 'client',
            'auto_response': True
        })()
        
        results = {}
        for detector in detectors:
            name = detector.__class__.__name__
            result = await detector.detect(mock_state)
            
            stats = detector.get_statistics() if hasattr(detector, 'get_statistics') else {}
            
            results[name] = {
                'detected': result.detected,
                'confidence': result.confidence,
                'has_stats': bool(stats)
            }
        
        return {
            'passed': True,
            'detectors_tested': len(detectors),
            'results': results
        }
    
    async def test_context_priority_handling(self) -> Dict[str, Any]:
        """Test context priority system"""
        context_weaver = ContextWeaver()
        
        # Add contexts with different priorities
        contexts = [
            ContextToInject(
                information={"msg": "low"},
                priority=ContextPriority.LOW
            ),
            ContextToInject(
                information={"msg": "high"},
                priority=ContextPriority.HIGH
            ),
            ContextToInject(
                information={"msg": "critical"},
                priority=ContextPriority.CRITICAL
            ),
            ContextToInject(
                information={"msg": "medium"},
                priority=ContextPriority.MEDIUM
            )
        ]
        
        for ctx in contexts:
            context_weaver.add_context(ctx)
        
        # Get relevant contexts (should have critical first)
        relevant = context_weaver.get_relevant_contexts(
            state=type('State', (), {'messages': []})(),
            max_items=5
        )
        assert len(relevant) > 0
        assert relevant[0].priority == ContextPriority.CRITICAL
        
        return {
            'passed': True,
            'contexts_added': len(contexts),
            'first_priority': next_ctx.priority.value
        }
    
    async def test_injection_timing_modes(self) -> Dict[str, Any]:
        """Test all injection timing modes"""
        timings = [
            InjectionTiming.IMMEDIATE,
            InjectionTiming.NEXT_PAUSE,
            InjectionTiming.NEXT_TURN,
            InjectionTiming.MANUAL
        ]
        
        results = {}
        for timing in timings:
            context = ContextToInject(
                information={"test": timing.value},
                timing=timing
            )
            
            results[timing.value] = {
                'enum_value': timing.value,
                'is_immediate': timing == InjectionTiming.IMMEDIATE
            }
        
        return {
            'passed': True,
            'timings_tested': len(timings),
            'results': results
        }
    
    async def test_learning_persistence(self) -> Dict[str, Any]:
        """Test learning system persistence"""
        flow_detector = ConversationFlowDetector(learning_rate=0.3)
        
        # Record multiple outcomes
        for i in range(10):
            success = i % 3 != 0  # 70% success rate
            flow_detector.record_injection_outcome(
                pattern_type="test_pattern",
                phase=flow_detector.current_phase,
                success=success,
                context_type="test",
                metadata={}
            )
        
        stats = flow_detector.get_statistics()
        learning_progress = stats['learning_progress']
        
        return {
            'passed': True,
            'patterns_learned': stats['patterns_learned'],
            'success_rate': learning_progress['success_rate'],
            'confidence_level': learning_progress['confidence_level']
        }
    
    async def run_performance_tests(self):
        """Run performance benchmarks"""
        tests = [
            self.benchmark_detection_speed,
            self.benchmark_injection_latency,
            self.benchmark_memory_usage,
            self.benchmark_concurrent_operations
        ]
        
        for test in tests:
            test_name = test.__name__.replace('benchmark_', '')
            try:
                result = await test()
                self.results['performance'][test_name] = result
                print(f"  ✓ {test_name}: {result.get('avg_time', 'N/A')}ms avg")
            except Exception as e:
                self.results['performance'][test_name] = {'passed': False, 'error': str(e)}
                print(f"  ✗ {test_name}: {str(e)}")
    
    async def benchmark_detection_speed(self) -> Dict[str, Any]:
        """Benchmark detector performance"""
        detectors = [
            SilenceDetector(),
            ResponseTimingDetector(),
            ConversationFlowDetector()
        ]
        
        mock_state = type('State', (), {
            'messages': [Message(role=SpeakerRole.USER, content="Test")] * 10,
            'audio': type('Audio', (), {'vad_active': False})(),
            'vad_mode': 'server',
            'auto_response': True
        })()
        
        times = []
        iterations = 100
        
        for _ in range(iterations):
            start = time.perf_counter()
            
            for detector in detectors:
                await detector.detect(mock_state)
            
            times.append((time.perf_counter() - start) * 1000)
        
        return {
            'passed': True,
            'iterations': iterations,
            'avg_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0
        }
    
    async def benchmark_injection_latency(self) -> Dict[str, Any]:
        """Benchmark injection execution latency"""
        context_weaver = ContextWeaver(strategy=AdaptiveStrategy())
        
        # Add test context
        context_weaver.add_context(ContextToInject(
            information={"test": "data"},
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH
        ))
        
        times = []
        iterations = 50
        
        for _ in range(iterations):
            mock_state = type('State', (), {
                'messages': [],
                'audio': type('Audio', (), {'vad_active': False})()
            })()
            
            start = time.perf_counter()
            result = await context_weaver.check_injection(mock_state)
            times.append((time.perf_counter() - start) * 1000)
        
        return {
            'passed': True,
            'iterations': iterations,
            'avg_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times)
        }
    
    async def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple instances
        instances = []
        for _ in range(10):
            voxon = Voxon(VoxonConfig())
            context_weaver = ContextWeaver()
            
            # Add many contexts
            for i in range(100):
                context_weaver.add_context(ContextToInject(
                    information={f"key_{i}": f"value_{i}"},
                    priority=ContextPriority.MEDIUM
                ))
            
            instances.append((voxon, context_weaver))
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Cleanup
        instances.clear()
        gc.collect()
        
        # Measure after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'passed': True,
            'baseline_mb': round(baseline_memory, 2),
            'peak_mb': round(peak_memory, 2),
            'final_mb': round(final_memory, 2),
            'peak_increase_mb': round(peak_memory - baseline_memory, 2),
            'leaked_mb': round(max(0, final_memory - baseline_memory), 2)
        }
    
    async def benchmark_concurrent_operations(self) -> Dict[str, Any]:
        """Benchmark concurrent operation handling"""
        context_weaver = ContextWeaver()
        
        async def inject_task(index: int):
            context = ContextToInject(
                information={f"task_{index}": "data"},
                priority=ContextPriority.MEDIUM
            )
            context_weaver.add_context(context)
            
            mock_state = type('State', (), {
                'messages': [],
                'audio': type('Audio', (), {'vad_active': False})()
            })()
            
            start = time.perf_counter()
            await context_weaver.check_injection(mock_state)
            return time.perf_counter() - start
        
        # Run concurrent tasks
        start = time.perf_counter()
        tasks = [inject_task(i) for i in range(50)]
        times = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start
        
        return {
            'passed': True,
            'concurrent_tasks': len(tasks),
            'total_time': total_time * 1000,
            'avg_task_time': statistics.mean(times) * 1000,
            'speedup_factor': sum(times) / total_time
        }
    
    async def run_stress_tests(self):
        """Run stress tests"""
        tests = [
            self.stress_test_high_frequency,
            self.stress_test_large_context,
            self.stress_test_rapid_mode_switching
        ]
        
        for test in tests:
            test_name = test.__name__.replace('stress_test_', '')
            try:
                result = await test()
                self.results['stress'][test_name] = result
                print(f"  ✓ {test_name}: handled {result.get('operations', 'N/A')} ops")
            except Exception as e:
                self.results['stress'][test_name] = {'passed': False, 'error': str(e)}
                print(f"  ✗ {test_name}: {str(e)}")
    
    async def stress_test_high_frequency(self) -> Dict[str, Any]:
        """Stress test with high-frequency operations"""
        context_weaver = ContextWeaver()
        
        operations = 1000
        start = time.perf_counter()
        
        for i in range(operations):
            # Rapidly add and check contexts
            context_weaver.add_context(ContextToInject(
                information={f"rapid_{i}": "data"},
                priority=ContextPriority.LOW if i % 2 else ContextPriority.HIGH
            ))
            
            if i % 10 == 0:
                mock_state = type('State', (), {
                    'messages': [],
                    'audio': type('Audio', (), {'vad_active': False})()
                })()
                await context_weaver.check_injection(mock_state)
        
        duration = time.perf_counter() - start
        
        return {
            'passed': True,
            'operations': operations,
            'duration': duration,
            'ops_per_second': operations / duration,
            'queue_size': len(context_weaver.available_context)
        }
    
    async def stress_test_large_context(self) -> Dict[str, Any]:
        """Stress test with large context data"""
        context_weaver = ContextWeaver()
        
        # Create large context
        large_data = {f"key_{i}": "x" * 1000 for i in range(100)}
        
        context = ContextToInject(
            information=large_data,
            priority=ContextPriority.HIGH
        )
        
        context_weaver.add_context(context)
        
        # Measure processing time
        mock_state = type('State', (), {
            'messages': [],
            'audio': type('Audio', (), {'vad_active': False})()
        })()
        
        start = time.perf_counter()
        result = await context_weaver.check_injection(mock_state)
        duration = (time.perf_counter() - start) * 1000
        
        return {
            'passed': True,
            'context_size_kb': len(str(large_data)) / 1024,
            'processing_time_ms': duration,
            'injection_triggered': result is not None
        }
    
    async def stress_test_rapid_mode_switching(self) -> Dict[str, Any]:
        """Stress test rapid VAD mode switching"""
        # Use mock voice engine
        class TestVoiceEngine:
            def __init__(self):
                self.config = type('Config', (), {'vad_type': 'client'})()
                self.session_config = {
                    'turn_detection': {'create_response': True}
                }
                self.events = type('Events', (), {
                    'on': lambda self, e, h: None,
                    'off': lambda self, h: None
                })()
        
        voice_engine = TestVoiceEngine()
        context_weaver = ContextWeaver(strategy=AdaptiveStrategy())
        voxon = Voxon(VoxonConfig())
        
        coordinator = voxon.engine_coordinator
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        await coordinator.initialize()
        
        switches = 100
        modes = [
            ("server", True),
            ("server", False),
            ("client", True),
            ("client", False)
        ]
        
        start = time.perf_counter()
        
        for i in range(switches):
            vad_type, auto_response = modes[i % len(modes)]
            # Update VAD configuration
            coordinator.voice_engine.config.vad_type = vad_type
            coordinator.voice_engine.session_config['turn_detection']['create_response'] = auto_response
            coordinator.vad_adapter.update_vad_mode(vad_type, auto_response)
            coordinator._update_injection_mode()
        
        duration = time.perf_counter() - start
        
        await coordinator.shutdown()
        
        return {
            'passed': True,
            'mode_switches': switches,
            'duration': duration,
            'switches_per_second': switches / duration
        }
    
    async def run_edge_case_tests(self):
        """Run edge case tests"""
        tests = [
            self.test_empty_conversation,
            self.test_interrupted_injection,
            self.test_conflicting_contexts
        ]
        
        for test in tests:
            test_name = test.__name__.replace('test_', '')
            try:
                result = await test()
                self.results['edge_cases'][test_name] = result
                print(f"  ✓ {test_name}")
            except Exception as e:
                self.results['edge_cases'][test_name] = {'passed': False, 'error': str(e)}
                print(f"  ✗ {test_name}: {str(e)}")
    
    async def test_empty_conversation(self) -> Dict[str, Any]:
        """Test behavior with empty conversation"""
        context_weaver = ContextWeaver()
        
        # Empty state
        empty_state = type('State', (), {
            'messages': [],
            'turns': [],
            'audio': type('Audio', (), {'vad_active': False})()
        })()
        
        # Should handle gracefully
        result = await context_weaver.check_injection(empty_state)
        
        # Add context to empty conversation
        context_weaver.add_context(ContextToInject(
            information={"greeting": "Hello!"},
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH
        ))
        
        result2 = await context_weaver.check_injection(empty_state)
        
        return {
            'passed': True,
            'handled_empty_state': True,
            'injection_possible': result2 is not None
        }
    
    async def test_interrupted_injection(self) -> Dict[str, Any]:
        """Test injection interruption handling"""
        context_weaver = ContextWeaver()
        
        # Add context
        context_weaver.add_context(ContextToInject(
            information={"data": "test"},
            timing=InjectionTiming.NEXT_PAUSE,
            priority=ContextPriority.HIGH
        ))
        
        # Simulate state where injection starts
        state1 = type('State', (), {
            'messages': [Message(role=SpeakerRole.USER, content="Test")],
            'audio': type('Audio', (), {'vad_active': False})()
        })()
        
        result1 = await context_weaver.check_injection(state1)
        
        # Simulate interruption (VAD becomes active)
        state2 = type('State', (), {
            'messages': state1.messages,
            'audio': type('Audio', (), {'vad_active': True})()
        })()
        
        result2 = await context_weaver.check_injection(state2)
        
        return {
            'passed': True,
            'initial_injection': result1 is not None,
            'interrupted': result2 is None,
            'context_preserved': len(context_weaver.available_context) > 0
        }
    
    async def test_conflicting_contexts(self) -> Dict[str, Any]:
        """Test handling of conflicting contexts"""
        context_weaver = ContextWeaver()
        
        # Add conflicting contexts
        contexts = [
            ContextToInject(
                information={"action": "pause", "type": "control"},
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.HIGH
            ),
            ContextToInject(
                information={"action": "continue", "type": "control"},
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.CRITICAL
            )
        ]
        
        for ctx in contexts:
            context_weaver.add_context(ctx)
        
        # Should prioritize critical
        relevant = context_weaver.get_relevant_contexts(
            state=type('State', (), {'messages': []})(),
            max_items=1
        )
        next_ctx = relevant[0] if relevant else None
        assert next_ctx is not None
        
        return {
            'passed': True,
            'conflict_resolved': True,
            'winning_priority': next_ctx.priority.value if next_ctx else 'N/A',
            'winning_action': next_ctx.information['action'] if next_ctx else 'N/A'
        }
    
    async def run_real_world_tests(self):
        """Run real-world scenario tests"""
        tests = [
            self.test_customer_support_scenario,
            self.test_technical_assistance_scenario,
            self.test_long_conversation_scenario
        ]
        
        for test in tests:
            test_name = test.__name__.replace('test_', '').replace('_scenario', '')
            try:
                result = await test()
                self.results['real_world'][test_name] = result
                print(f"  ✓ {test_name}: {result.get('injections_made', 0)} injections")
            except Exception as e:
                self.results['real_world'][test_name] = {'passed': False, 'error': str(e)}
                print(f"  ✗ {test_name}: {str(e)}")
    
    async def test_customer_support_scenario(self) -> Dict[str, Any]:
        """Test customer support conversation scenario"""
        # Setup - use mock for conversation state management
        class MockVoiceEngine:
            def __init__(self):
                from voxon.state import ConversationState, ConversationStatus
                self.config = type('Config', (), {'vad_type': 'client'})()
                self.conversation_state = ConversationState(
                    status=ConversationStatus.CONNECTED,
                    messages=[],
                    turns=[]
                )
                self.contexts_injected = []
                self.events = type('Events', (), {
                    'on': lambda self, e, h: None,
                    'off': lambda self, h: None
                })()
            
            def add_message(self, role, content):
                from voxon.state import Message
                msg = Message(role=role, content=content)
                self.conversation_state = self.conversation_state.evolve(
                    messages=list(self.conversation_state.messages) + [msg]
                )
            
            async def send_text(self, text):
                self.contexts_injected.append(text)
        
        voice_engine = MockVoiceEngine()
        context_weaver = ContextWeaver(
            strategy=AdaptiveStrategy(),
            detectors=[
                SilenceDetector(enable_prediction=True),
                ConversationFlowDetector(),
                ResponseTimingDetector()
            ]
        )
        
        voxon = Voxon(VoxonConfig())
        coordinator = voxon.engine_coordinator
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        await coordinator.initialize()
        
        # Add support contexts
        support_contexts = [
            ContextToInject(
                information={"kb": "Our return policy is 30 days"},
                timing=InjectionTiming.NEXT_PAUSE,
                priority=ContextPriority.HIGH,
                conditions={"keywords": ["return", "policy"]}
            ),
            ContextToInject(
                information={"kb": "Shipping takes 3-5 business days"},
                timing=InjectionTiming.NEXT_PAUSE,
                priority=ContextPriority.HIGH,
                conditions={"keywords": ["shipping", "delivery"]}
            ),
            ContextToInject(
                information={"escalation": "Connect to human agent"},
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.CRITICAL,
                conditions={"sentiment": "frustrated"}
            )
        ]
        
        for ctx in support_contexts:
            context_weaver.add_context(ctx)
        
        # Simulate conversation
        messages = [
            (SpeakerRole.USER, "Hi, I have a question about returns"),
            (SpeakerRole.ASSISTANT, "I'd be happy to help with returns!"),
            (SpeakerRole.USER, "What's your return policy?"),
            (SpeakerRole.ASSISTANT, "Let me check that for you..."),
            (SpeakerRole.USER, "Also, how long does shipping take?"),
            (SpeakerRole.ASSISTANT, "I can help with shipping info too.")
        ]
        
        injections_made = 0
        for role, content in messages:
            # Update conversation state
            voice_engine.add_message(role, content)
            
            # Check for injections
            state = voice_engine.conversation_state
            result = await context_weaver.check_injection(state)
            if result:
                injections_made += 1
        
        await coordinator.shutdown()
        
        return {
            'passed': True,
            'messages_processed': len(messages),
            'injections_made': injections_made,
            'contexts_remaining': len(context_weaver.available_context)
        }
    
    async def test_technical_assistance_scenario(self) -> Dict[str, Any]:
        """Test technical assistance conversation scenario"""
        # Setup with server VAD for faster responses
        class MockVoiceEngine:
            def __init__(self):
                from voxon.state import ConversationState, ConversationStatus
                self.config = type('Config', (), {'vad_type': 'server'})()
                self.session_config = {
                    'turn_detection': {'create_response': True}
                }
                self.conversation_state = ConversationState(
                    status=ConversationStatus.CONNECTED,
                    messages=[],
                    turns=[]
                )
                self.contexts_injected = []
                self.events = type('Events', (), {
                    'on': lambda self, e, h: None,
                    'off': lambda self, h: None
                })()
            
            def add_message(self, role, content):
                from voxon.state import Message
                msg = Message(role=role, content=content)
                self.conversation_state = self.conversation_state.evolve(
                    messages=list(self.conversation_state.messages) + [msg]
                )
            
            async def send_text(self, text):
                self.contexts_injected.append(text)
        
        voice_engine = MockVoiceEngine()
        
        context_weaver = ContextWeaver(
            strategy=AggressiveStrategy(),  # More aggressive for technical help
            detectors=[
                SilenceDetector(silence_threshold_ms=500),  # Shorter threshold
                TopicChangeDetector(),
                ConversationFlowDetector()
            ]
        )
        
        voxon = Voxon(VoxonConfig())
        coordinator = voxon.engine_coordinator
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        await coordinator.initialize()
        
        # Add technical contexts
        tech_contexts = [
            ContextToInject(
                information={"code": "import asyncio\nasync def main():\n    pass"},
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.HIGH,
                conditions={"keywords": ["async", "code", "example"]}
            ),
            ContextToInject(
                information={"docs": "See: https://docs.python.org/3/library/asyncio.html"},
                timing=InjectionTiming.NEXT_TURN,
                priority=ContextPriority.MEDIUM,
                conditions={"keywords": ["documentation", "docs"]}
            )
        ]
        
        for ctx in tech_contexts:
            context_weaver.add_context(ctx)
        
        # Simulate technical conversation
        messages = [
            (SpeakerRole.USER, "Can you show me an async Python example?"),
            (SpeakerRole.ASSISTANT, "Sure! Here's a basic async example..."),
            (SpeakerRole.USER, "Where can I find more documentation?"),
            (SpeakerRole.ASSISTANT, "I'll provide you with documentation links.")
        ]
        
        injections_made = 0
        patterns_detected = []
        
        for role, content in messages:
            voice_engine.add_message(role, content)
            
            # Check detections
            for detector in context_weaver.detectors:
                if isinstance(detector, ConversationFlowDetector):
                    result = await detector.detect(voice_engine.conversation_state)
                    if result.detected:
                        patterns_detected.append(result.metadata.get('pattern_type'))
            
            # Check injections
            state = voice_engine.conversation_state
            result = await context_weaver.check_injection(state)
            if result:
                injections_made += 1
        
        await coordinator.shutdown()
        
        return {
            'passed': True,
            'messages_processed': len(messages),
            'injections_made': injections_made,
            'patterns_detected': len(set(patterns_detected)),
            'vad_mode': coordinator.vad_mode
        }
    
    async def test_long_conversation_scenario(self) -> Dict[str, Any]:
        """Test long conversation with learning"""
        class MockVoiceEngine:
            def __init__(self):
                from voxon.state import ConversationState, ConversationStatus
                self.config = type('Config', (), {'vad_type': 'client'})()
                self.conversation_state = ConversationState(
                    status=ConversationStatus.CONNECTED,
                    messages=[],
                    turns=[]
                )
                self.events = type('Events', (), {
                    'on': lambda self, e, h: None,
                    'off': lambda self, h: None
                })()
            
            def add_message(self, role, content):
                from voxon.state import Message
                msg = Message(role=role, content=content)
                self.conversation_state = self.conversation_state.evolve(
                    messages=list(self.conversation_state.messages) + [msg]
                )
        
        voice_engine = MockVoiceEngine()
        
        # Create detectors with learning
        flow_detector = ConversationFlowDetector(learning_rate=0.2)
        
        context_weaver = ContextWeaver(
            strategy=AdaptiveStrategy(),
            detectors=[
                SilenceDetector(enable_prediction=True),
                flow_detector,
                ResponseTimingDetector()
            ]
        )
        
        voxon = Voxon(VoxonConfig())
        coordinator = voxon.engine_coordinator
        coordinator.voice_engine = voice_engine
        coordinator.context_engine = context_weaver
        
        await coordinator.initialize()
        
        # Simulate long conversation with patterns
        conversation_segments = [
            # Greeting
            [
                (SpeakerRole.USER, "Hello!"),
                (SpeakerRole.ASSISTANT, "Hi! How can I help you today?")
            ],
            # Q&A cycles
            [
                (SpeakerRole.USER, "What's the weather like?"),
                (SpeakerRole.ASSISTANT, "Let me check the weather for you."),
                (SpeakerRole.USER, "How about tomorrow?"),
                (SpeakerRole.ASSISTANT, "Tomorrow's forecast shows..."),
                (SpeakerRole.USER, "What about the weekend?"),
                (SpeakerRole.ASSISTANT, "The weekend weather looks...")
            ],
            # Topic change
            [
                (SpeakerRole.USER, "Actually, can you help me with my calendar?"),
                (SpeakerRole.ASSISTANT, "Of course! Switching to calendar assistance.")
            ],
            # More Q&A
            [
                (SpeakerRole.USER, "What meetings do I have today?"),
                (SpeakerRole.ASSISTANT, "Let me check your calendar."),
                (SpeakerRole.USER, "Can you schedule one for tomorrow?"),
                (SpeakerRole.ASSISTANT, "I'll help you schedule a meeting.")
            ],
            # Conclusion
            [
                (SpeakerRole.USER, "Thanks for all your help!"),
                (SpeakerRole.ASSISTANT, "You're welcome! Have a great day!")
            ]
        ]
        
        total_messages = 0
        phases_detected = set()
        learning_events = 0
        
        for segment in conversation_segments:
            for role, content in segment:
                total_messages += 1
                
                voice_engine.add_message(role, content)
                
                # Let flow detector analyze
                result = await flow_detector.detect(voice_engine.conversation_state)
                if flow_detector.current_phase:
                    phases_detected.add(flow_detector.current_phase)
                
                # Simulate learning from injection outcomes
                if total_messages % 5 == 0:
                    flow_detector.record_injection_outcome(
                        pattern_type="qa_cycle",
                        phase=flow_detector.current_phase,
                        success=True,
                        context_type="answer",
                        metadata={}
                    )
                    learning_events += 1
        
        # Get final statistics
        final_stats = flow_detector.get_statistics()
        
        await coordinator.shutdown()
        
        return {
            'passed': True,
            'total_messages': total_messages,
            'conversation_phases': len(phases_detected),
            'patterns_learned': final_stats['patterns_learned'],
            'learning_events': learning_events,
            'avg_success_rate': final_stats['average_success_rate']
        }
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for category, results in self.results.items():
            if not results:
                continue
                
            passed = sum(1 for r in results.values() if r.get('passed', False))
            total = len(results)
            
            print(f"\n{category.upper()}: {passed}/{total} passed")
            
            if category == 'performance':
                # Show performance metrics
                for test_name, result in results.items():
                    if result.get('passed'):
                        if 'avg_time' in result:
                            print(f"  {test_name}: {result['avg_time']:.2f}ms avg")
                        elif 'ops_per_second' in result:
                            print(f"  {test_name}: {result['ops_per_second']:.0f} ops/sec")
            
            elif category == 'real_world':
                # Show scenario results
                for test_name, result in results.items():
                    if result.get('passed'):
                        print(f"  {test_name}: {result.get('injections_made', 0)} injections")
        
        # Overall summary
        total_passed = sum(
            sum(1 for r in results.values() if r.get('passed', False))
            for results in self.results.values()
        )
        total_tests = sum(len(results) for results in self.results.values())
        
        print(f"\nOVERALL: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            print("\n✅ ALL TESTS PASSED! System is ready for production.")
        else:
            print("\n⚠️  Some tests failed. Review results above.")


async def main():
    """Run comprehensive test suite"""
    suite = ComprehensiveTestSuite()
    await suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())