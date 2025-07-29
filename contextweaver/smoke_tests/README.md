# ContextWeaver Smoke Tests

Real-world tests for the ContextWeaver intelligent context injection system.

## Overview

These smoke tests verify ContextWeaver functionality without using mocks, testing real behavior and performance.

## Test Modules

### 1. **test_01_basic_lifecycle.py**
- Engine initialization and configuration
- Start/stop operations
- Multiple engine instances
- Resource cleanup

### 2. **test_02_context_management.py**
- Adding and removing contexts
- Context expiration (TTL)
- Priority ordering
- Injection limits
- Conditional contexts

### 3. **test_03_detectors.py**
- SilenceDetector: Detects quiet periods
- PauseDetector: Identifies conversation pauses
- TopicChangeDetector: Recognizes topic shifts
- Multiple detectors working together
- Detection performance

### 4. **test_04_strategies.py**
- ConservativeStrategy: Careful injection
- AggressiveStrategy: Frequent injection
- AdaptiveStrategy: Learning behavior
- Strategy comparison
- Decision performance

### 5. **test_05_injection_flow.py**
- Complete injection decision flow
- Timing-based injection
- Condition evaluation
- Priority competition
- Real conversation simulation

### 6. **test_06_performance.py**
- 100+ contexts handling
- 1000+ operations per second
- Concurrent access safety
- Memory efficiency
- Sub-50ms detection cycles

## Running Tests

### Run all tests:
```bash
python -m contextweaver.smoke_tests.run_all_tests
```

### Run individual test:
```bash
python -m contextweaver.smoke_tests.test_01_basic_lifecycle
```

## Performance Benchmarks

Based on smoke tests:
- **Context Operations**: 1000+ per second
- **Detection Cycle**: < 50ms average
- **Memory**: Automatic cleanup of expired contexts
- **Scale**: Handles 100+ contexts efficiently
- **Concurrency**: Thread-safe operations

## Test Design Principles

1. **No Mocks**: Tests use real objects and timing
2. **Real Scenarios**: Simulates actual conversation flows
3. **Performance Focus**: Measures actual latencies
4. **Edge Cases**: Tests limits and error conditions
5. **Integration**: Tests components working together