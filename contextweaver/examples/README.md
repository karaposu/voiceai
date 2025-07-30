# ContextWeaver Examples

This directory contains practical examples demonstrating how to integrate ContextWeaver into various real-world applications.

## Examples Overview

### 1. Customer Support Bot (`customer_support_bot.py`)

A sophisticated customer support assistant that intelligently injects relevant information during conversations.

**Key Features:**
- Knowledge base integration for policies and FAQs
- Automatic escalation to human agents
- Frustration detection and empathy injection
- Order tracking integration
- Proactive assistance after silence

**Use Cases:**
- E-commerce support
- Technical helpdesk
- Service inquiries
- Complaint handling

**Run Example:**
```bash
python customer_support_bot.py
```

### 2. Code Assistant (`code_assistant.py`)

An AI programming assistant that provides contextual code examples and documentation.

**Key Features:**
- Language detection (Python, JavaScript, etc.)
- Pattern-based code snippet injection
- Documentation link suggestions
- Error debugging assistance
- Adaptive complexity based on user level

**Use Cases:**
- Pair programming
- Learning new languages
- Debugging sessions
- API exploration

**Run Example:**
```bash
python code_assistant.py
```

### 3. Educational Tutor (`educational_tutor.py`)

An adaptive learning assistant that provides personalized education with intelligent hint injection.

**Key Features:**
- Difficulty adjustment based on performance
- Progressive hint system
- Learning pattern analysis
- Encouragement and motivation
- Progress tracking and reporting

**Use Cases:**
- Math tutoring
- Language learning
- Test preparation
- Homework help

**Run Example:**
```bash
python educational_tutor.py
```

## Common Patterns

### 1. Context Preparation

All examples follow a pattern of preparing contexts based on domain knowledge:

```python
# Domain-specific context
context = ContextToInject(
    information={"key": "value"},
    timing=InjectionTiming.NEXT_PAUSE,
    priority=ContextPriority.HIGH,
    conditions={"keywords": ["trigger", "words"]},
    source="system_name"
)
context_weaver.add_context(context)
```

### 2. Event-Driven Injection

Examples use event handlers to trigger context checks:

```python
@voice_engine.events.on('user.message')
async def on_user_message(event):
    # Analyze message
    # Add relevant contexts
    # Check for injection opportunities
```

### 3. Adaptive Strategies

Each example uses strategies appropriate to its use case:

- **Customer Support**: AdaptiveStrategy with balanced threshold
- **Code Assistant**: AggressiveStrategy for quick help
- **Educational Tutor**: AdaptiveStrategy with high learning rate

### 4. Learning Integration

Examples demonstrate how to use the learning system:

```python
# Record outcome
detector.record_injection_outcome(
    pattern_type="pattern_name",
    phase=detector.current_phase,
    success=True,
    context_type="type",
    metadata={}
)
```

## Customization Guide

### Adapting for Your Use Case

1. **Choose Appropriate Detectors**
   ```python
   detectors = [
       SilenceDetector(silence_threshold_ms=your_threshold),
       # Add detectors that fit your timing needs
   ]
   ```

2. **Configure Strategy**
   ```python
   strategy = AdaptiveStrategy(
       initial_threshold=0.6,  # Adjust based on desired frequency
       learning_rate=0.2       # How quickly to adapt
   )
   ```

3. **Define Domain Contexts**
   ```python
   # Create contexts specific to your domain
   contexts = load_domain_knowledge()
   for ctx in contexts:
       context_weaver.add_context(ctx)
   ```

4. **Set Up Monitoring**
   ```python
   # Track performance
   stats = coordinator.get_stats()
   detector_stats = detector.get_statistics()
   ```

## Integration Tips

### 1. VAD Mode Selection

- **Server VAD + Auto Response**: Best for quick interactions (code help)
- **Client VAD + Manual Response**: Better for thoughtful conversations (tutoring)

### 2. Priority Management

- **CRITICAL**: Safety/legal requirements
- **HIGH**: Core functionality
- **MEDIUM**: Enhancements
- **LOW**: Nice-to-have features

### 3. Timing Strategies

- **IMMEDIATE**: Urgent information
- **NEXT_PAUSE**: Natural conversation flow
- **NEXT_TURN**: Between speakers
- **ON_TOPIC**: When relevant keywords appear

### 4. Performance Considerations

- Start with fewer detectors and add as needed
- Monitor injection frequency to avoid overload
- Use TTL for time-sensitive contexts
- Implement context cleanup for long sessions

## Testing Your Integration

1. **Unit Test Detectors**
   ```python
   result = await detector.detect(mock_state)
   assert result.confidence > threshold
   ```

2. **Integration Test Flow**
   ```python
   # Simulate conversation
   for message in test_conversation:
       state = update_state(message)
       context = await context_weaver.check_injection(state)
       assert_injection_appropriate(context)
   ```

3. **Performance Test**
   ```python
   # Measure latency
   start = time.perf_counter()
   await context_weaver.check_injection(state)
   latency = time.perf_counter() - start
   assert latency < 0.01  # 10ms
   ```

## Next Steps

1. Start with the example closest to your use case
2. Modify contexts and conditions for your domain
3. Test with real conversations
4. Monitor and tune based on results
5. Share your implementation patterns!

For more details, see the [API Documentation](../API_DOCUMENTATION.md).