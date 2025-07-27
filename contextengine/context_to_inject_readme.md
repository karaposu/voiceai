# Complete ContextToInject Explanation

## 🎯 What is ContextToInject?

ContextToInject is a self-contained, intelligent context package that knows not just **what** context to inject, but **when**, **why**, and **under what conditions**. It's designed to support asynchronous, multi-speed context calculations in real-time AI conversations.

## 📊 Core Architecture

### The Three Context Dimensions

**1. Information** - What the AI should know
- Facts, data, state
- User history, preferences
- Current situation details
- External knowledge
- Example: `{"user_name": "Sarah", "account_type": "premium", "previous_issues": 3}`

**2. Strategy** - How the AI should behave
- Response style and tone
- Conversation tactics
- Behavioral rules
- Approach modifications
- Example: `{"tone": "empathetic", "brevity": "concise", "formality": "casual"}`

**3. Attention** - What the AI should focus on
- Priority topics
- Goals for the conversation
- Things to emphasize or avoid
- Key outcomes to achieve
- Example: `{"focus": "problem_solving", "monitor_for": ["frustration"], "goal": "resolve_issue"}`

### The Control Dimensions

**4. Timing** - When to inject
- `IMMEDIATE`: Inject ASAP, potentially interrupting
- `NEXT_TURN`: At the next speaker change
- `NEXT_PAUSE`: During next silence
- `ON_TOPIC`: When relevant topic appears
- `ON_TRIGGER`: When conditions are met
- `LAZY`: Whenever convenient (perfect for slow calculations)
- `SCHEDULED`: At specific time
- `MANUAL`: Only when explicitly requested

**5. Conditions** - Requirements for injection
- Topic matching: `{"topics_include": ["startup", "business"]}`
- Mood detection: `{"user_mood": ["stressed", "anxious"]}`
- Conversation stage: `{"conversation_length_min": 5}`
- Time constraints: `{"time_since_last_injection_min": 300}`
- Custom functions: `{"custom": lambda state: state['urgency'] > 0.7}`

## 🚀 How It Enables Async Multi-Speed Processing

### The Parallel Pipeline Pattern

When a user speaks, multiple context calculations can start simultaneously:

**Fast Lane** (5-50ms)
- Creates context with basic analysis
- Sets `timing=IMMEDIATE` or `NEXT_TURN`
- Includes keywords, emotion markers
- Gets injected quickly at turn boundaries

**Medium Lane** (50-200ms)
- Creates context with deeper analysis
- Sets `timing=NEXT_PAUSE`
- Includes topic classification, intent
- Waits for natural pause to inject

**Slow Lane** (200ms-2s)
- Creates context with complex analysis
- Sets `timing=LAZY`
- Includes historical patterns, relationships
- Injects whenever there's an opportunity

### No Coordination Required

Each calculation independently creates its ContextToInject with appropriate timing. The engine automatically handles them based on their self-declared priorities and timing requirements.

## 🔄 Solving the N-1 Context Lag

### How Context Lag Works

In event-based systems, context calculations lag behind conversation:
- Message 1 → AI responds (no context yet)
- Meanwhile → Context 1 calculated
- Message 2 → AI responds (with Context 1)
- Meanwhile → Context 2 calculated
- Message 3 → AI responds (with Context 2)

### Why ContextToInject Makes This Work

1. **Progressive Enhancement**: Fast contexts inject early, deep contexts enhance later
2. **Natural Timing**: Contexts inject at conversation boundaries where lag isn't noticed
3. **Conditional Relevance**: Old contexts can check if they're still relevant before injecting
4. **Accumulation**: Each context can include summary of previous, maintaining continuity

## 📈 Lifecycle of a Context

### 1. Creation
Context is created by some analysis system:
```
context = ContextToInject(
    information={"sentiment": "frustrated"},
    timing=InjectionTiming.NEXT_PAUSE,
    priority=ContextPriority.HIGH
)
```

### 2. Validation
Before injection, context checks its conditions:
- Is it expired? (TTL check)
- Are conditions met? (topic, mood, etc.)
- Has it been injected too many times?

### 3. Queuing
Valid contexts enter a priority queue:
- Sorted by priority level
- Grouped by timing requirements
- Ready for injection

### 4. Injection
When appropriate moment arrives:
- Context is formatted for specific provider
- Injection is executed
- Context marks itself as injected

### 5. Expiration
Contexts can expire through:
- TTL timeout
- Maximum injection count
- Conditions no longer met
- Conversation ended

## 🎨 Key Architectural Benefits

### Self-Describing Contexts
Each context carries its own instructions for delivery. No external orchestration needed.

### Graceful Degradation
Slow contexts with LAZY timing won't block conversation if they're not ready.

### Natural Conversation Flow
Timing options align with human conversation patterns (turns, pauses, topics).

### Provider Agnostic
Core structure works regardless of AI provider - formatting happens at injection time.

### Failure Tolerance
If a context fails to calculate or inject, conversation continues unaffected.

### Resource Optimization
Expensive calculations can run with LAZY timing without impacting latency.

## 💡 Advanced Patterns

### Context Chains
Contexts can trigger other contexts:
- Initial context injects user info
- Triggers calculation of preference context
- Which triggers recommendation context

### Context Versioning
Same logical context can have multiple versions:
- Fast version (basic info)
- Full version (complete analysis)
- Summary version (for token limits)

### Context Merging
Multiple contexts can merge intelligently:
- Combine information fields
- Reconcile strategy conflicts
- Unify attention focus

### Adaptive Timing
Contexts can adjust their timing based on conversation:
- Start as LAZY
- Upgrade to IMMEDIATE if urgency detected
- Downgrade to MANUAL if conversation shifts

## 🏁 Why This Design Works

1. **Separation of Concerns**: Context creation, timing, and injection are independent
2. **Scalability**: Can add new context sources without changing injection logic
3. **Flexibility**: Supports everything from instant safety alerts to deep analysis
4. **Natural Feel**: Aligns with how humans progressively understand conversations
5. **Performance**: Fast path remains fast, slow analysis doesn't block

The ContextToInject dataclass is more than just a data container - it's an intelligent packet that enables sophisticated, multi-speed context processing while maintaining the real-time nature of voice conversations. It solves the fundamental challenge of adding intelligence without adding latency.