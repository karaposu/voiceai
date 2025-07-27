# ContextInjectionEngine Folder Structure (Corrected)

```
contextinjectionengine/
├── __init__.py
├── __version__.py
├── exceptions.py
├── schemas.py                     # All data models (ContextToInject, enums, etc.)
│
├── core/
│   ├── __init__.py
│   ├── engine.py                  # Main ContextInjectionEngine
│   ├── queue.py                   # Priority queue management
│   ├── scheduler.py               # Timing-based scheduling logic
│   └── validator.py               # Context validation before injection
│
├── strategies/
│   ├── __init__.py
│   ├── base.py                    # Abstract injection strategy
│   ├── immediate.py               # Immediate injection strategy
│   ├── pause_based.py             # Pause detection injection
│   ├── turn_based.py              # Turn-taking injection
│   ├── topic_based.py             # Topic-triggered injection
│   ├── lazy.py                    # Opportunistic injection
│   └── scheduled.py               # Time-based injection
│
├── formatters/
│   ├── __init__.py
│   ├── base.py                    # Abstract formatter
│   ├── openai.py                  # OpenAI format adapter
│   ├── anthropic.py               # Anthropic format adapter
│   ├── google.py                  # Google format adapter
│   └── generic.py                 # Generic text formatter
│
├── observers/
│   ├── __init__.py
│   ├── stream_observer.py         # Watch for stream events
│   ├── timing_observer.py         # Detect injection opportunities
│   └── state_observer.py          # Monitor conversation state changes
│
├── filters/
│   ├── __init__.py
│   ├── condition_filter.py        # Check if conditions are met
│   ├── expiry_filter.py           # Remove expired contexts
│   ├── duplicate_filter.py        # Prevent duplicate injections
│   └── relevance_filter.py        # Check if context still relevant
│
├── integration/
│   ├── __init__.py
│   ├── events.py                  # Event system integration
│   ├── hooks.py                   # Hook points for external systems
│   └── protocols.py               # Integration protocols/interfaces
│
├── storage/
│   ├── __init__.py
│   ├── queue_store.py             # Persistent queue storage
│   ├── history.py                 # Injection history tracking
│   └── cache.py                   # Recently injected contexts cache
│
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py                 # Injection performance metrics
│   ├── analytics.py               # Injection patterns analytics
│   └── health.py                  # Engine health monitoring
│
├── utils/
│   ├── __init__.py
│   ├── retry.py                   # Retry logic for failed injections
│   ├── merge.py                   # Context merging utilities
│   └── serialization.py           # Context serialization
│
└── examples/
    ├── __init__.py
    ├── basic_usage.py
    ├── event_integration.py
    ├── multi_provider.py
    └── priority_management.py
```

### Core Responsibilities

**ContextInjectionEngine receives `ContextToInject` objects and:**

1. **Validates** them (are they well-formed?)
2. **Filters** them (are conditions met? not expired?)
3. **Queues** them (by priority and timing)
4. **Schedules** them (based on timing strategy)
5. **Formats** them (for target provider)
6. **Injects** them (at the right moment)
7. **Tracks** them (for analytics and retry)

The engine is purely about **logistics and delivery**, not content creation