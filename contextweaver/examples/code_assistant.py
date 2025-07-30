#!/usr/bin/env python3
"""
Code Assistant Example

Demonstrates how to use ContextWeaver for intelligent code and documentation
injection in a technical assistance scenario.
"""

import asyncio
import re
from typing import Dict, List, Optional

from contextweaver import ContextWeaver
from contextweaver.strategies import AggressiveStrategy
from contextweaver.detectors import (
    SilenceDetector,
    TopicChangeDetector,
    ConversationFlowDetector
)
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority
from voxon import Voxon, VoxonConfig
from voxengine import VoiceEngine, VoiceEngineConfig


class CodeAssistant:
    """AI-powered coding assistant with intelligent context injection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.current_language = None
        self.current_topic = None
        self.code_context_stack = []
        
        # Code patterns and snippets
        self.code_patterns = {
            "python": {
                "async": {
                    "pattern": r"async|await|asyncio",
                    "snippets": {
                        "basic": """async def main():
    await asyncio.sleep(1)
    print("Hello async!")

asyncio.run(main())""",
                        "concurrent": """async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Run multiple requests concurrently
urls = ['http://api1.com', 'http://api2.com']
results = await asyncio.gather(*[fetch_data(url) for url in urls])"""
                    }
                },
                "error_handling": {
                    "pattern": r"try|except|error|exception",
                    "snippets": {
                        "basic": """try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    # Handle specific error
except Exception as e:
    logger.exception("Unexpected error")
    raise
finally:
    cleanup()""",
                        "custom": """class CustomError(Exception):
    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code"""
                    }
                }
            },
            "javascript": {
                "promises": {
                    "pattern": r"promise|then|catch|async|await",
                    "snippets": {
                        "basic": """// Using async/await
async function fetchData() {
    try {
        const response = await fetch('/api/data');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Failed to fetch:', error);
    }
}""",
                        "chaining": """fetch('/api/user')
    .then(response => response.json())
    .then(user => fetch(`/api/posts/${user.id}`))
    .then(response => response.json())
    .then(posts => console.log(posts))
    .catch(error => console.error(error));"""
                    }
                }
            }
        }
        
        # Documentation references
        self.docs_references = {
            "python": {
                "asyncio": "https://docs.python.org/3/library/asyncio.html",
                "typing": "https://docs.python.org/3/library/typing.html",
                "dataclasses": "https://docs.python.org/3/library/dataclasses.html"
            },
            "javascript": {
                "mdn": "https://developer.mozilla.org/docs/Web/JavaScript",
                "promises": "https://developer.mozilla.org/docs/Web/JavaScript/Guide/Using_promises"
            }
        }
    
    async def initialize(self):
        """Initialize all components"""
        # Set up voice engine optimized for technical discussions
        self.voice_engine = VoiceEngine(VoiceEngineConfig(
            api_key=self.api_key,
            vad_type="server",  # Fast response for code discussions
            voice="echo",       # Clear voice for code reading
            language="en"
        ))
        
        # Set up context weaver with aggressive strategy for code help
        self.context_weaver = ContextWeaver(
            strategy=AggressiveStrategy(
                threshold=0.5  # More eager to provide code examples
            ),
            detectors=[
                SilenceDetector(
                    silence_threshold_ms=800,  # Shorter for quick help
                    enable_prediction=True
                ),
                TopicChangeDetector(
                    similarity_threshold=0.4  # Detect language/topic switches
                ),
                ConversationFlowDetector(
                    learning_rate=0.25  # Learn coding patterns quickly
                )
            ]
        )
        
        # Set up Voxon orchestrator
        self.voxon = Voxon(VoxonConfig())
        coordinator = self.voxon.engine_coordinator
        coordinator.voice_engine = self.voice_engine
        coordinator.context_engine = self.context_weaver
        
        await coordinator.initialize()
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up event handlers"""
        @self.voice_engine.events.on('user.message')
        async def on_user_message(event):
            """Process user code questions"""
            message = event.message.content
            
            # Detect programming language
            detected_lang = self._detect_language(message)
            if detected_lang and detected_lang != self.current_language:
                self.current_language = detected_lang
                await self._inject_language_context(detected_lang)
            
            # Check for code patterns
            if self.current_language:
                patterns = self._find_code_patterns(message, self.current_language)
                for pattern_type, pattern_data in patterns:
                    await self._inject_code_example(
                        self.current_language,
                        pattern_type,
                        pattern_data
                    )
            
            # Check for help keywords
            if self._needs_documentation(message):
                await self._inject_documentation_links()
    
    def _detect_language(self, message: str) -> Optional[str]:
        """Detect programming language from message"""
        language_indicators = {
            "python": ["python", "py", "pip", "django", "flask", "pandas"],
            "javascript": ["javascript", "js", "node", "npm", "react", "vue"],
            "typescript": ["typescript", "ts", "tsx", "angular"],
            "java": ["java", "spring", "maven", "gradle"],
            "go": ["golang", "go ", "goroutine", "channel"],
            "rust": ["rust", "cargo", "crate", "borrowing"]
        }
        
        message_lower = message.lower()
        for lang, indicators in language_indicators.items():
            if any(ind in message_lower for ind in indicators):
                return lang
        return None
    
    def _find_code_patterns(self, message: str, language: str) -> List[tuple]:
        """Find code patterns in user message"""
        patterns_found = []
        
        if language in self.code_patterns:
            for pattern_type, pattern_info in self.code_patterns[language].items():
                if re.search(pattern_info["pattern"], message, re.IGNORECASE):
                    patterns_found.append((pattern_type, pattern_info))
        
        return patterns_found
    
    async def _inject_language_context(self, language: str):
        """Inject language-specific context"""
        lang_context = ContextToInject(
            information={
                "language": language,
                "message": f"I see you're working with {language.title()}. I'll provide {language}-specific help."
            },
            strategy={
                "code_formatting": True,
                "syntax_highlighting": language
            },
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.MEDIUM,
            source="language_detection"
        )
        self.context_weaver.add_context(lang_context)
    
    async def _inject_code_example(self, language: str, pattern_type: str, pattern_data: dict):
        """Inject relevant code examples"""
        snippets = pattern_data.get("snippets", {})
        
        # Choose appropriate snippet based on conversation context
        snippet_key = "basic"  # Default
        if self._is_advanced_user():
            snippet_key = list(snippets.keys())[-1]  # Use more advanced example
        
        if snippet_key in snippets:
            code_context = ContextToInject(
                information={
                    "code": snippets[snippet_key],
                    "language": language,
                    "topic": pattern_type,
                    "explanation": f"Here's an example of {pattern_type.replace('_', ' ')}:"
                },
                timing=InjectionTiming.NEXT_PAUSE,
                priority=ContextPriority.HIGH,
                conditions={
                    "confidence": 0.7,
                    "pattern": pattern_type
                },
                source="code_examples",
                ttl_seconds=300  # 5 minutes
            )
            self.context_weaver.add_context(code_context)
            
            # Track what we've shown
            self.code_context_stack.append({
                "language": language,
                "pattern": pattern_type,
                "timestamp": asyncio.get_event_loop().time()
            })
    
    def _needs_documentation(self, message: str) -> bool:
        """Check if user needs documentation links"""
        doc_keywords = [
            "documentation", "docs", "reference",
            "api", "manual", "guide", "tutorial",
            "how to", "example", "more info"
        ]
        return any(keyword in message.lower() for keyword in doc_keywords)
    
    async def _inject_documentation_links(self):
        """Inject relevant documentation links"""
        if not self.current_language:
            return
        
        docs = self.docs_references.get(self.current_language, {})
        if docs:
            # Find most relevant docs based on recent topics
            relevant_docs = self._get_relevant_docs(docs)
            
            docs_context = ContextToInject(
                information={
                    "documentation": relevant_docs,
                    "message": "Here are some helpful documentation links:"
                },
                timing=InjectionTiming.NEXT_TURN,
                priority=ContextPriority.MEDIUM,
                source="documentation"
            )
            self.context_weaver.add_context(docs_context)
    
    def _get_relevant_docs(self, docs: dict) -> dict:
        """Get relevant documentation based on context"""
        if not self.code_context_stack:
            return docs
        
        # Get recent topics
        recent_topics = [ctx["pattern"] for ctx in self.code_context_stack[-3:]]
        
        # Filter relevant docs
        relevant = {}
        for key, url in docs.items():
            if any(topic in key for topic in recent_topics):
                relevant[key] = url
        
        return relevant or docs  # Return all if no specific match
    
    def _is_advanced_user(self) -> bool:
        """Determine if user seems advanced based on conversation"""
        # Simple heuristic based on conversation history
        if len(self.code_context_stack) > 5:
            return True
        
        # Check for advanced keywords in recent context
        advanced_keywords = [
            "optimize", "performance", "concurrency",
            "architecture", "design pattern", "algorithm"
        ]
        # In real implementation, check conversation history
        return False
    
    async def _inject_debugging_help(self, error_message: str):
        """Inject debugging assistance for errors"""
        # Parse error type
        error_type = self._parse_error_type(error_message)
        
        debug_context = ContextToInject(
            information={
                "error_type": error_type,
                "debugging_steps": [
                    "1. Check the full stack trace",
                    "2. Verify input data and types",
                    "3. Add debug logging before the error",
                    "4. Test with minimal reproducible example"
                ],
                "common_causes": self._get_common_error_causes(error_type)
            },
            strategy={
                "approach": "systematic_debugging",
                "tone": "supportive"
            },
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH,
            source="debugging_assistant"
        )
        self.context_weaver.add_context(debug_context)
    
    def _parse_error_type(self, error_message: str) -> str:
        """Extract error type from message"""
        # Simple pattern matching
        if "TypeError" in error_message:
            return "TypeError"
        elif "ValueError" in error_message:
            return "ValueError"
        elif "AttributeError" in error_message:
            return "AttributeError"
        else:
            return "Unknown Error"
    
    def _get_common_error_causes(self, error_type: str) -> List[str]:
        """Get common causes for error types"""
        causes = {
            "TypeError": [
                "Passing wrong argument types to functions",
                "Calling methods on None objects",
                "Incorrect number of arguments"
            ],
            "ValueError": [
                "Invalid input data format",
                "Out of range values",
                "Incorrect conversion attempts"
            ],
            "AttributeError": [
                "Typo in attribute/method name",
                "Object doesn't have the attribute",
                "Accessing attribute before initialization"
            ]
        }
        return causes.get(error_type, ["Check the error message for details"])
    
    async def start_session(self):
        """Start a coding assistance session"""
        print("Code Assistant initialized!")
        print("I can help with code examples, debugging, and documentation.\n")
        
        await self.voice_engine.connect()
        
        # Initial context
        intro_context = ContextToInject(
            information={
                "greeting": "Hello! I'm your coding assistant. What language are you working with today?",
                "capabilities": [
                    "Code examples and snippets",
                    "Error debugging help",
                    "Documentation references",
                    "Best practices"
                ]
            },
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH
        )
        self.context_weaver.add_context(intro_context)


async def main():
    """Example usage"""
    assistant = CodeAssistant(api_key="your-api-key")
    await assistant.initialize()
    
    try:
        await assistant.start_session()
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await assistant.voxon.engine_coordinator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())