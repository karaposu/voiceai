#!/usr/bin/env python3
"""
Customer Support Bot Example

Demonstrates how to use ContextWeaver for intelligent context injection
in a customer support scenario.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

from contextweaver import ContextWeaver
from contextweaver.strategies import AdaptiveStrategy
from contextweaver.detectors import (
    SilenceDetector,
    ConversationFlowDetector,
    ResponseTimingDetector
)
from contextweaver.schema import ContextToInject, InjectionTiming, ContextPriority
from voxon import Voxon, VoxonConfig
from voxengine import VoiceEngine, VoiceEngineConfig


class CustomerSupportBot:
    """AI-powered customer support bot with intelligent context injection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.context_weaver = None
        self.voxon = None
        self.voice_engine = None
        
        # Support knowledge base
        self.knowledge_base = {
            "return_policy": {
                "information": {
                    "policy": "30-day return policy for all items",
                    "conditions": "Items must be unused and in original packaging",
                    "process": "Initiate return through your account or contact support"
                },
                "keywords": ["return", "refund", "exchange", "policy"]
            },
            "shipping_info": {
                "information": {
                    "standard": "5-7 business days, free over $50",
                    "express": "2-3 business days, $15",
                    "overnight": "Next business day, $30"
                },
                "keywords": ["shipping", "delivery", "ship", "arrive"]
            },
            "technical_support": {
                "information": {
                    "hours": "24/7 technical support available",
                    "contact": "tech@example.com or 1-800-TECH",
                    "remote": "Remote assistance available"
                },
                "keywords": ["technical", "broken", "not working", "help", "issue"]
            }
        }
        
        # Escalation triggers
        self.escalation_triggers = [
            "speak to human",
            "manager",
            "supervisor",
            "real person",
            "frustrated",
            "angry",
            "unacceptable"
        ]
    
    async def initialize(self):
        """Initialize all components"""
        # Set up voice engine
        self.voice_engine = VoiceEngine(VoiceEngineConfig(
            api_key=self.api_key,
            vad_type="client",  # Better for support conversations
            voice="alloy",
            language="en"
        ))
        
        # Set up context weaver with support-optimized detectors
        self.context_weaver = ContextWeaver(
            strategy=AdaptiveStrategy(
                initial_threshold=0.6,  # Balanced approach
                learning_rate=0.15      # Learn from interactions
            ),
            detectors=[
                SilenceDetector(
                    silence_threshold_ms=1500,  # Allow thinking time
                    enable_prediction=True
                ),
                ConversationFlowDetector(
                    learning_rate=0.2  # Learn conversation patterns
                ),
                ResponseTimingDetector(
                    prediction_window_ms=800  # Wider window for support
                )
            ]
        )
        
        # Set up Voxon orchestrator
        self.voxon = Voxon(VoxonConfig())
        coordinator = self.voxon.engine_coordinator
        coordinator.voice_engine = self.voice_engine
        coordinator.context_engine = self.context_weaver
        
        # Initialize with VAD awareness
        await coordinator.initialize()
        
        # Load initial contexts
        await self._load_support_contexts()
        
        # Set up event handlers
        self._setup_event_handlers()
    
    async def _load_support_contexts(self):
        """Load customer support contexts"""
        # Add knowledge base contexts
        for topic, data in self.knowledge_base.items():
            context = ContextToInject(
                information=data["information"],
                timing=InjectionTiming.ON_TOPIC,
                priority=ContextPriority.HIGH,
                conditions={"keywords": data["keywords"]},
                source="knowledge_base",
                ttl_seconds=3600  # Refresh hourly
            )
            self.context_weaver.add_context(context)
        
        # Add escalation context
        escalation_context = ContextToInject(
            information={
                "action": "escalate_to_human",
                "message": "I understand you'd like to speak with a human agent. Let me transfer you right away.",
                "transfer_queue": "tier2_support"
            },
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.CRITICAL,
            conditions={"keywords": self.escalation_triggers},
            source="escalation_system"
        )
        self.context_weaver.add_context(escalation_context)
        
        # Add proactive help context
        proactive_help = ContextToInject(
            information={
                "suggestions": [
                    "I can help with returns, shipping, or technical issues.",
                    "Would you like to know about our current promotions?",
                    "I can check your order status if you have an order number."
                ]
            },
            timing=InjectionTiming.NEXT_PAUSE,
            priority=ContextPriority.LOW,
            conditions={"silence_duration_ms": 3000},  # After 3s silence
            source="proactive_assistance"
        )
        self.context_weaver.add_context(proactive_help)
    
    def _setup_event_handlers(self):
        """Set up event handlers for the voice engine"""
        @self.voice_engine.events.on('conversation.updated')
        async def on_conversation_update(event):
            """Handle conversation updates"""
            # Check for context injection opportunity
            state = event.conversation_state
            context = await self.context_weaver.check_injection(state)
            
            if context:
                await self._inject_context(context)
        
        @self.voice_engine.events.on('user.message')
        async def on_user_message(event):
            """Analyze user messages for support needs"""
            message = event.message.content.lower()
            
            # Check for order numbers
            if self._contains_order_number(message):
                await self._inject_order_context(message)
            
            # Check for frustration indicators
            frustration_level = self._analyze_frustration(message)
            if frustration_level > 0.7:
                await self._inject_empathy_context()
    
    async def _inject_context(self, context: ContextToInject):
        """Inject context into the conversation"""
        info = context.information
        
        # Handle different context types
        if context.source == "escalation_system":
            # Immediate escalation
            await self.voice_engine.send_text(info["message"])
            await self._transfer_to_human(info["transfer_queue"])
            
        elif context.source == "knowledge_base":
            # Inject relevant information naturally
            response = self._format_kb_response(info)
            await self.voice_engine.send_text(response)
            
        elif context.source == "proactive_assistance":
            # Offer help
            suggestion = info["suggestions"][0]  # Pick most relevant
            await self.voice_engine.send_text(suggestion)
    
    def _format_kb_response(self, info: Dict[str, Any]) -> str:
        """Format knowledge base information naturally"""
        # Convert structured data to natural language
        if "policy" in info:
            return f"Regarding our return policy: {info['policy']}. {info.get('conditions', '')}"
        elif "standard" in info:
            return f"For shipping, we offer: Standard shipping ({info['standard']}), " \
                   f"Express ({info['express']}), and Overnight ({info['overnight']})."
        else:
            # Generic formatting
            return " ".join(f"{k.replace('_', ' ').title()}: {v}" 
                          for k, v in info.items() if isinstance(v, str))
    
    def _contains_order_number(self, message: str) -> bool:
        """Check if message contains an order number"""
        import re
        # Order number pattern: ORD-XXXXXX
        return bool(re.search(r'ORD-\d{6}', message.upper()))
    
    async def _inject_order_context(self, message: str):
        """Inject order-specific context"""
        import re
        order_match = re.search(r'ORD-\d{6}', message.upper())
        if order_match:
            order_number = order_match.group()
            
            # In real implementation, fetch from database
            order_context = ContextToInject(
                information={
                    "order_number": order_number,
                    "status": "In transit",
                    "delivery_date": "Tomorrow by 5 PM",
                    "tracking": "1Z999AA1234567890"
                },
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.HIGH,
                source="order_system"
            )
            self.context_weaver.add_context(order_context)
    
    def _analyze_frustration(self, message: str) -> float:
        """Analyze frustration level (0-1)"""
        frustration_words = [
            "frustrated", "angry", "annoyed", "upset",
            "terrible", "awful", "horrible", "worst",
            "unacceptable", "ridiculous"
        ]
        
        word_count = len(message.split())
        frustration_count = sum(1 for word in frustration_words 
                               if word in message.lower())
        
        # Simple scoring
        base_score = frustration_count / max(word_count, 1)
        
        # Check for caps (shouting)
        caps_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)
        
        return min(1.0, base_score + (caps_ratio * 0.5))
    
    async def _inject_empathy_context(self):
        """Inject empathy and de-escalation context"""
        empathy_context = ContextToInject(
            information={
                "tone": "empathetic",
                "acknowledgment": "I understand this is frustrating",
                "commitment": "I'm here to help resolve this quickly"
            },
            strategy={
                "approach": "active_listening",
                "pace": "slower",
                "validation": True
            },
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH,
            source="empathy_system"
        )
        self.context_weaver.add_context(empathy_context)
    
    async def _transfer_to_human(self, queue: str):
        """Transfer to human agent"""
        # Record learning outcome
        if hasattr(self.context_weaver.detectors[1], 'record_injection_outcome'):
            self.context_weaver.detectors[1].record_injection_outcome(
                pattern_type="escalation",
                phase=self.context_weaver.detectors[1].current_phase,
                success=True,
                context_type="human_transfer",
                metadata={"queue": queue}
            )
        
        # In real implementation, initiate transfer
        print(f"[SYSTEM] Transferring to {queue}...")
    
    async def start_conversation(self):
        """Start a support conversation"""
        print("Customer Support Bot initialized and ready!")
        print("The bot will intelligently inject relevant information based on conversation flow.\n")
        
        # Start the conversation
        await self.voice_engine.connect()
        
        # Initial greeting with context
        greeting_context = ContextToInject(
            information={
                "greeting": "Hello! I'm here to help with any questions about orders, returns, or technical support.",
                "name": "Support Assistant"
            },
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH
        )
        self.context_weaver.add_context(greeting_context)
    
    async def stop(self):
        """Clean shutdown"""
        await self.voxon.engine_coordinator.shutdown()


async def main():
    """Example usage"""
    # Initialize bot
    bot = CustomerSupportBot(api_key="your-api-key-here")
    await bot.initialize()
    
    try:
        # Start conversation
        await bot.start_conversation()
        
        # Keep running
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())