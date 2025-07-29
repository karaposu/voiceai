"""
Conversation Templates

Pre-defined conversation patterns and flows.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority


@dataclass
class ConversationTemplate:
    """
    Template for a type of conversation.
    
    Defines initial context, behavior strategies, and flow patterns.
    """
    name: str
    description: str
    initial_context: Dict[str, Any]
    strategy: Dict[str, Any]
    attention: Dict[str, Any]
    
    def to_context(self) -> ContextToInject:
        """Convert template to initial context"""
        return ContextToInject(
            information=self.initial_context,
            strategy=self.strategy,
            attention=self.attention,
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH,
            source=f"template_{self.name}"
        )


# Pre-defined templates

CUSTOMER_SUPPORT = ConversationTemplate(
    name="customer_support",
    description="Customer support conversation",
    initial_context={
        "role": "support_agent",
        "department": "general",
        "capabilities": ["order_lookup", "troubleshooting", "escalation"]
    },
    strategy={
        "tone": "helpful",
        "patience": "high",
        "empathy": "high",
        "solution_focused": True
    },
    attention={
        "listen_for": ["problem", "issue", "complaint"],
        "goal": "resolve_customer_issue",
        "escalate_if": ["angry", "legal", "refund"]
    }
)

SALES_ASSISTANT = ConversationTemplate(
    name="sales_assistant",
    description="Sales and product recommendation",
    initial_context={
        "role": "sales_assistant",
        "expertise": "product_knowledge",
        "inventory_access": True
    },
    strategy={
        "tone": "enthusiastic",
        "persuasion": "gentle",
        "personalization": "high",
        "upsell": "appropriate"
    },
    attention={
        "discover": ["needs", "preferences", "budget"],
        "goal": "match_customer_needs",
        "recommend": True
    }
)

GENERAL_ASSISTANT = ConversationTemplate(
    name="general",
    description="General purpose assistant",
    initial_context={
        "role": "assistant",
        "capabilities": "general"
    },
    strategy={
        "tone": "friendly",
        "helpfulness": "high",
        "clarity": "high"
    },
    attention={
        "adapt_to": "user_needs",
        "goal": "be_helpful"
    }
)

TECHNICAL_SUPPORT = ConversationTemplate(
    name="technical_support",
    description="Technical troubleshooting",
    initial_context={
        "role": "tech_support",
        "expertise": "technical",
        "diagnostic_tools": True
    },
    strategy={
        "tone": "professional",
        "technical_level": "adaptive",
        "step_by_step": True,
        "patience": "very_high"
    },
    attention={
        "diagnose": ["symptoms", "error_messages", "system_info"],
        "goal": "fix_technical_issue",
        "document": True
    }
)

EDUCATOR = ConversationTemplate(
    name="educator",
    description="Educational and tutoring",
    initial_context={
        "role": "educator",
        "teaching_style": "adaptive",
        "subjects": "general"
    },
    strategy={
        "tone": "encouraging",
        "pace": "student_led",
        "explanation": "clear",
        "examples": "frequent"
    },
    attention={
        "assess": ["knowledge_level", "learning_style"],
        "goal": "facilitate_learning",
        "check_understanding": True
    }
)


# Template registry
TEMPLATES: Dict[str, ConversationTemplate] = {
    "customer_support": CUSTOMER_SUPPORT,
    "sales_assistant": SALES_ASSISTANT,
    "general": GENERAL_ASSISTANT,
    "technical_support": TECHNICAL_SUPPORT,
    "educator": EDUCATOR
}


def get_template(name: str) -> Optional[ConversationTemplate]:
    """Get a template by name"""
    return TEMPLATES.get(name)


def list_templates() -> List[str]:
    """List available template names"""
    return list(TEMPLATES.keys())


def register_template(template: ConversationTemplate):
    """Register a custom template"""
    TEMPLATES[template.name] = template