"""
Conversation Flow Detector

Detects conversation patterns and learns from successful context injections
to improve future injection decisions.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import deque, defaultdict
import statistics
from enum import Enum
from .base import BaseDetector, DetectionResult


class ConversationPhase(str, Enum):
    """Phases of a conversation"""
    GREETING = "greeting"
    INTRODUCTION = "introduction"
    MAIN_TOPIC = "main_topic"
    CLARIFICATION = "clarification"
    CONCLUSION = "conclusion"
    IDLE = "idle"


class FlowPattern:
    """Represents a detected conversation flow pattern"""
    def __init__(self, pattern_type: str, confidence: float, metadata: Dict[str, Any]):
        self.pattern_type = pattern_type
        self.confidence = confidence
        self.metadata = metadata
        self.timestamp = datetime.now()
        self.successful_injections = 0
        self.failed_injections = 0
    
    @property
    def success_rate(self) -> float:
        total = self.successful_injections + self.failed_injections
        return self.successful_injections / total if total > 0 else 0.5


class ConversationFlowDetector(BaseDetector):
    """
    Detects conversation flow patterns and learns from injection outcomes.
    
    Features:
    - Identifies conversation phases (greeting, main topic, conclusion)
    - Detects recurring patterns (Q&A cycles, topic transitions)
    - Learns from successful/failed injections
    - Provides pattern-based injection recommendations
    """
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        pattern_memory_size: int = 50,
        learning_rate: float = 0.1
    ):
        super().__init__(threshold=min_confidence)
        
        # Pattern recognition
        self.detected_patterns: deque = deque(maxlen=pattern_memory_size)
        self.pattern_library: Dict[str, FlowPattern] = {}
        
        # Conversation state
        self.current_phase = ConversationPhase.IDLE
        self.phase_history: List[Tuple[ConversationPhase, datetime]] = []
        self.message_count = 0
        self.turn_count = 0
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.injection_outcomes: deque = deque(maxlen=100)
        
        # Pattern detection state
        self.keywords_seen: Set[str] = set()
        self.topic_keywords: Dict[str, int] = defaultdict(int)
        self.question_count = 0
        self.statement_count = 0
        
        # Timing patterns
        self.phase_durations: Dict[ConversationPhase, List[float]] = defaultdict(list)
        self.transition_patterns: List[Tuple[ConversationPhase, ConversationPhase]] = []
        
        # Success tracking
        self.successful_patterns: Dict[str, float] = {}
        self.pattern_context_map: Dict[str, List[str]] = defaultdict(list)
    
    async def detect(self, state) -> DetectionResult:
        """Detect conversation flow patterns and injection opportunities"""
        
        # Update conversation metrics
        self._update_conversation_state(state)
        
        # Detect current phase
        current_phase = self._detect_conversation_phase(state)
        if current_phase != self.current_phase:
            self._record_phase_transition(self.current_phase, current_phase)
            self.current_phase = current_phase
        
        # Detect patterns
        patterns = self._detect_patterns(state)
        
        # Find best injection opportunity based on patterns
        best_opportunity = self._find_best_opportunity(patterns, state)
        
        if best_opportunity:
            return DetectionResult(
                detected=True,
                confidence=best_opportunity['confidence'],
                timestamp=datetime.now(),
                metadata={
                    "pattern_type": best_opportunity['pattern_type'],
                    "phase": self.current_phase.value,
                    "reason": best_opportunity['reason'],
                    "success_rate": best_opportunity.get('success_rate', 0.5),
                    "learned_pattern": best_opportunity.get('learned', False)
                }
            )
        
        # No immediate opportunity, provide analysis
        return DetectionResult(
            detected=False,
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={
                "current_phase": self.current_phase.value,
                "message_count": self.message_count,
                "turn_count": self.turn_count,
                "active_patterns": [p.pattern_type for p in patterns],
                "learning_progress": self._get_learning_progress()
            }
        )
    
    def _update_conversation_state(self, state):
        """Update internal conversation state"""
        # Track messages
        if hasattr(state, 'messages'):
            self.message_count = len(state.messages)
            
            # Analyze recent messages for keywords
            recent_messages = state.messages[-5:] if len(state.messages) > 5 else state.messages
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    self._analyze_message_content(msg.content)
        
        # Track turns
        if hasattr(state, 'turns'):
            self.turn_count = len(state.turns)
    
    def _analyze_message_content(self, content: str):
        """Analyze message content for patterns"""
        if not content:
            return
        
        content_lower = content.lower()
        
        # Detect questions
        if any(q in content_lower for q in ['?', 'what', 'why', 'how', 'when', 'where']):
            self.question_count += 1
        else:
            self.statement_count += 1
        
        # Extract keywords (simple approach)
        words = content_lower.split()
        for word in words:
            if len(word) > 4 and word.isalpha():  # Simple keyword filter
                self.keywords_seen.add(word)
                self.topic_keywords[word] += 1
    
    def _detect_conversation_phase(self, state) -> ConversationPhase:
        """Detect current conversation phase"""
        
        # Greeting phase detection
        if self.message_count <= 2:
            greeting_keywords = {'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon'}
            if any(kw in self.keywords_seen for kw in greeting_keywords):
                return ConversationPhase.GREETING
        
        # Introduction phase
        if self.message_count <= 5 and self.turn_count <= 2:
            intro_keywords = {'help', 'assist', 'need', 'want', 'looking for', 'trying to'}
            if any(kw in ' '.join(self.keywords_seen) for kw in intro_keywords):
                return ConversationPhase.INTRODUCTION
        
        # Clarification phase
        if self.question_count > self.statement_count * 1.5:
            return ConversationPhase.CLARIFICATION
        
        # Conclusion phase
        conclusion_keywords = {'thank', 'thanks', 'bye', 'goodbye', 'done', 'finished'}
        if any(kw in self.keywords_seen for kw in conclusion_keywords):
            return ConversationPhase.CONCLUSION
        
        # Main topic (default for active conversation)
        if self.message_count > 2:
            return ConversationPhase.MAIN_TOPIC
        
        return ConversationPhase.IDLE
    
    def _detect_patterns(self, state) -> List[FlowPattern]:
        """Detect active conversation patterns"""
        patterns = []
        
        # Q&A Pattern
        if self._detect_qa_pattern():
            patterns.append(FlowPattern(
                "qa_cycle",
                confidence=0.8,
                metadata={"question_ratio": self.question_count / max(1, self.message_count)}
            ))
        
        # Topic Transition Pattern
        topic_transition = self._detect_topic_transition()
        if topic_transition:
            patterns.append(FlowPattern(
                "topic_transition",
                confidence=topic_transition['confidence'],
                metadata=topic_transition
            ))
        
        # Repetition Pattern
        if self._detect_repetition_pattern():
            patterns.append(FlowPattern(
                "repetition",
                confidence=0.7,
                metadata={"type": "clarification_needed"}
            ))
        
        # Phase-specific patterns
        phase_pattern = self._detect_phase_pattern()
        if phase_pattern:
            patterns.append(phase_pattern)
        
        # Store detected patterns
        for pattern in patterns:
            self.detected_patterns.append(pattern)
            
            # Update pattern library
            key = f"{pattern.pattern_type}_{self.current_phase}"
            if key not in self.pattern_library:
                self.pattern_library[key] = pattern
        
        return patterns
    
    def _detect_qa_pattern(self) -> bool:
        """Detect question-answer cycle pattern"""
        if self.message_count < 4:
            return False
        
        # High ratio of questions indicates Q&A pattern
        question_ratio = self.question_count / self.message_count
        return question_ratio > 0.4
    
    def _detect_topic_transition(self) -> Optional[Dict[str, Any]]:
        """Detect topic transition pattern"""
        if len(self.topic_keywords) < 2:
            return None
        
        # Get top keywords by frequency
        sorted_keywords = sorted(self.topic_keywords.items(), key=lambda x: x[1], reverse=True)
        top_keywords = sorted_keywords[:5]
        
        # Check for keyword diversity (indicates multiple topics)
        if len(top_keywords) >= 3:
            frequencies = [kw[1] for kw in top_keywords]
            if max(frequencies) < sum(frequencies) * 0.5:  # No single dominant topic
                return {
                    'confidence': 0.7,
                    'topics': [kw[0] for kw in top_keywords],
                    'transition_likely': True
                }
        
        return None
    
    def _detect_repetition_pattern(self) -> bool:
        """Detect repetition pattern (clarification needed)"""
        if len(self.topic_keywords) < 5:
            return False
        
        # Check for repeated keywords (high frequency)
        max_frequency = max(self.topic_keywords.values())
        avg_frequency = sum(self.topic_keywords.values()) / len(self.topic_keywords)
        
        # High max frequency relative to average suggests repetition
        return max_frequency > avg_frequency * 3
    
    def _detect_phase_pattern(self) -> Optional[FlowPattern]:
        """Detect phase-specific patterns"""
        
        if self.current_phase == ConversationPhase.GREETING:
            # Greeting phase usually needs context about available services
            return FlowPattern(
                "greeting_context_needed",
                confidence=0.9,
                metadata={"phase": "greeting", "suggestion": "introduce_capabilities"}
            )
        
        elif self.current_phase == ConversationPhase.CLARIFICATION:
            # Clarification phase benefits from examples or additional info
            return FlowPattern(
                "clarification_assistance",
                confidence=0.8,
                metadata={"phase": "clarification", "suggestion": "provide_examples"}
            )
        
        elif self.current_phase == ConversationPhase.CONCLUSION:
            # Conclusion phase might need summary or next steps
            return FlowPattern(
                "conclusion_summary",
                confidence=0.7,
                metadata={"phase": "conclusion", "suggestion": "summarize_or_next_steps"}
            )
        
        return None
    
    def _find_best_opportunity(self, patterns: List[FlowPattern], state) -> Optional[Dict[str, Any]]:
        """Find best injection opportunity based on patterns and learning"""
        if not patterns:
            return None
        
        opportunities = []
        
        for pattern in patterns:
            # Check if we've learned about this pattern
            pattern_key = f"{pattern.pattern_type}_{self.current_phase}"
            
            # Get historical success rate
            success_rate = self.successful_patterns.get(pattern_key, 0.5)
            
            # Calculate opportunity score
            score = pattern.confidence * success_rate
            
            # Boost score for patterns that worked well recently
            if pattern_key in self.pattern_library:
                library_pattern = self.pattern_library[pattern_key]
                if library_pattern.success_rate > 0.7:
                    score *= 1.2
            
            opportunities.append({
                'pattern_type': pattern.pattern_type,
                'confidence': score,
                'reason': self._get_opportunity_reason(pattern),
                'success_rate': success_rate,
                'learned': pattern_key in self.successful_patterns
            })
        
        # Return best opportunity
        return max(opportunities, key=lambda x: x['confidence']) if opportunities else None
    
    def _get_opportunity_reason(self, pattern: FlowPattern) -> str:
        """Get human-readable reason for injection opportunity"""
        reasons = {
            'qa_cycle': "Natural pause in Q&A cycle",
            'topic_transition': "Topic transition detected",
            'repetition': "Clarification may be needed",
            'greeting_context_needed': "Initial context helpful after greeting",
            'clarification_assistance': "Additional information may help clarify",
            'conclusion_summary': "Good time for summary or next steps"
        }
        
        return reasons.get(pattern.pattern_type, "Pattern-based opportunity detected")
    
    def _record_phase_transition(self, from_phase: ConversationPhase, to_phase: ConversationPhase):
        """Record phase transition"""
        now = datetime.now()
        
        # Record phase duration
        if self.phase_history:
            last_phase, last_time = self.phase_history[-1]
            duration = (now - last_time).total_seconds()
            self.phase_durations[last_phase].append(duration)
        
        # Record transition
        self.phase_history.append((to_phase, now))
        self.transition_patterns.append((from_phase, to_phase))
    
    def record_injection_outcome(
        self, 
        pattern_type: str, 
        phase: ConversationPhase,
        success: bool,
        context_type: str,
        metadata: Dict[str, Any]
    ):
        """Record outcome of context injection for learning"""
        outcome = {
            'pattern_type': pattern_type,
            'phase': phase,
            'success': success,
            'context_type': context_type,
            'timestamp': datetime.now(),
            'metadata': metadata
        }
        
        self.injection_outcomes.append(outcome)
        
        # Update pattern success rates
        pattern_key = f"{pattern_type}_{phase}"
        
        # Exponential moving average for success rate
        current_rate = self.successful_patterns.get(pattern_key, 0.5)
        new_rate = current_rate * (1 - self.learning_rate) + (1.0 if success else 0.0) * self.learning_rate
        self.successful_patterns[pattern_key] = new_rate
        
        # Update pattern library
        if pattern_key in self.pattern_library:
            pattern = self.pattern_library[pattern_key]
            if success:
                pattern.successful_injections += 1
            else:
                pattern.failed_injections += 1
        
        # Track successful context types for patterns
        if success:
            self.pattern_context_map[pattern_key].append(context_type)
    
    def get_recommended_context_type(self, pattern_type: str, phase: ConversationPhase) -> Optional[str]:
        """Get recommended context type based on learning"""
        pattern_key = f"{pattern_type}_{phase}"
        
        if pattern_key in self.pattern_context_map:
            context_types = self.pattern_context_map[pattern_key]
            if context_types:
                # Return most common successful context type
                from collections import Counter
                return Counter(context_types).most_common(1)[0][0]
        
        # Default recommendations
        defaults = {
            'greeting_context_needed': 'capabilities_overview',
            'clarification_assistance': 'examples',
            'conclusion_summary': 'summary',
            'qa_cycle': 'detailed_answer',
            'topic_transition': 'topic_context'
        }
        
        return defaults.get(pattern_type)
    
    def _get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress metrics"""
        total_outcomes = len(self.injection_outcomes)
        successful_outcomes = sum(1 for o in self.injection_outcomes if o['success'])
        
        return {
            'total_injections': total_outcomes,
            'success_rate': successful_outcomes / total_outcomes if total_outcomes > 0 else 0,
            'patterns_learned': len(self.successful_patterns),
            'confidence_level': statistics.mean(self.successful_patterns.values()) if self.successful_patterns else 0.5
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            'current_phase': self.current_phase.value,
            'message_count': self.message_count,
            'turn_count': self.turn_count,
            'patterns_detected': len(self.detected_patterns),
            'patterns_learned': len(self.successful_patterns),
            'average_success_rate': statistics.mean(self.successful_patterns.values()) if self.successful_patterns else 0.5,
            'phase_distribution': {
                phase.value: len(durations) 
                for phase, durations in self.phase_durations.items()
            },
            'learning_progress': self._get_learning_progress()
        }