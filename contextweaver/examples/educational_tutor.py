#!/usr/bin/env python3
"""
Educational Tutor Example

Demonstrates how to use ContextWeaver for adaptive learning with
intelligent hint injection and progress tracking.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

from contextweaver import ContextWeaver
from contextweaver.strategies import AdaptiveStrategy
from contextweaver.detectors import (
    SilenceDetector,
    PauseDetector,
    ConversationFlowDetector,
    ConversationPhase
)
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority
from voxon import Voxon, VoxonConfig
from voxengine import VoiceEngine, VoiceEngineConfig


class DifficultyLevel(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3


class LearningTopic(Enum):
    MATH = "mathematics"
    SCIENCE = "science"
    HISTORY = "history"
    LANGUAGE = "language"


class EducationalTutor:
    """Adaptive AI tutor with intelligent hint and explanation injection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.student_profile = {
            "name": None,
            "level": DifficultyLevel.BEGINNER,
            "topics": [],
            "strengths": [],
            "weaknesses": [],
            "learning_style": "visual",  # visual, auditory, kinesthetic
            "progress": {}
        }
        
        self.current_topic = None
        self.current_question = None
        self.hint_level = 0  # 0: no hint, 1: subtle, 2: moderate, 3: explicit
        self.incorrect_attempts = 0
        
        # Learning content database
        self.curriculum = {
            LearningTopic.MATH: {
                DifficultyLevel.BEGINNER: [
                    {
                        "question": "What is 7 + 5?",
                        "answer": "12",
                        "hints": [
                            "Try counting up from 7",
                            "7 + 5 is the same as 7 + 3 + 2",
                            "Think: 7 + 3 = 10, then 10 + 2 = ?"
                        ],
                        "explanation": "When adding 7 + 5, we can break it down: 7 + 3 = 10, then 10 + 2 = 12"
                    },
                    {
                        "question": "What is 15 - 8?",
                        "answer": "7",
                        "hints": [
                            "Try counting backward from 15",
                            "15 - 5 = 10, so 15 - 8 = 10 - 3",
                            "Think: 15 - 5 - 3 = ?"
                        ],
                        "explanation": "We can solve 15 - 8 by breaking it down: 15 - 5 = 10, then 10 - 3 = 7"
                    }
                ],
                DifficultyLevel.INTERMEDIATE: [
                    {
                        "question": "What is 23 × 4?",
                        "answer": "92",
                        "hints": [
                            "Break it down: 20 × 4 + 3 × 4",
                            "20 × 4 = 80, and 3 × 4 = 12",
                            "80 + 12 = ?"
                        ],
                        "explanation": "Using the distributive property: 23 × 4 = (20 × 4) + (3 × 4) = 80 + 12 = 92"
                    }
                ]
            }
        }
        
        # Encouragement phrases
        self.encouragements = {
            "correct": [
                "Excellent work!",
                "That's exactly right!",
                "You're doing great!",
                "Fantastic job!"
            ],
            "incorrect": [
                "Not quite, but good try!",
                "Let's think about this differently.",
                "You're on the right track, keep going!",
                "Almost there, let's try again."
            ],
            "progress": [
                "You're improving so much!",
                "I can see you're getting better at this!",
                "Your hard work is paying off!",
                "You've mastered this concept!"
            ]
        }
    
    async def initialize(self):
        """Initialize all components"""
        # Voice engine with friendly settings
        self.voice_engine = VoiceEngine(VoiceEngineConfig(
            api_key=self.api_key,
            vad_type="client",
            voice="nova",  # Friendly voice
            language="en"
        ))
        
        # Context weaver with adaptive learning
        self.context_weaver = ContextWeaver(
            strategy=AdaptiveStrategy(
                initial_threshold=0.5,  # More responsive for education
                learning_rate=0.2       # Learn student patterns
            ),
            detectors=[
                SilenceDetector(
                    silence_threshold_ms=3000,  # Give time to think
                    enable_prediction=True
                ),
                PauseDetector(
                    pause_threshold_ms=1500     # Natural thinking pauses
                ),
                ConversationFlowDetector(
                    learning_rate=0.3           # Learn student patterns quickly
                )
            ]
        )
        
        # Voxon orchestrator
        self.voxon = Voxon(VoxonConfig())
        coordinator = self.voxon.engine_coordinator
        coordinator.voice_engine = self.voice_engine
        coordinator.context_engine = self.context_weaver
        
        await coordinator.initialize()
        self._setup_event_handlers()
        await self._load_initial_contexts()
    
    async def _load_initial_contexts(self):
        """Load educational contexts"""
        # Hint injection context
        hint_context = ContextToInject(
            information={
                "action": "provide_hint",
                "level": "adaptive"
            },
            timing=InjectionTiming.NEXT_PAUSE,
            priority=ContextPriority.MEDIUM,
            conditions={
                "silence_duration_ms": 5000,  # After 5s thinking
                "incorrect_attempts": 1        # After first wrong answer
            },
            source="hint_system"
        )
        self.context_weaver.add_context(hint_context)
        
        # Encouragement injection
        encouragement_context = ContextToInject(
            information={
                "action": "encourage",
                "type": "struggle"
            },
            timing=InjectionTiming.NEXT_PAUSE,
            priority=ContextPriority.HIGH,
            conditions={
                "incorrect_attempts": 2,
                "frustration_detected": True
            },
            source="encouragement_system"
        )
        self.context_weaver.add_context(encouragement_context)
        
        # Progress summary
        progress_context = ContextToInject(
            information={
                "action": "summarize_progress"
            },
            timing=InjectionTiming.ON_TOPIC,
            priority=ContextPriority.LOW,
            conditions={
                "questions_completed": 5,
                "phase": "conclusion"
            },
            source="progress_tracker"
        )
        self.context_weaver.add_context(progress_context)
    
    def _setup_event_handlers(self):
        """Set up event handlers"""
        @self.voice_engine.events.on('user.message')
        async def on_user_message(event):
            """Process student answers"""
            message = event.message.content
            
            if self.current_question:
                # Check answer
                is_correct = await self._check_answer(message)
                
                if is_correct:
                    await self._handle_correct_answer()
                else:
                    await self._handle_incorrect_answer(message)
            else:
                # Process general interaction
                await self._process_general_message(message)
        
        @self.voice_engine.events.on('conversation.phase_change')
        async def on_phase_change(event):
            """Adapt to conversation phase"""
            phase = event.phase
            
            if phase == ConversationPhase.GREETING:
                await self._inject_welcome_context()
            elif phase == ConversationPhase.CONCLUSION:
                await self._inject_session_summary()
    
    async def _check_answer(self, answer: str) -> bool:
        """Check if student's answer is correct"""
        if not self.current_question:
            return False
        
        correct_answer = self.current_question["answer"]
        
        # Normalize answers for comparison
        normalized_answer = answer.strip().lower()
        normalized_correct = correct_answer.strip().lower()
        
        # Allow for minor variations
        return normalized_answer == normalized_correct
    
    async def _handle_correct_answer(self):
        """Handle correct answer with positive reinforcement"""
        # Record success
        self._record_progress(True)
        
        # Choose encouragement
        encouragement = self._select_encouragement("correct")
        
        # Create immediate positive feedback
        feedback_context = ContextToInject(
            information={
                "feedback": encouragement,
                "explanation": self.current_question.get("explanation", ""),
                "next_action": "advance_to_next"
            },
            strategy={
                "tone": "celebratory",
                "energy": "high"
            },
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH,
            source="feedback_system"
        )
        self.context_weaver.add_context(feedback_context)
        
        # Reset for next question
        self.incorrect_attempts = 0
        self.hint_level = 0
        
        # Potentially adjust difficulty
        if self._should_increase_difficulty():
            await self._adjust_difficulty(1)
    
    async def _handle_incorrect_answer(self, answer: str):
        """Handle incorrect answer with supportive guidance"""
        self.incorrect_attempts += 1
        
        # Analyze error type
        error_analysis = self._analyze_error(answer)
        
        # Determine response based on attempts
        if self.incorrect_attempts == 1:
            # First attempt - gentle correction
            response_context = ContextToInject(
                information={
                    "feedback": self._select_encouragement("incorrect"),
                    "suggestion": "Let's try approaching this differently.",
                    "error_type": error_analysis
                },
                strategy={
                    "tone": "supportive",
                    "patience": "high"
                },
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.HIGH,
                source="error_handler"
            )
        elif self.incorrect_attempts >= 2:
            # Multiple attempts - provide hint
            hint = self._get_appropriate_hint()
            response_context = ContextToInject(
                information={
                    "feedback": "Let me give you a hint.",
                    "hint": hint,
                    "hint_level": self.hint_level
                },
                strategy={
                    "tone": "guiding",
                    "clarity": "high"
                },
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.HIGH,
                source="hint_system"
            )
            self.hint_level += 1
        
        self.context_weaver.add_context(response_context)
        
        # Record learning pattern
        self._record_error_pattern(error_analysis)
    
    def _analyze_error(self, incorrect_answer: str) -> str:
        """Analyze the type of error made"""
        if not self.current_question:
            return "unknown"
        
        correct = self.current_question["answer"]
        
        # Math-specific error analysis
        if self.current_topic == LearningTopic.MATH:
            try:
                correct_num = float(correct)
                incorrect_num = float(incorrect_answer)
                
                diff = abs(correct_num - incorrect_num)
                
                if diff == 1:
                    return "off_by_one"
                elif diff == 10:
                    return "place_value_error"
                elif incorrect_num == -correct_num:
                    return "sign_error"
                else:
                    return "calculation_error"
            except:
                return "format_error"
        
        return "conceptual_error"
    
    def _get_appropriate_hint(self) -> str:
        """Get hint based on current level and student needs"""
        if not self.current_question or "hints" not in self.current_question:
            return "Take your time and think about what we've learned."
        
        hints = self.current_question["hints"]
        
        # Ensure we don't exceed available hints
        hint_index = min(self.hint_level, len(hints) - 1)
        
        return hints[hint_index]
    
    def _select_encouragement(self, category: str) -> str:
        """Select appropriate encouragement phrase"""
        import random
        phrases = self.encouragements.get(category, ["Keep going!"])
        return random.choice(phrases)
    
    def _record_progress(self, success: bool):
        """Record student progress"""
        if not self.current_topic:
            return
        
        topic_key = self.current_topic.value
        
        if topic_key not in self.student_profile["progress"]:
            self.student_profile["progress"][topic_key] = {
                "attempts": 0,
                "correct": 0,
                "incorrect": 0,
                "hint_usage": 0,
                "avg_attempts": []
            }
        
        progress = self.student_profile["progress"][topic_key]
        progress["attempts"] += 1
        
        if success:
            progress["correct"] += 1
            progress["avg_attempts"].append(self.incorrect_attempts + 1)
        else:
            progress["incorrect"] += 1
        
        if self.hint_level > 0:
            progress["hint_usage"] += 1
        
        # Update learning detector
        for detector in self.context_weaver.detectors:
            if isinstance(detector, ConversationFlowDetector):
                detector.record_injection_outcome(
                    pattern_type="question_answer",
                    phase=detector.current_phase,
                    success=success,
                    context_type="educational",
                    metadata={
                        "topic": topic_key,
                        "difficulty": self.student_profile["level"].name,
                        "attempts": self.incorrect_attempts + 1
                    }
                )
    
    def _should_increase_difficulty(self) -> bool:
        """Determine if difficulty should increase"""
        if not self.current_topic:
            return False
        
        progress = self.student_profile["progress"].get(self.current_topic.value, {})
        
        # Simple heuristic: 80% success rate over last 5 questions
        if progress.get("attempts", 0) >= 5:
            recent_success_rate = progress["correct"] / progress["attempts"]
            return recent_success_rate >= 0.8
        
        return False
    
    async def _adjust_difficulty(self, change: int):
        """Adjust difficulty level"""
        current_value = self.student_profile["level"].value
        new_value = max(1, min(3, current_value + change))
        
        if new_value != current_value:
            self.student_profile["level"] = DifficultyLevel(new_value)
            
            # Inject notification
            level_context = ContextToInject(
                information={
                    "message": f"Great job! Let's try some {self.student_profile['level'].name.lower()} level questions.",
                    "new_level": self.student_profile["level"].name
                },
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.MEDIUM,
                source="difficulty_adjuster"
            )
            self.context_weaver.add_context(level_context)
    
    def _record_error_pattern(self, error_type: str):
        """Record error patterns for adaptive learning"""
        if error_type not in self.student_profile["weaknesses"]:
            self.student_profile["weaknesses"].append(error_type)
    
    async def _inject_welcome_context(self):
        """Inject welcoming context"""
        welcome_context = ContextToInject(
            information={
                "greeting": "Welcome back! Ready for another fun learning session?",
                "options": [
                    "Continue where we left off",
                    "Try a new topic",
                    "Review what we've learned"
                ]
            },
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH,
            source="welcome_system"
        )
        self.context_weaver.add_context(welcome_context)
    
    async def _inject_session_summary(self):
        """Inject session summary"""
        if not self.current_topic:
            return
        
        progress = self.student_profile["progress"].get(self.current_topic.value, {})
        
        if progress.get("attempts", 0) > 0:
            success_rate = progress["correct"] / progress["attempts"] * 100
            
            summary_context = ContextToInject(
                information={
                    "summary": f"Great session! You got {progress['correct']} out of {progress['attempts']} questions correct ({success_rate:.0f}%).",
                    "strengths": self._identify_strengths(),
                    "next_steps": self._suggest_next_steps()
                },
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.HIGH,
                source="progress_summary"
            )
            self.context_weaver.add_context(summary_context)
    
    def _identify_strengths(self) -> List[str]:
        """Identify student's strengths"""
        strengths = []
        
        for topic, progress in self.student_profile["progress"].items():
            if progress.get("attempts", 0) >= 3:
                success_rate = progress["correct"] / progress["attempts"]
                if success_rate >= 0.7:
                    strengths.append(f"Strong understanding of {topic}")
        
        return strengths or ["Great effort and persistence!"]
    
    def _suggest_next_steps(self) -> str:
        """Suggest next learning steps"""
        if self.student_profile["level"] == DifficultyLevel.BEGINNER:
            return "Keep practicing to move to intermediate level!"
        elif self.student_profile["level"] == DifficultyLevel.INTERMEDIATE:
            return "You're doing great! Ready for advanced challenges soon."
        else:
            return "Excellent mastery! Consider helping others learn."
    
    async def start_session(self, student_name: str, topic: LearningTopic):
        """Start a tutoring session"""
        self.student_profile["name"] = student_name
        self.current_topic = topic
        
        print(f"Starting tutoring session for {student_name} - Topic: {topic.value}")
        
        await self.voice_engine.connect()
        
        # Initial greeting
        greeting_context = ContextToInject(
            information={
                "greeting": f"Hi {student_name}! I'm excited to help you learn {topic.value} today.",
                "warm_up": "Let's start with something fun to get warmed up!"
            },
            timing=InjectionTiming.IMMEDIATE,
            priority=ContextPriority.HIGH
        )
        self.context_weaver.add_context(greeting_context)
        
        # Load first question
        await self._load_next_question()
    
    async def _load_next_question(self):
        """Load the next question based on level and topic"""
        questions = self.curriculum.get(self.current_topic, {}).get(
            self.student_profile["level"], []
        )
        
        if questions:
            # Simple selection - in real app would be smarter
            import random
            self.current_question = random.choice(questions)
            
            # Inject question
            question_context = ContextToInject(
                information={
                    "question": self.current_question["question"],
                    "topic": self.current_topic.value,
                    "level": self.student_profile["level"].name
                },
                timing=InjectionTiming.IMMEDIATE,
                priority=ContextPriority.HIGH,
                source="question_system"
            )
            self.context_weaver.add_context(question_context)


async def main():
    """Example usage"""
    tutor = EducationalTutor(api_key="your-api-key")
    await tutor.initialize()
    
    try:
        # Start a math tutoring session
        await tutor.start_session("Alex", LearningTopic.MATH)
        
        # Keep running
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        print("\nEnding tutoring session...")
    finally:
        # Show final progress
        print("\nSession Summary:")
        print(f"Student: {tutor.student_profile['name']}")
        print(f"Level: {tutor.student_profile['level'].name}")
        print(f"Progress: {tutor.student_profile['progress']}")
        
        await tutor.voxon.engine_coordinator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())