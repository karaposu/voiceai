"""
Topic Change Detector

Detects when conversation topic changes for context updates.
"""

from datetime import datetime
from typing import List, Optional, Set
from .base import BaseDetector, DetectionResult


class TopicChangeDetector(BaseDetector):
    """Detect topic changes in conversation"""
    
    def __init__(
        self,
        keywords_per_topic: int = 5,
        change_threshold: float = 0.7
    ):
        super().__init__()
        self.keywords_per_topic = keywords_per_topic
        self.change_threshold = change_threshold
        self.current_keywords: Set[str] = set()
        self.topic_history: List[Set[str]] = []
    
    async def detect(self, state) -> DetectionResult:
        """Detect if conversation topic has changed"""
        
        if not hasattr(state, 'messages') or len(state.messages) < 2:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"reason": "insufficient_messages"}
            )
        
        # Extract keywords from recent messages
        recent_keywords = self._extract_keywords(state.messages[-3:])
        
        if not self.current_keywords:
            # First topic
            self.current_keywords = recent_keywords
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"reason": "initial_topic"}
            )
        
        # Calculate topic similarity
        overlap = len(self.current_keywords & recent_keywords)
        total = len(self.current_keywords | recent_keywords)
        
        if total == 0:
            similarity = 1.0
        else:
            similarity = overlap / total
        
        topic_changed = similarity < (1 - self.change_threshold)
        
        if topic_changed:
            # Update topic
            self.topic_history.append(self.current_keywords)
            self.current_keywords = recent_keywords
            
            return DetectionResult(
                detected=True,
                confidence=1.0 - similarity,
                timestamp=datetime.now(),
                metadata={
                    "previous_keywords": list(self.topic_history[-1]) if self.topic_history else [],
                    "new_keywords": list(recent_keywords),
                    "similarity": similarity
                }
            )
        
        return DetectionResult(
            detected=False,
            confidence=1.0 - similarity,
            timestamp=datetime.now(),
            metadata={"similarity": similarity}
        )
    
    def _extract_keywords(self, messages: List) -> Set[str]:
        """Simple keyword extraction"""
        # This is a simplified version - in production you'd use NLP
        keywords = set()
        
        for msg in messages:
            if hasattr(msg, 'content'):
                # Extract longer words as potential keywords
                words = msg.content.lower().split()
                keywords.update(w for w in words if len(w) > 4)
        
        # Keep only top N keywords
        return set(list(keywords)[:self.keywords_per_topic])