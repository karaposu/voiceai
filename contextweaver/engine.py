"""
Context Weaver Engine

Main engine that weaves context into conversations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from .detectors import (
    BaseDetector, 
    DetectionResult,
    SilenceDetector,
    PauseDetector, 
    TopicChangeDetector
)
from .strategies import InjectionStrategy, ConservativeStrategy
from contextengine.schema import ContextToInject, InjectionTiming, ContextPriority


class ContextWeaver:
    """
    Manages context injection for conversations.
    
    Coordinates detectors and strategies to inject context
    at optimal moments without interrupting flow.
    """
    
    def __init__(
        self,
        strategy: Optional[InjectionStrategy] = None,
        detectors: Optional[List[BaseDetector]] = None,
        logger: Optional[logging.Logger] = None,
        parallel_detection: bool = True,
        detection_timeout_ms: int = 50
    ):
        self.logger = logger or logging.getLogger(__name__)
        
        # Use provided or default strategy
        self.strategy = strategy or ConservativeStrategy()
        
        # Use provided or default detectors
        self.detectors = detectors or [
            SilenceDetector(),
            PauseDetector(),
            TopicChangeDetector()
        ]
        
        # Context storage - now stores ContextToInject objects
        self.available_context: Dict[str, ContextToInject] = {}
        self.injection_queue: List[ContextToInject] = []
        
        # State
        self.is_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance settings
        self.parallel_detection = parallel_detection
        self.detection_timeout_ms = detection_timeout_ms
        
    async def start(self):
        """Start the context injection engine"""
        if self.is_active:
            return
            
        self.is_active = True
        self.logger.info("Context weaver started")
    
    async def stop(self):
        """Stop the context injection engine"""
        self.is_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            
        self.logger.info("Context weaver stopped")
    
    def add_context(self, context: ContextToInject):
        """
        Add a context for potential injection.
        
        Args:
            context: ContextToInject object to add
        """
        self.available_context[context.context_id] = context
        self.logger.debug(f"Added context: {context.context_id} with priority {context.priority_value}")
    
    def add_raw_context(
        self,
        information: Optional[Dict[str, Any]] = None,
        strategy: Optional[Dict[str, Any]] = None,
        attention: Optional[Dict[str, Any]] = None,
        timing: InjectionTiming = InjectionTiming.NEXT_PAUSE,
        priority: ContextPriority = ContextPriority.MEDIUM,
        **kwargs
    ) -> ContextToInject:
        """
        Create and add context from raw data.
        
        Returns:
            The created ContextToInject
        """
        context = ContextToInject(
            information=information or {},
            strategy=strategy or {},
            attention=attention or {},
            timing=timing,
            priority=priority,
            **kwargs
        )
        self.add_context(context)
        return context
    
    def remove_context(self, context_id: str):
        """Remove context by ID"""
        self.available_context.pop(context_id, None)
    
    async def check_injection(self, state: Any) -> Optional[ContextToInject]:
        """
        Check if context should be injected.
        
        Args:
            state: Current conversation state
            
        Returns:
            ContextToInject object or None
        """
        if not self.is_active:
            return None
            
        # Clean expired contexts
        self._clean_expired_contexts()
        
        # Choose detection mode
        if self.parallel_detection:
            detections = await self._run_parallel_detection(state)
        else:
            detections = await self._run_sequential_detection(state)
        
        # Let strategy decide
        decision = await self.strategy.decide(
            detections=detections,
            state=state,
            available_context=self.available_context
        )
        
        if decision.should_inject and decision.context_to_inject:
            self.logger.info(f"Injecting context: {decision.reason}")
            self.strategy.record_injection()
            decision.context_to_inject.mark_injected()
            return decision.context_to_inject
            
        return None
    
    async def _run_parallel_detection(self, state: Any) -> List[DetectionResult]:
        """Run all detectors in parallel"""
        detection_tasks = []
        detector_names = []
        
        for detector in self.detectors:
            detector_names.append(detector.__class__.__name__)
            # Create task with timeout to prevent hanging
            task = self._detect_with_timeout(detector, state, self.detection_timeout_ms)
            detection_tasks.append(task)
        
        # Execute all detections in parallel
        start_time = asyncio.get_event_loop().time()
        detection_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        detection_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Process results
        detections = []
        for i, result in enumerate(detection_results):
            if isinstance(result, Exception):
                self.logger.error(f"Detector {detector_names[i]} failed: {result}")
                # Add failed detection result
                detections.append(DetectionResult(
                    detected=False,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={"error": str(result), "detector": detector_names[i]}
                ))
            else:
                detections.append(result)
        
        self.logger.debug(f"Parallel detection completed in {detection_time:.2f}ms with {len(detections)} detectors")
        return detections
    
    async def _run_sequential_detection(self, state: Any) -> List[DetectionResult]:
        """Run all detectors sequentially (fallback mode)"""
        detections = []
        start_time = asyncio.get_event_loop().time()
        
        for detector in self.detectors:
            try:
                result = await detector.detect(state)
                detections.append(result)
            except Exception as e:
                self.logger.error(f"Detector {detector.__class__.__name__} failed: {e}")
                detections.append(DetectionResult(
                    detected=False,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    metadata={"error": str(e), "detector": detector.__class__.__name__}
                ))
        
        detection_time = (asyncio.get_event_loop().time() - start_time) * 1000
        self.logger.debug(f"Sequential detection completed in {detection_time:.2f}ms with {len(detections)} detectors")
        return detections
    
    async def _detect_with_timeout(self, detector: BaseDetector, state: Any, timeout_ms: int) -> DetectionResult:
        """
        Run detector with timeout to prevent blocking.
        
        Args:
            detector: Detector to run
            state: Current state
            timeout_ms: Timeout in milliseconds
            
        Returns:
            DetectionResult
        """
        try:
            return await asyncio.wait_for(
                detector.detect(state),
                timeout=timeout_ms / 1000  # Convert to seconds
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Detector {detector.__class__.__name__} timed out after {timeout_ms}ms")
            return DetectionResult(
                detected=False,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"error": "timeout", "timeout_ms": timeout_ms}
            )
    
    def get_relevant_contexts(
        self,
        state: Any,
        max_items: int = 5
    ) -> List[ContextToInject]:
        """
        Get most relevant contexts for current state.
        
        Args:
            state: Current conversation state
            max_items: Maximum context items to return
            
        Returns:
            List of relevant ContextToInject objects
        """
        # Filter contexts that should inject
        injectable = [
            ctx for ctx in self.available_context.values()
            if ctx.should_inject(state.__dict__ if hasattr(state, '__dict__') else state)
        ]
        
        # Sort by priority and timing urgency
        timing_scores = {
            InjectionTiming.IMMEDIATE: 10,
            InjectionTiming.NEXT_TURN: 8,
            InjectionTiming.NEXT_PAUSE: 6,
            InjectionTiming.ON_TOPIC: 5,
            InjectionTiming.ON_TRIGGER: 4,
            InjectionTiming.SCHEDULED: 3,
            InjectionTiming.LAZY: 2,
            InjectionTiming.MANUAL: 0
        }
        
        injectable.sort(
            key=lambda ctx: ctx.priority_value + timing_scores.get(ctx.timing, 0),
            reverse=True
        )
        
        return injectable[:max_items]
    
    def _clean_expired_contexts(self):
        """Remove expired contexts"""
        expired_ids = [
            ctx_id for ctx_id, ctx in self.available_context.items()
            if ctx.is_expired() or 
               (ctx.max_injections and ctx.injection_count >= ctx.max_injections)
        ]
        
        for ctx_id in expired_ids:
            self.available_context.pop(ctx_id)
            self.logger.debug(f"Removed expired context: {ctx_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        # Count contexts by timing
        timing_counts = {}
        for ctx in self.available_context.values():
            timing_counts[ctx.timing.value] = timing_counts.get(ctx.timing.value, 0) + 1
        
        # Count contexts by priority
        priority_counts = {}
        for ctx in self.available_context.values():
            priority_counts[ctx.priority.name if isinstance(ctx.priority, ContextPriority) else str(ctx.priority)] = \
                priority_counts.get(ctx.priority.name if isinstance(ctx.priority, ContextPriority) else str(ctx.priority), 0) + 1
        
        return {
            "is_active": self.is_active,
            "detector_count": len(self.detectors),
            "context_items": len(self.available_context),
            "strategy": self.strategy.name,
            "injection_rate_per_min": self.strategy.get_injection_rate(60),
            "timing_distribution": timing_counts,
            "priority_distribution": priority_counts
        }