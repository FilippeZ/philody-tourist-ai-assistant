"""orchestrator/observability.py - LLM Observability & LangSmith Tracing Layer.

Provides enterprise-grade observability, distributed tracing, and telemetry for:
1. RAG retrieval chains (vector searches, contextual chunking, scores)
2. Tool calls (Weather, Feasibility, Transit, Ticketing, Wearable IoT)
3. LangGraph State Graph workflows & Reflection Loops
4. Pydantic Output Parsers & Fallback/Retry chains

Seamlessly integrates with LangSmith when configured via environment variables
(LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT),
while maintaining an in-memory high-resolution trace buffer for local telemetry and REST API exposition.
"""

from __future__ import annotations

import functools
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Check LangSmith availability
_LANGSMITH_AVAILABLE = False
try:
    import langsmith
    from langsmith import Client, traceable
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    traceable = None


@dataclass
class ObservabilitySpan:
    """Represents an individual step/call inside an execution trace."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    span_type: str  # "chain", "tool", "retriever", "llm", "parser", "reflection"
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    status: str = "running"  # "running", "success", "error"

    def finish(self, outputs: Optional[Any] = None, error: Optional[Exception] = None):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)
        if error:
            self.status = "error"
            self.error = str(error)
        else:
            self.status = "success"
            if isinstance(outputs, dict):
                self.outputs = outputs
            elif outputs is not None:
                self.outputs = {"result": str(outputs)[:500]}


@dataclass
class ObservabilityTrace:
    """Represents a full end-to-end user request trace."""
    trace_id: str
    session_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_ms: Optional[float] = None
    spans: List[ObservabilitySpan] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"

    def finish(self, status: str = "success"):
        self.end_time = time.time()
        self.total_duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_ms": self.total_duration_ms,
            "status": self.status,
            "tags": self.tags,
            "metadata": self.metadata,
            "spans_count": len(self.spans),
            "spans": [asdict(s) for s in self.spans],
        }


class ObservabilityHub:
    """
    Central hub managing LangSmith tracing and in-memory trace analytics.
    """

    def __init__(self):
        self.project_name = os.environ.get("LANGSMITH_PROJECT", "athens-ai-tourist-assistant")
        self.langsmith_enabled = (
            _LANGSMITH_AVAILABLE
            and os.environ.get("LANGSMITH_TRACING", "false").lower() in ("true", "1")
            and bool(os.environ.get("LANGSMITH_API_KEY"))
        )
        self.client: Optional[Any] = None
        if self.langsmith_enabled:
            try:
                self.client = Client()
                logger.info("[Observability] LangSmith tracing enabled for project '%s'", self.project_name)
            except Exception as e:
                logger.warning("[Observability] Failed to initialize LangSmith Client: %s", e)
                self.langsmith_enabled = False

        self._active_traces: Dict[str, ObservabilityTrace] = {}
        self._completed_traces: List[ObservabilityTrace] = []
        self._max_history = 100

    def start_trace(self, name: str, session_id: str = "default_session", tags: Optional[List[str]] = None) -> ObservabilityTrace:
        trace_id = str(uuid.uuid4())
        trace = ObservabilityTrace(
            trace_id=trace_id,
            session_id=session_id,
            name=name,
            start_time=time.time(),
            tags=tags or ["athens-tourist-agent", "rag"],
        )
        self._active_traces[trace_id] = trace
        return trace

    def finish_trace(self, trace_id: str, status: str = "success") -> Optional[ObservabilityTrace]:
        trace = self._active_traces.pop(trace_id, None)
        if trace:
            trace.finish(status=status)
            self._completed_traces.append(trace)
            if len(self._completed_traces) > self._max_history:
                self._completed_traces.pop(0)
            return trace
        return None

    def create_span(
        self,
        trace_id: str,
        name: str,
        span_type: str,
        inputs: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ObservabilitySpan:
        span_id = str(uuid.uuid4())[:8]
        span = ObservabilitySpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            span_type=span_type,
            start_time=time.time(),
            inputs=inputs or {},
            metadata=metadata or {},
        )
        if trace_id in self._active_traces:
            self._active_traces[trace_id].spans.append(span)
        return span

    def get_status(self) -> Dict[str, Any]:
        """Returns the operational status of the observability layer."""
        return {
            "langsmith_available": _LANGSMITH_AVAILABLE,
            "langsmith_tracing_enabled": self.langsmith_enabled,
            "langsmith_project": self.project_name,
            "active_traces_count": len(self._active_traces),
            "completed_traces_count": len(self._completed_traces),
        }

    def get_recent_traces(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Returns recent execution traces for monitoring endpoints."""
        traces = self._completed_traces[-limit:]
        return [t.to_dict() for t in reversed(traces)]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Calculates aggregate latency and error rates across recorded traces."""
        if not self._completed_traces:
            return {
                "total_traces": 0,
                "avg_latency_ms": 0.0,
                "error_rate_pct": 0.0,
                "tool_calls_breakdown": {},
                "reflection_loops_triggered": 0,
            }

        total = len(self._completed_traces)
        durations = [t.total_duration_ms for t in self._completed_traces if t.total_duration_ms is not None]
        errors = [t for t in self._completed_traces if t.status == "error"]

        tool_calls: Dict[str, int] = {}
        reflection_loops = 0

        for t in self._completed_traces:
            for s in t.spans:
                if s.span_type == "tool":
                    tool_calls[s.name] = tool_calls.get(s.name, 0) + 1
                elif s.span_type == "reflection" or "reflection" in s.name.lower():
                    reflection_loops += 1

        avg_latency = round(sum(durations) / len(durations), 2) if durations else 0.0
        err_pct = round((len(errors) / total) * 100.0, 2)

        return {
            "total_traces": total,
            "avg_latency_ms": avg_latency,
            "error_rate_pct": err_pct,
            "tool_calls_breakdown": tool_calls,
            "reflection_loops_triggered": reflection_loops,
        }


# Singleton Observability Hub
observability_hub = ObservabilityHub()


# ---------------------------------------------------------------------------
# Decorators for Tracing Tools, RAG Chains & Reflection Nodes
# ---------------------------------------------------------------------------
def trace_tool_call(tool_name: str):
    """Decorator to automatically trace external tool invocations."""
    def decorator(func: Callable):
        # Native LangSmith traceable if enabled
        target_func = func
        if _LANGSMITH_AVAILABLE and traceable is not None and observability_hub.langsmith_enabled:
            target_func = traceable(run_type="tool", name=tool_name)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = kwargs.pop("_trace_id", None) or getattr(observability_hub, "_current_trace_id", "local_trace")
            span = observability_hub.create_span(
                trace_id=trace_id,
                name=tool_name,
                span_type="tool",
                inputs={"args": [str(a)[:100] for a in args], "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}},
            )
            try:
                result = target_func(*args, **kwargs)
                span.finish(outputs=result)
                return result
            except Exception as e:
                span.finish(error=e)
                raise
        return wrapper
    return decorator


def trace_rag_chain(name: str = "rag_retrieval_chain"):
    """Decorator to trace vector search retrieval operations."""
    def decorator(func: Callable):
        target_func = func
        if _LANGSMITH_AVAILABLE and traceable is not None and observability_hub.langsmith_enabled:
            target_func = traceable(run_type="retriever", name=name)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = kwargs.pop("_trace_id", None) or getattr(observability_hub, "_current_trace_id", "local_trace")
            span = observability_hub.create_span(
                trace_id=trace_id,
                name=name,
                span_type="retriever",
                inputs={"query": kwargs.get("query") or (args[1] if len(args) > 1 else str(args))},
            )
            try:
                result = target_func(*args, **kwargs)
                doc_count = len(result) if isinstance(result, list) else 1
                span.finish(outputs={"retrieved_docs_count": doc_count})
                return result
            except Exception as e:
                span.finish(error=e)
                raise
        return wrapper
    return decorator


def trace_graph_node(node_name: str):
    """Decorator to trace LangGraph state nodes."""
    def decorator(func: Callable):
        target_func = func
        if _LANGSMITH_AVAILABLE and traceable is not None and observability_hub.langsmith_enabled:
            target_func = traceable(run_type="chain", name=node_name)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract state whether func is a standalone function or a bound method
            if len(args) > 1 and isinstance(args[1], dict):
                state = args[1]
            elif len(args) > 0 and isinstance(args[0], dict):
                state = args[0]
            else:
                state = kwargs.get("state", {})
            trace_id = state.get("trace_id") if isinstance(state, dict) else getattr(observability_hub, "_current_trace_id", "local_trace")
            span = observability_hub.create_span(
                trace_id=trace_id or "local_trace",
                name=node_name,
                span_type="chain",
                inputs={"session_id": state.get("session_id") if isinstance(state, dict) else None, "intent": state.get("intent") if isinstance(state, dict) else None},
            )
            try:
                result = target_func(*args, **kwargs)
                span.finish(outputs={"status": "completed"})
                return result
            except Exception as e:
                span.finish(error=e)
                raise
        return wrapper
    return decorator
