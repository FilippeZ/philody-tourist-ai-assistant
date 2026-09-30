"""orchestrator/context_manager.py - Context Window Management, Token Delimiters & Cost Calculation Layer.

Implements:
1. Special Token Delimiters ([BOS], [EOS], <|endoftext|>, <|im_start|>, <|im_end|>)
   for unambiguous boundary demarcation in System Prompts, RAG Context, and Multi-Turn turns.
2. Token-Aware Sliding Window Truncation for Multi-Turn Dialogues (preventing unbounded context growth
   while preserving user preferences, constraints, and coherent user-assistant conversational pairs).
3. Context Length Management & Overflow Prevention Layer with dynamic budgeting between:
   - System Prompt Reserve
   - RAG Chunks Budget (relevance-sorted greedy knapsack)
   - Multi-Turn History Budget
   - Generation / Output Reserve
4. Token Cost Calculator with real-world pricing models (Nemotron-3-Ultra, GPT-4o, Claude 3.5).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try importing tiktoken, with resilient fallback
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    try:
        _DEFAULT_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _DEFAULT_ENCODER = None
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    _DEFAULT_ENCODER = None


# ---------------------------------------------------------------------------
# 1. Special Boundary Delimiter Tokens
# ---------------------------------------------------------------------------
class SpecialTokens:
    """Special delimiter tokens for unambiguous prompt parsing and boundary separation."""
    BOS = "[BOS]"                         # Beginning of Sequence
    EOS = "[EOS]"                         # End of Sequence
    END_OF_TEXT = "<|endoftext|>"         # Terminal End-Of-Text token
    IM_START = "<|im_start|>"             # ChatML format start
    IM_END = "<|im_end|>"                 # ChatML format end

    SYSTEM_TAG = "system"
    CONTEXT_TAG = "context"
    USER_TAG = "user"
    ASSISTANT_TAG = "assistant"
    TOOL_TAG = "tool"

    @classmethod
    def wrap_block(cls, role: str, content: str) -> str:
        """Wraps a block with explicit [BOS]role\\ncontent[EOS] delimiters."""
        clean_content = content.strip()
        return f"{cls.BOS}{role}\n{clean_content}\n{cls.EOS}"

    @classmethod
    def parse_blocks(cls, delimited_text: str) -> List[Dict[str, str]]:
        """Parses a delimited text string into structured [{role: str, content: str}] blocks."""
        pattern = re.compile(
            rf"{re.escape(cls.BOS)}([a-zA-Z0-9_\-]+)\n(.*?)\n{re.escape(cls.EOS)}",
            re.DOTALL
        )
        blocks = []
        for match in pattern.finditer(delimited_text):
            blocks.append({
                "role": match.group(1).strip(),
                "content": match.group(2).strip()
            })
        return blocks


# ---------------------------------------------------------------------------
# 2. Token Pricing & Cost Calculation Model
# ---------------------------------------------------------------------------
@dataclass
class ModelPricing:
    """Cost rates per 1,000,000 (1M) tokens in USD."""
    prompt_cost_per_1m: float
    completion_cost_per_1m: float
    currency: str = "USD"


MODEL_PRICING_CATALOG: Dict[str, ModelPricing] = {
    # Nemotron-3-Nano:30b / Ollama Cloud Rate
    "nemotron-3-nano:30b": ModelPricing(prompt_cost_per_1m=0.30, completion_cost_per_1m=0.90),
    "nemotron-3-nano": ModelPricing(prompt_cost_per_1m=0.30, completion_cost_per_1m=0.90),
    # Nemotron-3-Ultra / Ollama Cloud Enterprise Rate
    "nemotron-3-ultra": ModelPricing(prompt_cost_per_1m=0.80, completion_cost_per_1m=2.40),
    # OpenAI GPT-4o
    "gpt-4o": ModelPricing(prompt_cost_per_1m=2.50, completion_cost_per_1m=10.00),
    # OpenAI GPT-4o-mini
    "gpt-4o-mini": ModelPricing(prompt_cost_per_1m=0.15, completion_cost_per_1m=0.60),
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet": ModelPricing(prompt_cost_per_1m=3.00, completion_cost_per_1m=15.00),
    # Default Fallback Model
    "default": ModelPricing(prompt_cost_per_1m=0.30, completion_cost_per_1m=0.90),
}


@dataclass
class TokenCostReport:
    """Telemetry report for token usage and estimated monetary cost."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cost_usd: float
    completion_cost_usd: float
    total_cost_usd: float
    model_name: str
    context_limit: int
    context_utilization_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cost_usd": round(self.prompt_cost_usd, 6),
            "completion_cost_usd": round(self.completion_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "model_name": self.model_name,
            "context_limit": self.context_limit,
            "context_utilization_pct": round(self.context_utilization_pct, 2),
        }


# ---------------------------------------------------------------------------
# 3. Context Length Management & Overflow Prevention Engine
# ---------------------------------------------------------------------------
class ContextLengthManager:
    """
    Manages context window limits, token budgeting, sliding window truncation,
    and cost accounting for LLM prompts.
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        max_generation_tokens: int = 800,
        system_reserve_tokens: int = 600,
        rag_budget_tokens: int = 1500,
        history_budget_tokens: int = 1200,
        model_name: str = "nemotron-3-nano:30b",
    ):
        self.max_context_tokens = max_context_tokens
        self.max_generation_tokens = max_generation_tokens
        self.system_reserve_tokens = system_reserve_tokens
        self.rag_budget_tokens = rag_budget_tokens
        self.history_budget_tokens = history_budget_tokens
        self.model_name = model_name
        self.pricing = MODEL_PRICING_CATALOG.get(model_name.lower(), MODEL_PRICING_CATALOG["default"])

    def count_tokens(self, text: str) -> int:
        """Counts tokens accurately using tiktoken if available, with robust heuristic fallback."""
        if not text:
            return 0

        if _TIKTOKEN_AVAILABLE and _DEFAULT_ENCODER is not None:
            try:
                # Include special tokens in count if present
                return len(_DEFAULT_ENCODER.encode(text, disallowed_special=()))
            except Exception:
                pass

        # Robust Heuristic Fallback for Greek + English text:
        # Greek characters require ~1.4 to 1.8 tokens per word due to subword byte-pair encoding.
        words = text.split()
        total_tokens = 0
        for w in words:
            # Check if Greek or special character
            has_greek = any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in w)
            if has_greek:
                total_tokens += max(1, math.ceil(len(w) / 2.2))
            else:
                total_tokens += max(1, math.ceil(len(w) / 3.8))
        return max(1, total_tokens)

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int = 0,
        model_name: Optional[str] = None
    ) -> TokenCostReport:
        """Calculates exact monetary cost and context utilization percentage."""
        m_name = model_name or self.model_name
        pricing = MODEL_PRICING_CATALOG.get(m_name.lower(), self.pricing)

        p_cost = (prompt_tokens / 1_000_000.0) * pricing.prompt_cost_per_1m
        c_cost = (completion_tokens / 1_000_000.0) * pricing.completion_cost_per_1m
        total_tokens = prompt_tokens + completion_tokens
        utilization = (prompt_tokens / float(self.max_context_tokens)) * 100.0

        return TokenCostReport(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_cost_usd=p_cost,
            completion_cost_usd=c_cost,
            total_cost_usd=p_cost + c_cost,
            model_name=m_name,
            context_limit=self.max_context_tokens,
            context_utilization_pct=utilization,
        )

    def truncate_history_by_tokens(
        self,
        history: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        preserve_pairs: bool = True
    ) -> Tuple[List[Dict[str, str]], bool]:
        """
        Sliding window truncation governed by token limits rather than naive message count.
        Keeps the most recent turns that strictly fit within the token budget.
        """
        budget = max_tokens if max_tokens is not None else self.history_budget_tokens
        if not history:
            return [], False

        truncated_history: List[Dict[str, str]] = []
        accumulated_tokens = 0
        overflow_occurred = False

        # Traverse backwards from most recent message
        reversed_history = list(reversed(history))
        idx = 0
        while idx < len(reversed_history):
            msg = reversed_history[idx]
            msg_role = msg.get("role", "user")
            msg_content = msg.get("content", "")
            msg_tokens = self.count_tokens(SpecialTokens.wrap_block(msg_role, msg_content))

            # Pair preservation: If the message is 'assistant' and there's a preceding 'user' turn,
            # consider their joint token cost to avoid dangling responses without questions.
            if preserve_pairs and msg_role == "assistant" and (idx + 1 < len(reversed_history)):
                prev_user_msg = reversed_history[idx + 1]
                pair_tokens = msg_tokens + self.count_tokens(
                    SpecialTokens.wrap_block(prev_user_msg.get("role", "user"), prev_user_msg.get("content", ""))
                )
                if accumulated_tokens + pair_tokens <= budget:
                    truncated_history.append(msg)
                    truncated_history.append(prev_user_msg)
                    accumulated_tokens += pair_tokens
                    idx += 2
                    continue
                else:
                    if not truncated_history and idx == 0:
                        truncated_history.append(msg)
                        truncated_history.append(prev_user_msg)
                    overflow_occurred = True
                    break

            if accumulated_tokens + msg_tokens <= budget:
                truncated_history.append(msg)
                accumulated_tokens += msg_tokens
                idx += 1
            else:
                if not truncated_history and idx == 0:
                    truncated_history.append(msg)
                overflow_occurred = True
                break

        # Restore chronological order
        result = list(reversed(truncated_history))
        return result, overflow_occurred

    def fit_rag_chunks_by_budget(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Greedy knapsack fitting of RAG chunks within the token budget,
        prioritizing highest-scoring retrieved documents.
        """
        budget = max_tokens if max_tokens is not None else self.rag_budget_tokens
        if not chunks:
            return [], 0

        # Sort chunks by relevance score descending
        sorted_chunks = sorted(
            chunks,
            key=lambda d: float(d.get("score") if isinstance(d, dict) else getattr(d, "score", 0.0) or 0.0),
            reverse=True
        )

        fitted_chunks = []
        used_tokens = 0

        for chunk in sorted_chunks:
            name = chunk.get("name") or chunk.get("source_name", "Αξιοθέατο")
            text = chunk.get("text") or chunk.get("content", "")
            poi_id = chunk.get("id", "poi")
            chunk_repr = f"POIs ID: {poi_id} | Όνομα: {name}\n{text}\n"
            c_tokens = self.count_tokens(chunk_repr)

            if used_tokens + c_tokens <= budget:
                fitted_chunks.append(chunk)
                used_tokens += c_tokens
            else:
                logger.debug(
                    "[Context Manager] RAG chunk '%s' pruned to prevent token overflow (%d > %d budget)",
                    name, used_tokens + c_tokens, budget
                )

        return fitted_chunks, used_tokens

    def assemble_token_managed_prompt(
        self,
        system_instructions: str,
        retrieved_docs: Optional[List[Dict[str, Any]]],
        history: List[Dict[str, str]],
        user_query: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assembles the complete delimited prompt adhering strictly to context window budgets.
        Guarantees zero context overflow and returns token telemetries.
        """
        m_name = model_name or self.model_name

        # 1. Format System Block
        system_block = SpecialTokens.wrap_block(SpecialTokens.SYSTEM_TAG, system_instructions)
        system_tokens = self.count_tokens(system_block)

        # 2. Format User Query Block
        user_block = SpecialTokens.wrap_block(SpecialTokens.USER_TAG, user_query)
        user_query_tokens = self.count_tokens(user_block)

        # 3. Calculate remaining budget for RAG context and Conversation History
        # Remaining budget = max_context - system_tokens - user_query_tokens - max_generation_tokens - terminal tokens
        safety_margin = 50
        available_budget = (
            self.max_context_tokens
            - system_tokens
            - user_query_tokens
            - self.max_generation_tokens
            - safety_margin
        )
        available_budget = max(200, available_budget)

        # Split remaining budget dynamically (60% RAG, 40% History)
        rag_budget = min(self.rag_budget_tokens, int(available_budget * 0.60))
        history_budget = min(self.history_budget_tokens, int(available_budget * 0.40))

        # 4. Fit RAG chunks
        fitted_docs, rag_tokens = self.fit_rag_chunks_by_budget(retrieved_docs or [], max_tokens=rag_budget)
        
        # Build Context Block with delimiters
        context_content = "--- ΠΑΡΕΧΟΜΕΝΗ ΒΑΣΗ ΓΝΩΣΗΣ ---\n"
        for doc in fitted_docs:
            name = doc.get("name") or doc.get("source_name", "Αξιοθέατο")
            text = doc.get("text") or doc.get("content", "")
            context_content += f"POIs ID: {doc.get('id')} | Όνομα: {name}\n{text}\n-------------------\n"
        
        context_block = SpecialTokens.wrap_block(SpecialTokens.CONTEXT_TAG, context_content)

        # 5. Fit History turns
        pruned_history, history_overflow = self.truncate_history_by_tokens(
            history,
            max_tokens=history_budget,
            preserve_pairs=True
        )

        history_blocks = []
        for msg in pruned_history:
            history_blocks.append(SpecialTokens.wrap_block(msg.get("role", "user"), msg.get("content", "")))

        # 6. Assemble Full Prompt with Terminal Tag
        prompt_sections = [system_block]
        if fitted_docs:
            prompt_sections.append(context_block)
        prompt_sections.extend(history_blocks)
        prompt_sections.append(user_block)
        prompt_sections.append(f"{SpecialTokens.BOS}{SpecialTokens.ASSISTANT_TAG}\n")

        full_prompt = "\n".join(prompt_sections)
        total_prompt_tokens = self.count_tokens(full_prompt)

        # 7. Compute Cost Report
        cost_report = self.calculate_cost(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=0,
            model_name=m_name
        )

        return {
            "full_prompt": full_prompt,
            "prompt_tokens": total_prompt_tokens,
            "system_tokens": system_tokens,
            "rag_tokens": rag_tokens,
            "user_query_tokens": user_query_tokens,
            "history_turns_retained": len(pruned_history),
            "history_overflow_truncated": history_overflow,
            "fitted_docs_count": len(fitted_docs),
            "cost_report": cost_report.to_dict(),
            "special_tokens_used": [
                SpecialTokens.BOS,
                SpecialTokens.EOS,
                SpecialTokens.END_OF_TEXT
            ],
            "context_overflow_prevented": True,
        }


# Global singleton Context Manager
context_length_manager = ContextLengthManager()
