"""orchestrator/llm_client.py - Cloud LLM Client (NVIDIA Nemotron-3-Ultra via Ollama API).

Strictly enforces:
1. Cloud LLM calls using Ollama API format (with OpenAI-compatible fallback for Ollama Cloud /v1 proxies).
2. Permanent removal and disabling of offline rule-based DeterministicSynthesizer.
3. Graceful error raising (HTTPException 503 'Cloud LLM is currently unavailable.') when calls fail, timeout, or credentials are missing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
import requests
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class DeterministicSynthesizer:
    """
    Offline rule-based fallback generator.
    PERMANENTLY DISABLED: The system strictly mandates the Cloud LLM (NVIDIA Nemotron-3-Ultra via Ollama API).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud LLM is currently unavailable."
        )

    def synthesize(self, *args: Any, **kwargs: Any) -> str:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud LLM is currently unavailable."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud LLM is currently unavailable."
        )


class CloudOllamaClient:
    """
    Strict Cloud LLM Client connecting to NVIDIA Nemotron-3-Ultra via Ollama API.
    Does NOT allow offline fallbacks. Propagates errors as HTTPException(503).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 25.0,
    ) -> None:
        # 1. Read configuration from environment (.env)
        ollama_host_env = os.getenv("OLLAMA_HOST", "").strip()
        if ollama_host_env in ["0.0.0.0", "http://0.0.0.0", "https://0.0.0.0"]:
            ollama_host_env = ""
        self.host = (
            host
            or ollama_host_env
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://ollama.com"
        ).strip().rstrip("/")
        if self.host and not self.host.startswith("http://") and not self.host.startswith("https://"):
            self.host = f"https://{self.host}" if "ollama.com" in self.host else f"http://{self.host}"
        if self.host in ["http://0.0.0.0", "https://0.0.0.0"]:
            self.host = "https://ollama.com"

        self.api_key = (
            api_key
            or os.getenv("OLLAMA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()

        self.model = (
            model
            or os.getenv("OLLAMA_MODEL")
            or os.getenv("LLM_MODEL")
            or "nemotron-3-nano:30b"
        ).strip()

        timeout_env = os.getenv("LLM_TIMEOUT")
        self.timeout = float(timeout_env) if timeout_env else timeout

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        return headers

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """
        Calls the Cloud LLM using the Ollama API format.
        Strictly raises HTTPException(503) if the API call fails, times out, or key is missing.
        """
        # Strict enforcement: Missing API Key check
        if not self.api_key:
            logger.error("[CloudOllamaClient] Missing OLLAMA_API_KEY in environment.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cloud LLM is currently unavailable."
            )

        headers = self._get_headers()

        # Determine target endpoint based on host structure
        # Standard Ollama endpoint is /api/chat. If host explicitly ends with /v1, use /v1/chat/completions.
        if self.host.endswith("/v1"):
            target_url = f"{self.host}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        else:
            target_url = f"{self.host}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }

        try:
            logger.info("[CloudOllamaClient] Calling Cloud LLM (%s) at %s", self.model, target_url)
            response = requests.post(
                target_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            # If native /api/chat returned 404, retry via /v1/chat/completions (Ollama OpenAI compat layer)
            if response.status_code == 404 and not self.host.endswith("/v1"):
                alt_url = f"{self.host}/v1/chat/completions"
                logger.info("[CloudOllamaClient] Retrying via Ollama /v1 endpoint: %s", alt_url)
                alt_payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                response = requests.post(
                    alt_url,
                    headers=headers,
                    json=alt_payload,
                    timeout=self.timeout,
                )

            response.raise_for_status()
            data = response.json()

            # Parse content from either Ollama native (/api/chat) or OpenAI-compatible format
            content = ""
            if "message" in data and isinstance(data["message"], dict):
                msg = data["message"]
                content = msg.get("content", "").strip()
                if not content and "reasoning" in msg:
                    content = msg.get("reasoning", "").strip()
                if not content and "thinking" in msg:
                    content = msg.get("thinking", "").strip()
            elif "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                msg_dict = data["choices"][0].get("message", {})
                content = msg_dict.get("content", "").strip()
                if not content and "reasoning" in msg_dict:
                    content = msg_dict.get("reasoning", "").strip()
                if not content and "reasoning_content" in msg_dict:
                    content = msg_dict.get("reasoning_content", "").strip()
                if not content and "thinking" in msg_dict:
                    content = msg_dict.get("thinking", "").strip()
            else:
                raise ValueError(f"Unexpected response structure from LLM API: {data}")

            # Strip <think>...</think> tags if reasoning was included in output
            if "<think>" in content and "</think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            if not content:
                raise ValueError("Empty response received from Cloud LLM.")

            return content

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("[CloudOllamaClient] LLM execution failed (%s): %s", type(exc).__name__, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cloud LLM is currently unavailable."
            ) from exc

    def generate_rag_synthesis(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Strict Cloud LLM synthesis for factual Q&A."""
        from orchestrator.prompts import build_rag_context_prompt
        from orchestrator.system_prompts import DELIMITED_SYSTEM_PROMPT
        from orchestrator.context_manager import SpecialTokens

        user_prompt = build_rag_context_prompt(
            user_query=query,
            retrieved_docs=docs,
            use_special_tokens=True,
        )

        messages = [{"role": "system", "content": DELIMITED_SYSTEM_PROMPT}]
        if history:
            for h in history:
                messages.append({
                    "role": h.get("role", "user"),
                    "content": SpecialTokens.wrap_block(h.get("role", "user"), h.get("content", ""))
                })
        messages.append({"role": "user", "content": user_prompt})

        llm_reply = self.chat(messages=messages, temperature=0.3, max_tokens=700)
        citation_block = "\n".join(f"- [Πηγή: {d['name']} (ID: `{d['id']}`)]" for d in docs)
        return f"{llm_reply}\n\n**📚 Πηγές & Τεκμηρίωση:**\n{citation_block}"

    def generate_itinerary_synthesis(
        self,
        user_request: str,
        itinerary: Dict[str, Any],
        weather: Dict[str, Any],
        user_state: Any,
        is_fatigue_coffee: bool = False,
        museums_removed: bool = False,
    ) -> str:
        """Strict Cloud LLM narrative synthesis for validated itineraries."""
        from orchestrator.system_prompts import NATURAL_SYNTHESIS_PROMPT
        w_info = weather or {"condition": "Clear", "temperature_c": 22.0}

        user_content = (
            f"Αίτημα Χρήστη: '{user_request}'\n\n"
            f"Εγκεκριμένο Δρομολόγιο (JSON):\n{json.dumps(itinerary, ensure_ascii=False, indent=2)}\n\n"
            f"Καιρικές Συνθήκες: {w_info.get('condition', 'Clear')}, {w_info.get('temperature_c', 22)}°C\n"
            f"Ειδικές Παράμετροι: Παιδιά={getattr(user_state, 'traveling_with_kids', False)}, "
            f"ΑμεΑ={getattr(user_state, 'wheelchair_accessible', False)}, "
            f"Καφές/Ξεκούραση={is_fatigue_coffee}, "
            f"Αφαίρεση Μουσείων={museums_removed}"
        )

        messages = [
            {"role": "system", "content": NATURAL_SYNTHESIS_PROMPT},
            {"role": "user", "content": user_content}
        ]

        return self.chat(messages=messages, temperature=0.4, max_tokens=1000)

    def generate_chat_synthesis(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        intent: str = "general",
    ) -> str:
        """Strict Cloud LLM synthesis for conversational messages."""
        from orchestrator.system_prompts import ATHENS_EXPERT_SYSTEM_PROMPT

        messages = [{"role": "system", "content": ATHENS_EXPERT_SYSTEM_PROMPT}]
        if history:
            for h in history:
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        messages.append({"role": "user", "content": user_message})

        return self.chat(messages=messages, temperature=0.5, max_tokens=600)


# Global singleton helper
_cloud_ollama_client: Optional[CloudOllamaClient] = None


def get_cloud_ollama_client() -> CloudOllamaClient:
    global _cloud_ollama_client
    if _cloud_ollama_client is None:
        _cloud_ollama_client = CloudOllamaClient()
    return _cloud_ollama_client
