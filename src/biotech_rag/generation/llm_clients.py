from functools import lru_cache
import logging
from typing import Any, Dict, List, Optional
import os

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Fix for AttributeError: module 'langchain' has no attribute 'verbose'/'debug'
# This occurs in some environments where langchain-core and langchain versions mismatch.
try:
    import langchain

    if not hasattr(langchain, "verbose"):
        langchain.verbose = False
    if not hasattr(langchain, "debug"):
        langchain.debug = False
    if not hasattr(langchain, "llm_cache"):
        langchain.llm_cache = None
except ImportError:
    pass

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from biotech_rag.indexing.openrouter_client import chat_completion_request

try:
    from biotech_rag.config import settings
except Exception:
    settings = None  # type: ignore
from biotech_rag.utils.cache import get_cached_response, set_cached_response

LOGGER = logging.getLogger(__name__)


class OpenRouterChat(BaseChatModel):
    """Custom LangChain Chat Model for OpenRouter to ensure reliability and caching."""

    model_name: str = "deepseek/deepseek-r1-distill-llama-70b:floor"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    use_cache: bool = True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Convert LangChain messages to OpenRouter format
        formatted_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                role = "user"
            formatted_messages.append({"role": role, "content": m.content})

        # Check Cache
        cache_key = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temp": self.temperature,
        }
        if self.use_cache:
            cached = get_cached_response(cache_key, self.model_name)
            if cached:
                LOGGER.info(f"Using cached response for {self.model_name}")
                return self._create_chat_result(cached)

        # Direct Request (Same as embedding method)
        response_data = chat_completion_request(
            messages=formatted_messages,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Track usage
        track_llm_usage(response_data)

        try:
            content = response_data["choices"][0]["message"]["content"]
            if self.use_cache:
                set_cached_response(cache_key, self.model_name, content)
            return self._create_chat_result(content)
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected OpenRouter response format: {response_data}") from e

    def _create_chat_result(self, text: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"


def get_openrouter_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> OpenRouterChat:
    """Get the custom OpenRouter LLM client."""
    if model is None:
        if settings is not None:
            model = settings.llm_model
        else:
            model = os.getenv("LLM_MODEL", "deepseek/deepseek-r1-distill-llama-70b:floor")

    if temperature is None:
        if settings is not None:
            temperature = settings.llm_temperature
        else:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    if max_tokens is None:
        if settings is not None:
            max_tokens = settings.llm_max_tokens
        else:
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4000"))

    return OpenRouterChat(model_name=model, temperature=temperature, max_tokens=max_tokens)


def query_llm(prompt: str, model: Optional[str] = None) -> str:
    """Simplified string-in/string-out query."""
    llm = get_openrouter_llm(model=model)
    return llm.invoke(prompt).content


@lru_cache(maxsize=1000)
def cached_query_llm(
    prompt: str, model: str = "deepseek/deepseek-r1-distill-llama-70b:floor"
) -> str:
    """Cached version of query_llm for repeated prompts."""
    return query_llm(prompt, model)


def batch_llm_calls(prompts: List[str], llm=None, batch_size: int = 5) -> List[str]:
    """
    Batch LLM calls to reduce costs and latency.

    Args:
        prompts: List of prompts.
        llm: LLM instance (defaults to get_openrouter_llm).
        batch_size: Process in batches of this size.

    Returns:
        List of responses.
    """
    if llm is None:
        llm = get_openrouter_llm()
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batch_results = [llm.invoke(p).content for p in batch]
        results.extend(batch_results)
        LOGGER.info(
            f"Processed batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}"
        )
    return results


def track_llm_usage(response_data: Dict) -> Dict:
    """
    Extract and log usage from OpenRouter response.

    Args:
        response_data: Full response from chat_completion_request.

    Returns:
        Usage dict (tokens, cost estimate if available).
    """
    usage = response_data.get("usage", {})
    if usage:
        LOGGER.info(f"LLM Usage: {usage}")
    return usage
