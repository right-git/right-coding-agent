import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from src.config.logging import logger
from .types import AgentTool, LLMProvider, T


class LLMClient:
    def __init__(
        self,
        providers: Sequence[LLMProvider],
        num_retries: int = 3,
        timeout: int = 30,
        cooldown_time: int = 5,
    ):
        self.providers = list(providers)
        self.num_retries = num_retries
        self.timeout = timeout
        self.cooldown_time = cooldown_time
        self.checkpointer = MemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
        self._agent_cache: dict[str, Any] = {}

    def fix_base(self, url: str | None) -> str | None:
        if not url:
            return None
        return url[:-1] if url.endswith("/") else url

    def get_default_provider(self) -> LLMProvider:
        if not self.providers:
            raise RuntimeError("No LLM providers configured")
        return self.providers[0]

    def resolve_model_name(
        self,
        provider: LLMProvider,
        model_name: str | None = None,
    ) -> str:
        resolved_model_name = model_name or provider.model_name
        if not resolved_model_name:
            raise ValueError(f"Model name is not configured for provider {provider.provider_name}")
        return resolved_model_name

    def build_client_kwargs(
        self,
        provider: LLMProvider,
        model_name: str,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        client_kwargs: dict[str, Any] = {
            "model": model_name,
            "model_provider": provider.provider_name,
            "api_key": provider.api_key,
            "timeout": self.timeout,
        }

        if provider.additional_headers:
            client_kwargs["default_headers"] = provider.additional_headers

        if temperature is not None:
            client_kwargs["temperature"] = temperature

        if seed is not None:
            client_kwargs["seed"] = seed

        if reasoning_effort is not None:
            client_kwargs["reasoning_effort"] = reasoning_effort

        if provider.provider_name in ("openai", "azure_openai"):
            # Without this, streamed turns report no usage_metadata and the
            # usage footer goes blank.
            client_kwargs["stream_usage"] = True

        if provider.provider_name == "azure_openai":
            if not provider.api_base:
                raise ValueError("api_base is required for azure_openai provider")

            client_kwargs["azure_endpoint"] = self.fix_base(provider.api_base)
            client_kwargs["azure_deployment"] = provider.deployment_name or model_name

            if provider.api_version:
                client_kwargs["api_version"] = provider.api_version
        elif provider.api_base:
            client_kwargs["base_url"] = self.fix_base(provider.api_base)

        return client_kwargs

    def build_chat_model(
        self,
        model_name: str | None = None,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
    ) -> BaseChatModel:
        selected_provider = provider or self.get_default_provider()
        resolved_model_name = self.resolve_model_name(selected_provider, model_name)
        return init_chat_model(
            **self.build_client_kwargs(
                provider=selected_provider,
                model_name=resolved_model_name,
                temperature=temperature,
                seed=seed,
                reasoning_effort=reasoning_effort,
            )
        )

    async def ask_agent(
        self,
        system_prompt: str,
        agent_input: dict[str, Any],
        model_name: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        tools: Sequence[AgentTool] | None = None,
        response_format: type[T] | None = None,
        middlewares: Sequence[AgentMiddleware] | None = None,
        context_schema: type[Any] | None = None,
        context: Any | None = None,
        thread_id: str | None = None,
        on_message: Callable[[Any], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if not self.providers:
            raise RuntimeError("No LLM providers configured")

        last_error: Exception | None = None
        resolved_tools = list(tools or [])
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None

        for provider in self.providers:
            resolved_model_name = self.resolve_model_name(provider, model_name)

            for attempt in range(self.num_retries):
                stage = "build_model"
                try:
                    if attempt > 0:
                        logger.warning(
                            "Retrying provider [{}] attempt [{}/{}]",
                            provider.provider_name,
                            attempt + 1,
                            self.num_retries,
                        )

                    logger.info(
                        "Starting agent call provider [{}] model [{}] endpoint [{}] "
                        "attempt [{}/{}] message_count [{}] tools [{}] "
                        "response_format [{}] context_schema [{}]",
                        provider.provider_name,
                        resolved_model_name,
                        provider.api_base,
                        attempt + 1,
                        self.num_retries,
                        len(agent_input.get("messages", [])),
                        [getattr(tool, "name", getattr(tool, "__name__", str(tool))) for tool in resolved_tools],
                        getattr(response_format, "__name__", None),
                        getattr(context_schema, "__name__", None),
                    )

                    stage = "create_agent"
                    agent = create_agent(
                        model=self.build_chat_model(
                            model_name=resolved_model_name,
                            provider=provider,
                            temperature=temperature,
                            seed=seed,
                            reasoning_effort=reasoning_effort,
                        ),
                        tools=resolved_tools,
                        response_format=response_format,
                        system_prompt=system_prompt,
                        middleware=middlewares,
                        context_schema=context_schema,
                        checkpointer=self.checkpointer if thread_id else None,
                    )

                    stage = "invoke_agent"
                    if on_message is None and on_token is None:
                        response = await agent.ainvoke(input=agent_input, context=context, config=config)
                    else:
                        response = await self._stream_agent(agent, agent_input, context, config, on_message, on_token)

                    if thread_id:
                        self._agent_cache[thread_id] = agent
                        state = agent.get_state(config)
                        if state.next:
                            response["__interrupted__"] = True

                    logger.info(
                        "Agent call succeeded provider [{}] model [{}] attempt [{}/{}]",
                        provider.provider_name,
                        resolved_model_name,
                        attempt + 1,
                        self.num_retries,
                    )
                    return response
                except Exception as exc:
                    last_error = exc
                    logger.exception(
                        "Failed provider [{}] model [{}] endpoint [{}] "
                        "attempt [{}/{}] at stage [{}] message_count [{}] "
                        "tools [{}] response_format [{}] context_schema [{}]. "
                        "Error: {}",
                        provider.provider_name,
                        resolved_model_name,
                        provider.api_base,
                        attempt + 1,
                        self.num_retries,
                        stage,
                        len(agent_input.get("messages", [])),
                        [getattr(tool, "name", getattr(tool, "__name__", str(tool))) for tool in resolved_tools],
                        getattr(response_format, "__name__", None),
                        getattr(context_schema, "__name__", None),
                        exc,
                    )
                    if attempt < self.num_retries - 1:
                        await asyncio.sleep(self.cooldown_time)

        if last_error is not None and len(self.providers) == 1:
            raise last_error

        raise RuntimeError("All configured LLM providers failed") from last_error

    @staticmethod
    def _forward(callback: Callable[[Any], None], payload: Any) -> None:
        """Deliver one stream event; a display failure must not kill the turn."""
        try:
            callback(payload)
        except Exception:
            logger.exception("Stream callback failed")

    async def _stream_agent(
        self,
        agent: Any,
        agent_input: dict[str, Any],
        context: Any | None,
        config: dict[str, Any] | None,
        on_message: Callable[[Any], None] | None,
        on_token: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        """Run the agent while forwarding events as they happen.

        `updates` chunks carry every newly produced message (tool calls and
        tool results included) the moment a node finishes; `messages` chunks
        carry the model's own tokens; the last `values` chunk is the same
        final state `ainvoke` would have returned.
        """
        modes = ["updates", "values"] + (["messages"] if on_token else [])
        response: dict[str, Any] | None = None

        async for mode, chunk in agent.astream(input=agent_input, context=context, config=config, stream_mode=modes):
            if mode == "values":
                response = chunk
            elif mode == "messages":
                piece, metadata = chunk
                if (metadata or {}).get("langgraph_node") != "model":
                    continue  # e.g. the summarization middleware's own model call
                if not isinstance(piece, AIMessageChunk):
                    continue
                # .text is a property; do NOT call it — the compat shim it
                # returns is callable and calling it warns on every token.
                text = str(piece.text)
                if text:
                    self._forward(on_token, text)
            elif on_message is not None:
                for node_update in (chunk or {}).values():
                    if not isinstance(node_update, dict):
                        continue
                    for message in node_update.get("messages") or []:
                        if isinstance(message, (AIMessage, ToolMessage)):
                            self._forward(on_message, message)

        if response is None:
            raise RuntimeError("Agent stream ended without a final state")
        return response

    async def resume_agent(
        self,
        resume_value: Any,
        thread_id: str,
        context: Any | None = None,
    ) -> dict[str, Any]:
        agent = self._agent_cache.get(thread_id)
        if not agent:
            raise RuntimeError("No agent found for this thread")

        config = {"configurable": {"thread_id": thread_id}}
        response = await agent.ainvoke(Command(resume=resume_value), config=config)

        state = agent.get_state(config)
        if state.next:
            response["__interrupted__"] = True

        return response
