import unittest
from io import StringIO
from unittest.mock import AsyncMock, Mock

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import SummarizationMiddleware

from src.llm.middlewares.message_log import MessageLogMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import HumanMessage
from rich.console import Console

from src.config.prompts import Prompts
from src.llm.agents import Agents
from src.llm.client import LLMClient
from src.llm.types import LLMProvider
from src.main import available_models, trim_incomplete_tool_calls
from src.ui import ChatUI


class AgentFilesystemConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_right_coding_agent_has_no_filesystem_middleware(self):
        agent = Agents(
            [
                LLMProvider(
                    provider_name="openai",
                    api_key="test-key",
                    api_base="http://localhost",
                )
            ]
        )
        agent.ask_agent = AsyncMock(return_value={"messages": []})

        await agent.right_coding_agent(
            messages=[HumanMessage("create test.txt")],
            model="openai/gpt-4.1-mini",
            thread_id="thread-1",
        )

        middlewares = agent.ask_agent.await_args.kwargs["middlewares"]
        self.assertFalse(any(isinstance(middleware, FilesystemMiddleware) for middleware in middlewares))


class ToolContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_right_coding_agent_exposes_only_meta_tools(self):
        agent = Agents(
            [
                LLMProvider(
                    provider_name="openai",
                    api_key="test-key",
                    api_base="http://localhost",
                )
            ]
        )
        agent.ask_agent = AsyncMock(return_value={"messages": []})

        await agent.right_coding_agent(
            messages=[HumanMessage("list files")],
            model="openai/gpt-4.1-mini",
            thread_id="thread-2",
        )

        tool_names = [
            getattr(tool, "name", getattr(tool, "__name__", str(tool)))
            for tool in agent.ask_agent.await_args.kwargs["tools"]
        ]

        self.assertEqual(
            tool_names,
            [
                "search_tools",
                "get_tool",
                "run_tools",
            ],
        )

    async def test_right_coding_agent_uses_required_middlewares(self):
        agent = Agents(
            [
                LLMProvider(
                    provider_name="openai",
                    api_key="test-key",
                    api_base="http://localhost",
                )
            ]
        )
        agent.ask_agent = AsyncMock(return_value={"messages": []})

        await agent.right_coding_agent(
            messages=[HumanMessage("delete AGENTS.md using shell")],
            model="openai/gpt-4.1-mini",
        )

        middlewares = agent.ask_agent.await_args.kwargs["middlewares"]

        self.assertTrue(any(isinstance(middleware, SummarizationMiddleware) for middleware in middlewares))
        self.assertIsInstance(middlewares[-1], MessageLogMiddleware)

    def test_system_prompt_does_not_list_tools(self):
        self.assertNotIn("## Tools", Prompts.right_coding_agent_sys)
        self.assertNotIn("**read_file**", Prompts.right_coding_agent_sys)
        self.assertNotIn("**write_file**", Prompts.right_coding_agent_sys)
        self.assertNotIn("**edit_file**", Prompts.right_coding_agent_sys)
        self.assertNotIn("**execute**", Prompts.right_coding_agent_sys)


class ConversationStateTests(unittest.TestCase):
    def test_trim_incomplete_tool_calls_drops_broken_suffix(self):
        messages = [
            HumanMessage("delete the file"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": "rm AGENTS.md"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        trimmed = trim_incomplete_tool_calls(messages)

        self.assertEqual(trimmed, [messages[0]])

    def test_trim_incomplete_tool_calls_keeps_completed_tool_turn(self):
        messages = [
            HumanMessage("list files"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ls",
                        "args": {"path": "/tmp"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="['/tmp/a']", tool_call_id="call_1", name="ls"),
            AIMessage(content="done"),
        ]

        self.assertEqual(trim_incomplete_tool_calls(messages), messages)


class ErrorHandlingTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_provider_raises_root_cause(self):
        client = LLMClient(
            [
                LLMProvider(
                    provider_name="openai",
                    api_key="test-key",
                    api_base="http://localhost",
                )
            ],
            num_retries=1,
        )
        client.build_chat_model = Mock(return_value=object())  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with unittest.mock.patch("src.llm.client.create_agent") as create_agent_mock:
                agent = AsyncMock()
                agent.ainvoke.side_effect = RuntimeError("boom")
                create_agent_mock.return_value = agent
                await client.ask_agent(
                    system_prompt="test",
                    agent_input={"messages": [HumanMessage("hi")]},
                    model_name="openai/gpt-4.1-mini",
                    tools=[],
                    middlewares=[],
                )


class ModelSelectionTests(unittest.TestCase):
    def test_available_models_include_default_model(self):
        self.assertIn("google/gemini-3.7-flash", available_models)


class ChatUIRenderingTests(unittest.TestCase):
    def test_print_response_hides_reasoning_payload_and_renders_text_block(self):
        ui = ChatUI(model="openai/gpt-5.3-codex-mini")
        ui.console = Console(
            file=StringIO(),
            record=True,
            force_terminal=False,
            width=120,
        )

        ui.print_response(
            [
                AIMessage(
                    content=[
                        {
                            "id": "rs_tmp_ml51l6tbns",
                            "summary": [],
                            "type": "reasoning",
                            "encrypted_content": "secret",
                            "status": "completed",
                            "format": "openai-responses-v1",
                        },
                        {
                            "type": "text",
                            "text": "README.md populated with project overview.",
                            "annotations": [],
                            "id": "msg_tmp_tejci35y9k",
                        },
                    ]
                )
            ]
        )

        rendered = ui.console.export_text()

        self.assertIn("README.md populated with project overview.", rendered)
        self.assertNotIn("encrypted_content", rendered)


if __name__ == "__main__":
    unittest.main()
