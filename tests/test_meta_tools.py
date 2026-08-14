import json
import sys
import unittest
from pathlib import Path

from langchain_core.tools import tool


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.meta_tools import (
    MAX_RESULT_CHARS,
    META_TOOLS,
    ToolRegistry,
    get_registry,
    get_tool,
    run_tools,
    search_tools,
    set_registry,
)


@tool(parse_docstring=True)
async def fetch_page(url: str, timeout: int = 5) -> str:
    """Fetch a web page as text.

    Args:
        url: Address of the page.
        timeout: Seconds to wait.

    Returns:
        Page body.
    """
    return f"page:{url}:{timeout}"


@tool(parse_docstring=True)
async def boom() -> str:
    """Always fails, for error-path tests.

    Returns:
        Never returns.
    """
    raise RuntimeError("kaput")


def make_job_status():
    """A stateful tool: 'running' twice, then 'success'."""
    state = {"checks": 0}

    @tool(parse_docstring=True)
    async def job_status(job_id: str) -> str:
        """Status of a background job.

        Args:
            job_id: The job identifier.

        Returns:
            The status string.
        """
        state["checks"] += 1
        return "success" if state["checks"] >= 3 else "running"

    return job_status, state


class ToolRegistryTests(unittest.TestCase):
    def test_duplicate_names_are_rejected(self):
        registry = ToolRegistry([fetch_page])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            registry.register(fetch_page)

    def test_names_shadowed_by_interpreter_builtins_are_rejected(self):
        @tool(parse_docstring=True)
        async def sleep(seconds: int) -> str:
            """Would shadow the interpreter builtin.

            Args:
                seconds: How long.

            Returns:
                Nothing useful.
            """
            return "no"

        with self.assertRaisesRegex(ValueError, "collides"):
            ToolRegistry([sleep])

    def test_search_ranks_name_matches_first(self):
        job_status, _ = make_job_status()
        registry = ToolRegistry([job_status, fetch_page])

        matches = registry.search("fetch page")

        self.assertEqual(matches[0].name, "fetch_page")

    def test_search_with_empty_query_lists_everything(self):
        job_status, _ = make_job_status()
        registry = ToolRegistry([job_status, fetch_page])

        self.assertEqual(len(registry.search("")), 2)

    def test_search_with_no_hits_returns_nothing(self):
        registry = ToolRegistry([fetch_page])

        self.assertEqual(registry.search("zzz-nothing"), [])

    def test_brief_shows_signature_with_defaults_and_summary(self):
        registry = ToolRegistry([fetch_page])

        self.assertEqual(
            registry.brief(fetch_page),
            "fetch_page(url, timeout=5) — Fetch a web page as text.",
        )

    def test_document_contains_contract_and_call_example(self):
        registry = ToolRegistry([fetch_page])

        documentation = registry.document("fetch_page")

        self.assertIn("fetch_page(url, timeout=5)", documentation)
        self.assertIn("Fetch a web page as text.", documentation)
        self.assertIn('"timeout"', documentation)
        self.assertIn("fetch_page(url=..., timeout=...)", documentation)

    def test_document_of_unknown_tool_is_none(self):
        self.assertIsNone(ToolRegistry([fetch_page]).document("nope"))


class ScriptCallableTests(unittest.IsolatedAsyncioTestCase):
    async def test_positional_and_keyword_arguments_are_mapped(self):
        call = ToolRegistry([fetch_page]).callables()["fetch_page"]

        self.assertEqual(await call("a"), "page:a:5")
        self.assertEqual(await call("a", 9), "page:a:9")
        self.assertEqual(await call("a", timeout=7), "page:a:7")
        self.assertEqual(await call(url="b", timeout=1), "page:b:1")

    async def test_too_many_positional_arguments_fail(self):
        call = ToolRegistry([fetch_page]).callables()["fetch_page"]

        with self.assertRaisesRegex(TypeError, "positional"):
            await call("a", 9, "extra")

    async def test_duplicate_argument_values_fail(self):
        call = ToolRegistry([fetch_page]).callables()["fetch_page"]

        with self.assertRaisesRegex(TypeError, "multiple values"):
            await call("a", url="b")


class MetaToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(set_registry, None)
        self.job_status, self.job_state = make_job_status()
        set_registry(ToolRegistry([fetch_page, self.job_status, boom]))

    async def test_search_tools_reports_matches(self):
        result = await search_tools.ainvoke({"query": "fetch page"})

        self.assertIn("fetch_page(url, timeout=5)", result)
        self.assertIn("get_tool", result)

    async def test_search_tools_falls_back_to_the_catalogue(self):
        result = await search_tools.ainvoke({"query": "zzz-nothing"})

        self.assertIn("Nothing matched", result)
        self.assertIn("fetch_page", result)
        self.assertIn("job_status", result)

    async def test_get_tool_returns_the_contract(self):
        result = await get_tool.ainvoke({"name": "fetch_page"})

        self.assertIn("Argument schema:", result)
        self.assertIn("Address of the page.", result)

    async def test_get_tool_suggests_names_for_unknown_tools(self):
        result = await get_tool.ainvoke({"name": "fetch"})

        self.assertIn("Unknown tool: 'fetch'", result)
        self.assertIn("fetch_page", result)

    async def test_run_tools_executes_parallel_calls_and_logs(self):
        outcome = json.loads(
            await run_tools.ainvoke(
                {
                    "code": (
                        'pages = parallel(fetch_page("a"), fetch_page("b", 9))\n'
                        "print(len(pages))\n"
                        "return [page.upper() for page in pages]\n"
                    )
                }
            )
        )

        self.assertIsNone(outcome["error"])
        self.assertEqual(outcome["result"], ["PAGE:A:5", "PAGE:B:9"])
        self.assertEqual(outcome["logs"], ["2"])

    async def test_run_tools_polls_with_sleep_until_done(self):
        outcome = json.loads(
            await run_tools.ainvoke(
                {
                    "code": (
                        'status = job_status("j1")\n'
                        'while status == "running":\n'
                        "    sleep(0)\n"
                        '    status = job_status("j1")\n'
                        "return status\n"
                    )
                }
            )
        )

        self.assertIsNone(outcome["error"])
        self.assertEqual(outcome["result"], "success")
        self.assertEqual(self.job_state["checks"], 3)

    async def test_run_tools_lets_scripts_catch_tool_failures(self):
        outcome = json.loads(
            await run_tools.ainvoke(
                {
                    "code": (
                        "try:\n"
                        "    boom()\n"
                        "except Exception as error:\n"
                        '    return "caught: " + str(error)\n'
                    )
                }
            )
        )

        self.assertIsNone(outcome["error"])
        self.assertIn("caught:", outcome["result"])
        self.assertIn("kaput", outcome["result"])

    async def test_run_tools_reports_syntax_errors(self):
        outcome = json.loads(await run_tools.ainvoke({"code": "return ((("}))

        self.assertIsNone(outcome["result"])
        self.assertIn("SyntaxError", outcome["error"])

    async def test_run_tools_blocks_imports(self):
        outcome = json.loads(
            await run_tools.ainvoke({"code": "import os\nreturn os"})
        )

        self.assertIn("PolicyError", outcome["error"])

    async def test_run_tools_truncates_oversized_output(self):
        @tool(parse_docstring=True)
        async def big() -> str:
            """A tool with a huge result.

            Returns:
                A very long string.
            """
            return "x" * (MAX_RESULT_CHARS + 20_000)

        set_registry(ToolRegistry([big]))

        result = await run_tools.ainvoke({"code": "return big()"})

        self.assertIn("truncated", result)
        self.assertLess(len(result), MAX_RESULT_CHARS + 200)


class DefaultRegistryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_registry, None)
        set_registry(None)

    def test_default_registry_holds_web_and_screen_tools(self):
        names = [tool_obj.name for tool_obj in get_registry().all_tools()]

        self.assertEqual(
            names,
            [
                "web_search",
                "screen_locate",
                "screen_screenshot",
                "screen_mark",
                "screen_click",
                "screen_type",
                "screen_key",
                "screen_scroll",
            ],
        )

    def test_meta_tools_have_stable_names(self):
        self.assertEqual(
            [tool_obj.name for tool_obj in META_TOOLS],
            ["search_tools", "get_tool", "run_tools"],
        )


if __name__ == "__main__":
    unittest.main()
