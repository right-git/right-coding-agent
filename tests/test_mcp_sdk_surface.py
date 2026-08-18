import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestMcpSdkSurface(unittest.TestCase):
    """Every SDK name the mcp layer builds on, in one place.

    If an mcp upgrade moves or renames one of these, this test points at the
    exact break before any runtime code misbehaves.
    """

    def test_client_entry_points_exist(self):
        from mcp import ClientSession, StdioServerParameters  # noqa: F401
        from mcp.client.stdio import stdio_client, get_default_environment  # noqa: F401
        from mcp.client.streamable_http import streamable_http_client  # noqa: F401
        from mcp.client.sse import sse_client  # noqa: F401

    def test_session_methods_exist(self):
        from mcp import ClientSession

        for method in (
            "initialize",
            "list_tools",
            "call_tool",
            "list_prompts",
            "get_prompt",
            "list_resources",
            "read_resource",
        ):
            self.assertTrue(callable(getattr(ClientSession, method)), method)

    def test_oauth_names_exist(self):
        from mcp.client.auth import OAuthClientProvider, TokenStorage  # noqa: F401
        from mcp.shared.auth import (  # noqa: F401
            OAuthClientInformationFull,
            OAuthClientMetadata,
            OAuthToken,
        )

    def test_model_field_names_the_adapter_reads(self):
        # SDK 2.0 exposes snake_case attributes for the wire format's
        # camelCase fields; utils.read_field retries the snake spelling on a
        # camelCase miss. If an upgrade renames these again, fail HERE with
        # the exact field, not in production with silently-empty schemas
        # (that shipped once: every real tool registered with no arguments).
        from mcp import types

        expectations = {
            "Tool": {"name", "description", "input_schema", "annotations"},
            "CallToolResult": {"content", "structured_content", "is_error"},
            "ImageContent": {"type", "data", "mime_type"},
            "ResourceLink": {"name", "uri", "description", "mime_type"},
            "ToolAnnotations": {"read_only_hint", "destructive_hint"},
            "Prompt": {"name", "description", "arguments"},
            "PromptArgument": {"name", "required"},
            "Resource": {"name", "uri", "description", "mime_type"},
            "BlobResourceContents": {"uri", "mime_type", "blob"},
            "TextResourceContents": {"uri", "mime_type", "text"},
        }
        for type_name, fields in expectations.items():
            model = getattr(types, type_name)
            missing = fields - set(model.model_fields)
            self.assertFalse(missing, f"{type_name} lost fields: {missing}")

    def test_private_http_client_factory_exists(self):
        # transports.py depends on this private path (mcp.shared._httpx_utils
        # is not part of the SDK's public surface); an SDK upgrade that moves
        # or renames it must fail HERE, not at REPL startup when a server
        # first tries to connect over http/sse.
        from mcp.shared._httpx_utils import create_mcp_http_client  # noqa: F401


if __name__ == "__main__":
    unittest.main()
