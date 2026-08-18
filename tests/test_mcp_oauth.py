import asyncio
import contextlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp.shared.auth import AuthorizationCodeResult, OAuthClientInformationFull, OAuthToken

from src.llm.tools.mcp import oauth as oauth_module
from src.llm.tools.mcp.config import McpServerConfig
from src.llm.tools.mcp.oauth import (
    CallbackServer,
    FileTokenStorage,
    NeedsInteractiveAuth,
    build_oauth_provider,
    clear_tokens,
    default_token_path,
    has_stored_tokens,
)


class TestDefaultTokenPath(unittest.TestCase):
    def test_lives_under_the_agent_home_directory(self):
        with tempfile.TemporaryDirectory() as home:
            path = default_token_path(Path(home))
        self.assertEqual(path.name, "mcp-tokens.json")
        self.assertEqual(path.parent.name, ".right-agent")


class TestFileTokenStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "tokens.json"

    def storage(self, name="srv", url="https://s/mcp"):
        return FileTokenStorage(name, url, path=self.path)

    def test_round_trip_tokens(self):
        storage = self.storage()
        token = OAuthToken(access_token="abc", token_type="Bearer", refresh_token="r1")
        asyncio.run(storage.set_tokens(token))
        loaded = asyncio.run(storage.get_tokens())
        self.assertEqual(loaded.access_token, "abc")
        self.assertEqual(loaded.refresh_token, "r1")

    def test_missing_returns_none(self):
        self.assertIsNone(asyncio.run(self.storage().get_tokens()))
        self.assertIsNone(asyncio.run(self.storage().get_client_info()))

    def test_servers_are_isolated(self):
        asyncio.run(self.storage("a", "https://a/").set_tokens(OAuthToken(access_token="ta", token_type="Bearer")))
        self.assertIsNone(asyncio.run(self.storage("b", "https://b/").get_tokens()))

    def test_same_name_different_url_is_a_different_entry(self):
        asyncio.run(self.storage("a", "https://one/").set_tokens(OAuthToken(access_token="t1", token_type="Bearer")))
        self.assertIsNone(asyncio.run(self.storage("a", "https://two/").get_tokens()))

    def test_client_info_round_trip(self):
        info = OAuthClientInformationFull(client_id="cid", redirect_uris=["http://127.0.0.1:43110/callback"])
        storage = self.storage()
        asyncio.run(storage.set_client_info(info))
        self.assertEqual(asyncio.run(storage.get_client_info()).client_id, "cid")

    def test_tokens_and_client_info_share_one_entry(self):
        storage = self.storage()
        asyncio.run(storage.set_tokens(OAuthToken(access_token="t", token_type="Bearer")))
        asyncio.run(storage.set_client_info(OAuthClientInformationFull(client_id="cid", redirect_uris=[])))
        self.assertEqual(asyncio.run(storage.get_tokens()).access_token, "t")
        self.assertEqual(asyncio.run(storage.get_client_info()).client_id, "cid")

    def test_writes_create_the_parent_directory(self):
        nested = Path(self.tmp.name) / "deep" / "nest" / "tokens.json"
        storage = FileTokenStorage("srv", "https://s/mcp", path=nested)
        asyncio.run(storage.set_tokens(OAuthToken(access_token="t", token_type="Bearer")))
        self.assertTrue(nested.exists())

    def test_corrupt_file_reads_as_empty(self):
        self.path.write_text("{not json at all", encoding="utf-8")
        self.assertIsNone(asyncio.run(self.storage().get_tokens()))
        self.assertFalse(has_stored_tokens("srv", "https://s/mcp", path=self.path))

    def test_corrupt_file_is_replaced_on_write(self):
        self.path.write_text("{not json at all", encoding="utf-8")
        asyncio.run(self.storage().set_tokens(OAuthToken(access_token="fresh", token_type="Bearer")))
        self.assertEqual(asyncio.run(self.storage().get_tokens()).access_token, "fresh")

    def test_unreadable_entry_shape_reads_as_none(self):
        self.path.write_text(json.dumps({"srv|https://s/mcp": {"tokens": "nonsense"}}), encoding="utf-8")
        self.assertIsNone(asyncio.run(self.storage().get_tokens()))

    def test_other_servers_survive_a_write(self):
        asyncio.run(self.storage("a", "https://a/").set_tokens(OAuthToken(access_token="ta", token_type="Bearer")))
        asyncio.run(self.storage("b", "https://b/").set_tokens(OAuthToken(access_token="tb", token_type="Bearer")))
        self.assertEqual(asyncio.run(self.storage("a", "https://a/").get_tokens()).access_token, "ta")
        self.assertEqual(asyncio.run(self.storage("b", "https://b/").get_tokens()).access_token, "tb")

    @unittest.skipIf(sys.platform == "win32", "POSIX file modes only")
    def test_file_permissions_are_owner_only(self):
        asyncio.run(self.storage().set_tokens(OAuthToken(access_token="t", token_type="Bearer")))
        mode = os.stat(self.path).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    @unittest.skipIf(sys.platform == "win32", "POSIX file modes only")
    def test_a_world_readable_file_is_tightened_on_write(self):
        # A file written by an older build (or a stray editor) must not stay
        # readable by every account on the machine just because it pre-existed.
        self.path.write_text("{}", encoding="utf-8")
        os.chmod(self.path, 0o644)
        asyncio.run(self.storage().set_tokens(OAuthToken(access_token="t", token_type="Bearer")))
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    @unittest.skipIf(sys.platform == "win32", "POSIX file modes only")
    def test_the_agent_home_directory_is_owner_only(self):
        nested = Path(self.tmp.name) / "fresh-home" / "mcp-tokens.json"
        storage = FileTokenStorage("srv", "https://s/mcp", path=nested)
        asyncio.run(storage.set_tokens(OAuthToken(access_token="t", token_type="Bearer")))
        self.assertEqual(os.stat(nested.parent).st_mode & 0o777, 0o700)

    def test_writes_are_atomic_and_leave_no_residue(self):
        storage = self.storage()
        asyncio.run(storage.set_tokens(OAuthToken(access_token="t1", token_type="Bearer")))
        asyncio.run(storage.set_tokens(OAuthToken(access_token="t2", token_type="Bearer")))
        siblings = [item.name for item in self.path.parent.iterdir()]
        self.assertEqual(siblings, [self.path.name])
        self.assertEqual(asyncio.run(storage.get_tokens()).access_token, "t2")

    def test_a_failed_write_leaves_the_previous_file_intact(self):
        # The file holds every server's entry, so a half-written save would log
        # them all out; the replace-based write can only succeed or do nothing.
        asyncio.run(self.storage().set_tokens(OAuthToken(access_token="keep", token_type="Bearer")))
        with mock.patch("src.llm.tools.mcp.oauth.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                asyncio.run(self.storage().set_tokens(OAuthToken(access_token="lost", token_type="Bearer")))
        self.assertEqual(asyncio.run(self.storage().get_tokens()).access_token, "keep")
        self.assertEqual([item.name for item in self.path.parent.iterdir()], [self.path.name])

    def test_helpers(self):
        self.assertFalse(has_stored_tokens("srv", "https://s/mcp", path=self.path))
        asyncio.run(self.storage().set_tokens(OAuthToken(access_token="t", token_type="Bearer")))
        self.assertTrue(has_stored_tokens("srv", "https://s/mcp", path=self.path))
        self.assertTrue(clear_tokens("srv", "https://s/mcp", path=self.path))
        self.assertFalse(has_stored_tokens("srv", "https://s/mcp", path=self.path))

    def test_clear_tokens_reports_false_when_nothing_stored(self):
        self.assertFalse(clear_tokens("srv", "https://s/mcp", path=self.path))

    def test_clear_tokens_leaves_other_servers_alone(self):
        asyncio.run(self.storage("a", "https://a/").set_tokens(OAuthToken(access_token="ta", token_type="Bearer")))
        asyncio.run(self.storage("b", "https://b/").set_tokens(OAuthToken(access_token="tb", token_type="Bearer")))
        self.assertTrue(clear_tokens("a", "https://a/", path=self.path))
        self.assertTrue(has_stored_tokens("b", "https://b/", path=self.path))


class TestCallbackServer(unittest.TestCase):
    def make_server(self):
        server = CallbackServer(port=0)
        server.start()
        self.addCleanup(server.stop)
        return server

    def test_receives_code_from_local_request(self):
        server = self.make_server()

        async def scenario():
            waiter = asyncio.create_task(server.wait_for_code(timeout=5))
            await asyncio.sleep(0.05)
            url = server.redirect_uri() + "?code=xyz&state=st"
            await asyncio.to_thread(urllib.request.urlopen, url)
            return await waiter

        code, state = asyncio.run(scenario())
        self.assertEqual(code, "xyz")
        self.assertEqual(state, "st")

    def test_redirect_uri_reports_the_bound_port(self):
        server = self.make_server()
        uri = server.redirect_uri()
        self.assertTrue(uri.startswith("http://127.0.0.1:"))
        self.assertTrue(uri.endswith("/callback"))
        self.assertNotIn(":0/", uri)

    def test_error_parameter_raises(self):
        server = self.make_server()

        async def scenario():
            waiter = asyncio.create_task(server.wait_for_code(timeout=5))
            await asyncio.sleep(0.05)
            url = server.redirect_uri() + "?error=access_denied"
            # The denial page answers 400, which urllib surfaces as an
            # exception; a browser just renders it. The waiter is what matters.
            with contextlib.suppress(urllib.error.HTTPError):
                await asyncio.to_thread(urllib.request.urlopen, url)
            return await waiter

        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(scenario())
        self.assertIn("access_denied", str(caught.exception))

    def test_auth_result_carries_state_and_iss(self):
        server = self.make_server()

        async def scenario():
            waiter = asyncio.create_task(server.wait_for_auth_result(timeout=5))
            await asyncio.sleep(0.05)
            url = server.redirect_uri() + "?code=c1&state=s1&iss=https%3A%2F%2Fas.example"
            await asyncio.to_thread(urllib.request.urlopen, url)
            return await waiter

        result = asyncio.run(scenario())
        self.assertIsInstance(result, AuthorizationCodeResult)
        self.assertEqual(result.code, "c1")
        self.assertEqual(result.state, "s1")
        self.assertEqual(result.iss, "https://as.example")

    def test_browser_gets_a_readable_page(self):
        server = self.make_server()

        async def scenario():
            waiter = asyncio.create_task(server.wait_for_code(timeout=5))
            await asyncio.sleep(0.05)
            response = await asyncio.to_thread(urllib.request.urlopen, server.redirect_uri() + "?code=c&state=s")
            body = response.read().decode("utf-8")
            await waiter
            return response.status, body

        status, body = asyncio.run(scenario())
        self.assertEqual(status, 200)
        self.assertIn("close this tab", body.lower())

    def test_stop_wakes_an_abandoned_waiter_immediately(self):
        # `wait_for_auth_result` blocks in a plain `queue.get` on a NON-daemon
        # executor thread. If `stop()` only closed the socket, that thread
        # would stay parked for the rest of the timeout and hang /quit in
        # `concurrent.futures`' atexit join.
        server = CallbackServer(port=0)
        server.start()

        async def scenario():
            waiter = asyncio.create_task(server.wait_for_auth_result(timeout=300))
            await asyncio.sleep(0.05)
            started = time.monotonic()
            server.stop()
            with self.assertRaises(RuntimeError):
                await asyncio.wait_for(waiter, timeout=5)
            return time.monotonic() - started

        self.assertLess(asyncio.run(scenario()), 2.0)

    def test_stop_leaves_no_executor_thread_behind(self):
        server = CallbackServer(port=0)
        server.start()

        async def scenario():
            waiter = asyncio.create_task(server.wait_for_auth_result(timeout=300))
            await asyncio.sleep(0.05)
            server.stop()
            with contextlib.suppress(RuntimeError):
                await asyncio.wait_for(waiter, timeout=5)

        asyncio.run(scenario())
        # asyncio.run() only returns after shutting its executor down, which it
        # can only do once the queue.get thread has actually returned.
        self.assertFalse(any(t.name.startswith("mcp-oauth-callback") and t.is_alive() for t in threading.enumerate()))

    def test_a_real_result_still_wins_over_the_stop_sentinel(self):
        server = self.make_server()

        async def scenario():
            await asyncio.to_thread(urllib.request.urlopen, server.redirect_uri() + "?code=real&state=s")
            server.stop()
            return await server.wait_for_code(timeout=5)

        code, _ = asyncio.run(scenario())
        self.assertEqual(code, "real")

    def test_timeout_raises_without_a_callback(self):
        server = self.make_server()
        with self.assertRaises(TimeoutError):
            asyncio.run(server.wait_for_code(timeout=0.2))

    def test_stop_shuts_the_thread_down_and_is_idempotent(self):
        server = CallbackServer(port=0)
        server.start()
        thread = server.thread
        self.assertTrue(thread.is_alive())
        server.stop()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        server.stop()

    def test_stop_releases_the_port_for_a_fresh_server(self):
        first = CallbackServer(port=0)
        first.start()
        port = first.port
        first.stop()
        second = CallbackServer(port=port)
        self.addCleanup(second.stop)
        self.assertEqual(second.port, port)

    def test_stop_without_start_closes_the_socket(self):
        server = CallbackServer(port=0)
        port = server.port
        server.stop()
        # The socket bound in __init__ must be released even if never served.
        reused = CallbackServer(port=port)
        self.addCleanup(reused.stop)
        self.assertEqual(reused.port, port)


class TestBuildOAuthProvider(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "tokens.json"
        self.config = McpServerConfig(name="remote", transport="http", url="https://r/mcp")

    def storage(self):
        return FileTokenStorage(self.config.name, self.config.url, path=self.path)

    def test_non_interactive_redirect_raises_needs_interactive_auth(self):
        provider, callback = build_oauth_provider(self.config, interactive=False, storage=self.storage())
        self.assertIsNone(callback)
        with self.assertRaises(NeedsInteractiveAuth) as caught:
            asyncio.run(provider.context.redirect_handler("https://as.example/authorize"))
        self.assertIn("/mcp login remote", str(caught.exception))

    def test_non_interactive_callback_raises_needs_interactive_auth(self):
        provider, _ = build_oauth_provider(self.config, interactive=False, storage=self.storage())
        with self.assertRaises(NeedsInteractiveAuth):
            asyncio.run(provider.context.callback_handler())

    def test_non_interactive_never_opens_a_browser(self):
        opened = []
        provider, _ = build_oauth_provider(self.config, interactive=False, storage=self.storage(), opener=opened.append)
        with self.assertRaises(NeedsInteractiveAuth):
            asyncio.run(provider.context.redirect_handler("https://as.example/authorize"))
        self.assertEqual(opened, [])

    def test_provider_is_an_httpx_auth_with_our_metadata(self):
        provider, _ = build_oauth_provider(self.config, interactive=False, storage=self.storage(), port=43110)
        metadata = provider.context.client_metadata
        self.assertEqual([str(uri) for uri in metadata.redirect_uris], ["http://127.0.0.1:43110/callback"])
        self.assertIn("authorization_code", metadata.grant_types)
        self.assertIn("refresh_token", metadata.grant_types)
        self.assertEqual(metadata.response_types, ["code"])

    def test_interactive_opens_the_browser_and_waits_for_the_code(self):
        opened = []
        provider, callback = build_oauth_provider(
            self.config, interactive=True, storage=self.storage(), port=0, opener=opened.append
        )
        self.addCleanup(callback.stop)
        callback.start()

        async def scenario():
            await provider.context.redirect_handler("https://as.example/authorize?x=1")
            waiter = asyncio.create_task(provider.context.callback_handler())
            await asyncio.sleep(0.05)
            await asyncio.to_thread(urllib.request.urlopen, callback.redirect_uri() + "?code=ok&state=st")
            return await waiter

        result = asyncio.run(scenario())
        self.assertEqual(opened, ["https://as.example/authorize?x=1"])
        self.assertIsInstance(result, AuthorizationCodeResult)
        self.assertEqual(result.code, "ok")
        self.assertEqual(result.state, "st")

    def test_interactive_metadata_matches_the_bound_callback_port(self):
        provider, callback = build_oauth_provider(self.config, interactive=True, storage=self.storage(), port=0)
        self.addCleanup(callback.stop)
        redirect = [str(uri) for uri in provider.context.client_metadata.redirect_uris]
        self.assertEqual(redirect, [callback.redirect_uri()])

    def test_a_browser_that_will_not_open_does_not_break_the_flow(self):
        def broken_opener(url):
            raise RuntimeError("no browser here")

        provider, callback = build_oauth_provider(
            self.config, interactive=True, storage=self.storage(), port=0, opener=broken_opener
        )
        self.addCleanup(callback.stop)
        # The user can still paste the URL by hand, so a dead browser must not
        # abort the flow before the callback server ever gets a chance.
        asyncio.run(provider.context.redirect_handler("https://as.example/authorize"))

    def test_a_failed_provider_build_closes_the_callback_socket(self):
        # The callback binds in its constructor, before the provider is built.
        # Leaking that socket would wedge every later login this session, since
        # the production port is fixed.
        created = []
        real_server = oauth_module.CallbackServer

        def spy(port):
            server = real_server(port)
            created.append(server)
            return server

        with mock.patch.object(oauth_module, "CallbackServer", spy):
            with mock.patch.object(oauth_module, "OAuthClientProvider", side_effect=RuntimeError("bad metadata")):
                with self.assertRaises(RuntimeError):
                    build_oauth_provider(self.config, interactive=True, storage=self.storage(), port=0)

        self.assertEqual(len(created), 1)
        reused = CallbackServer(port=created[0].port)
        self.addCleanup(reused.stop)
        self.assertEqual(reused.port, created[0].port)

    def test_storage_defaults_to_the_shared_token_file(self):
        provider, _ = build_oauth_provider(self.config, interactive=False)
        self.assertIsInstance(provider.context.storage, FileTokenStorage)
        self.assertEqual(provider.context.storage.path, default_token_path())


if __name__ == "__main__":
    unittest.main()
