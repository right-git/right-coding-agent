"""OAuth for remote MCP servers: token file, loopback callback, providers.

Single-user CLI shape: tokens live in one JSON file under `~/.right-agent/`,
consent happens in the user's own browser, and the authorization code comes
back to a one-shot HTTP server on 127.0.0.1. Nothing here ever raises into the
REPL — the manager maps a failed authorization to a `NEEDS_AUTH` status.

Verified live against the installed SDK, `mcp` 2.0.0; two shapes moved since
the 1.x-era draft this was written against:

- `callback_handler` must return a `mcp.shared.auth.AuthorizationCodeResult`
  (`code`/`state`/`iss`), not the 1.x `(code, state)` tuple. `iss` is not
  decoration: the SDK validates the RFC 9207 authorization-response issuer and
  rejects a response that omits `iss` when the server advertised support for
  it, so the callback server forwards that query parameter too.
- `OAuthClientProvider` is an `httpx2.Auth` (the SDK vendors httpx2), which is
  what `create_mcp_http_client(auth=...)` and `sse_client(auth=...)` in
  `transports.py` expect. `OAuthClientMetadata` requires only `redirect_uris`;
  everything else below is supplied to pin the flow we actually support.

The callback server binds its socket in `__init__`, not in `start()`, so
`redirect_uri()` can report the real port before the provider's client
metadata is built — that is what makes `port=0` usable in tests.
"""

import asyncio
import json
import os
import queue
import threading
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import AuthorizationCodeResult, OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from src.config.logging import logger
from src.config.settings import settings

from .config import McpServerConfig

# How long the browser flow may take before login gives up. Consent screens
# involve typing passwords and approving 2FA, so this is generous on purpose.
CALLBACK_TIMEOUT = 300.0

CLIENT_NAME = "right-coding-agent"

_CALLBACK_PATH = "/callback"

_PAGE = (
    "<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
    "<body style='font-family:system-ui;text-align:center;padding-top:4rem'>"
    "<h2>{heading}</h2><p>{message}</p></body></html>"
)

# One process-wide lock around the token file's read-modify-write: several
# servers can connect (and refresh) concurrently, and they all share the file.
_FILE_LOCK = threading.Lock()


class NeedsInteractiveAuth(Exception):
    """The server wants a browser consent this connection may not open.

    Raised by the non-interactive provider's handlers; `McpManager` maps it to
    the `NEEDS_AUTH` status that tells the user to run `/mcp login <name>`.
    """


def default_token_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".right-agent" / "mcp-tokens.json"


def _entry_key(server_name: str, server_url: str) -> str:
    """Tokens are bound to a name *and* a url: repointing a server re-logs in."""
    return f"{server_name}|{server_url}"


def _read_all(path: Path) -> dict[str, Any]:
    """The whole token file; {} when missing, unreadable, or corrupt.

    A broken file must never take down a connection — the worst case is one
    re-login, which is exactly what an empty mapping produces.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning("Unreadable MCP token file [{}]; treating it as empty", path)
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_all(path: Path, payload: dict[str, Any]) -> None:
    """Persist the token file, owner-readable only.

    Opened through `os.open` with mode 0o600 so a *new* file is never even
    briefly world-readable; the explicit `chmod` afterwards fixes an existing
    file (whose mode `O_CREAT` leaves alone) and is tolerated failing on
    filesystems that do not implement POSIX modes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    try:
        os.chmod(path, 0o600)
    except OSError as error:
        logger.debug("Could not restrict permissions on MCP token file [{}]: {}", path, error)


def _entry(path: Path, key: str) -> dict[str, Any]:
    entry = _read_all(path).get(key)
    return entry if isinstance(entry, dict) else {}


class FileTokenStorage:
    """The SDK's `TokenStorage` protocol over one shared JSON file.

    Layout: `{"<name>|<url>": {"tokens": {...}, "client_info": {...}}}`. Every
    read is tolerant — a corrupt file, a half-written entry, or a payload the
    current SDK models reject all read as "nothing stored", which costs a
    re-login instead of a crashed connection.
    """

    def __init__(self, server_name: str, server_url: str, path: Path | None = None) -> None:
        self.server_name = server_name
        self.server_url = server_url or ""
        self.path = Path(path) if path is not None else default_token_path()
        self.key = _entry_key(server_name, self.server_url)

    def _load(self, field: str, model: Any) -> Any | None:
        raw = _entry(self.path, self.key).get(field)
        if not isinstance(raw, dict):
            return None
        try:
            return model.model_validate(raw)
        except Exception:
            logger.warning("Ignoring unreadable MCP {} for server [{}]", field, self.server_name)
            return None

    def _store(self, field: str, value: Any) -> None:
        dumped = value.model_dump(mode="json", exclude_none=True)
        with _FILE_LOCK:
            payload = _read_all(self.path)
            entry = payload.get(self.key)
            if not isinstance(entry, dict):
                entry = {}
            entry[field] = dumped
            payload[self.key] = entry
            _write_all(self.path, payload)

    async def get_tokens(self) -> OAuthToken | None:
        return self._load("tokens", OAuthToken)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._store("tokens", tokens)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._load("client_info", OAuthClientInformationFull)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._store("client_info", client_info)


def has_stored_tokens(server_name: str, server_url: str, path: Path | None = None) -> bool:
    """True when a usable access token is on file for this server."""
    file = Path(path) if path is not None else default_token_path()
    tokens = _entry(file, _entry_key(server_name, server_url or "")).get("tokens")
    return isinstance(tokens, dict) and bool(tokens.get("access_token"))


def clear_tokens(server_name: str, server_url: str, path: Path | None = None) -> bool:
    """Forget everything stored for one server; True when something went.

    The client registration goes with the tokens: a logout should leave no
    credential behind, and a stale dynamic registration (whose redirect URI may
    no longer match) is worth re-doing on the next login.
    """
    file = Path(path) if path is not None else default_token_path()
    key = _entry_key(server_name, server_url or "")
    with _FILE_LOCK:
        payload = _read_all(file)
        if key not in payload:
            return False
        payload.pop(key)
        _write_all(file, payload)
    return True


class _CallbackHandler(BaseHTTPRequestHandler):
    """Answers exactly one authorization redirect, then hands it to the queue."""

    # BaseHTTPRequestHandler logs every hit to stderr, straight over the REPL.
    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("MCP OAuth callback: {}", format % args)

    def _respond(self, status: int, title: str, heading: str, message: str) -> None:
        body = _PAGE.format(title=title, heading=heading, message=message).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        parsed = urlparse(self.path)
        if parsed.path != _CALLBACK_PATH:
            # Browsers ask for /favicon.ico; that must not count as a callback.
            self._respond(404, "Not found", "Not found", "Nothing to see here.")
            return
        params = parse_qs(parsed.query)
        first = {key: values[0] for key, values in params.items() if values}
        if "error" in first:
            detail = first.get("error_description") or first["error"]
            self.server.results.put({"error": detail})  # type: ignore[attr-defined]
            self._respond(400, "Authorization failed", "Authorization failed", detail)
            return
        if "code" not in first:
            self.server.results.put({"error": "authorization callback carried no code"})  # type: ignore[attr-defined]
            self._respond(400, "Authorization failed", "Authorization failed", "No authorization code received.")
            return
        self.server.results.put(  # type: ignore[attr-defined]
            {"code": first["code"], "state": first.get("state"), "iss": first.get("iss")}
        )
        self._respond(200, "Authorized", "Authorized", "You can close this tab.")


class CallbackServer:
    """A one-shot loopback listener for the OAuth authorization redirect.

    The socket binds in `__init__` so `redirect_uri()` reports the real port
    even when the caller asked for port 0; `start()` only adds the serving
    thread. `stop()` is idempotent and safe before `start()` — which matters,
    because `HTTPServer.shutdown()` blocks forever if `serve_forever` never
    ran, and login's `finally` must never hang the REPL.
    """

    def __init__(self, port: int) -> None:
        self.server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
        self.server.results = queue.Queue()  # type: ignore[attr-defined]
        self.thread: threading.Thread | None = None
        self._serving = False
        self._closed = False

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}{_CALLBACK_PATH}"

    def start(self) -> None:
        if self._serving or self._closed:
            return
        self._serving = True
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="mcp-oauth-callback",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._serving:
            self._serving = False
            try:
                self.server.shutdown()
            except Exception:
                logger.exception("MCP OAuth callback server did not shut down cleanly")
        try:
            self.server.server_close()
        except Exception:
            logger.exception("MCP OAuth callback socket did not close cleanly")
        thread, self.thread = self.thread, None
        if thread is not None:
            thread.join(timeout=5.0)

    async def wait_for_auth_result(self, timeout: float = CALLBACK_TIMEOUT) -> AuthorizationCodeResult:
        """Block (off-loop) until the browser comes back, or time out."""
        try:
            received = await asyncio.to_thread(self.server.results.get, True, timeout)  # type: ignore[attr-defined]
        except queue.Empty:
            raise TimeoutError(f"no OAuth callback received within {timeout:.0f}s") from None
        if received.get("error"):
            raise RuntimeError(f"authorization failed: {received['error']}")
        return AuthorizationCodeResult(
            code=received["code"],
            state=received.get("state"),
            iss=received.get("iss"),
        )

    async def wait_for_code(self, timeout: float = CALLBACK_TIMEOUT) -> tuple[str, str | None]:
        result = await self.wait_for_auth_result(timeout)
        return result.code, result.state


def build_oauth_provider(
    config: McpServerConfig,
    *,
    interactive: bool,
    storage: FileTokenStorage | None = None,
    port: int | None = None,
    opener: Callable[[str], Any] | None = None,
) -> tuple[OAuthClientProvider, CallbackServer | None]:
    """An `httpx2.Auth` OAuth provider for one server, plus its listener.

    `interactive=False` is the everyday path: it can refresh an existing token
    silently, and raises `NeedsInteractiveAuth` the moment the server wants a
    consent screen. `interactive=True` (only `McpManager.login`) opens the
    browser and waits on the returned `CallbackServer`, which the caller owns
    and must `stop()`.
    """
    port = settings.mcp_oauth_port if port is None else port
    storage = storage or FileTokenStorage(config.name, config.url or "")
    opener = opener or webbrowser.open
    callback = CallbackServer(port=port) if interactive else None
    redirect_uri = callback.redirect_uri() if callback else f"http://127.0.0.1:{port}{_CALLBACK_PATH}"
    hint = f"run /mcp login {config.name}"

    async def redirect_handler(authorization_url: str) -> None:
        if callback is None:
            raise NeedsInteractiveAuth(hint)
        # Logged unconditionally: when the browser cannot be opened (headless
        # box, ssh session) this line is the only way to reach the consent URL.
        logger.info("MCP server [{}] authorization URL: {}", config.name, authorization_url)
        try:
            opener(authorization_url)
        except Exception as error:
            logger.warning("Could not open a browser for MCP server [{}]: {}", config.name, error)

    async def callback_handler() -> AuthorizationCodeResult:
        if callback is None:
            raise NeedsInteractiveAuth(hint)
        return await callback.wait_for_auth_result()

    provider = OAuthClientProvider(
        server_url=config.url or "",
        client_metadata=OAuthClientMetadata(
            client_name=CLIENT_NAME,
            redirect_uris=[redirect_uri],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
    return provider, callback
