import asyncio
from typing import Any, Literal

import httpx
from bs4 import BeautifulSoup
from html_to_markdown import ConversionOptions, convert
from src.config.logging import logger


class WebParser:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message with contextual tags if verbose is enabled.

        Args:
            message (str): The message to log.
            level (str): The logging level as a string. One of "debug", "info", "warning", "error", "critical".
        """
        if not self.verbose:
            return
        msg = f"{message}"

        if level == "debug":
            logger.debug(msg)
        elif level == "info":
            logger.info(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        elif level == "critical":
            logger.critical(msg)
        else:
            logger.info(msg)

    def log_debug(self, message: str) -> None:
        self.log(message, level="debug")

    def log_info(self, message: str) -> None:
        self.log(message, level="info")

    def log_warning(self, message: str) -> None:
        self.log(message, level="warning")

    def log_error(self, message: str) -> None:
        self.log(message, level="error")

    def log_critical(self, message: str) -> None:
        self.log(message, level="critical")

    def _extract_front_matter(self, markdown: str) -> str:
        lines = markdown.splitlines()
        if not lines or lines[0] != "---":
            return ""

        for index, line in enumerate(lines[1:], start=1):
            if line == "---":
                return "\n".join(lines[: index + 1])

        return ""

    async def make_request(
        self,
        url: str,
        method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"] = "GET",
        payload: dict | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        tries: int = 3,
        sleep_on_429: int = 30,
        timeout: int = 30,
        follow_redirects: bool = True,
    ):
        last_exception = None

        method = method.upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
            raise ValueError(f"Unsupported HTTP method: {method}")

        for attempt in range(tries):
            try:
                async with httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=follow_redirects,
                    headers=headers,
                    params=params,
                ) as client:
                    request_args = {}
                    if method in {"POST", "PUT", "PATCH"} and payload is not None:
                        request_args["json"] = payload

                    response = await client.request(method, url, **request_args)

                    if response.status_code == 429:
                        if attempt < tries - 1:
                            await asyncio.sleep(sleep_on_429)
                            continue
                        response.raise_for_status()

                    response.raise_for_status()
                    self.log(response.content, "debug")
                    return response

            except httpx.HTTPStatusError as exc:
                last_exception = exc
                if exc.response.status_code == 429 and attempt < tries - 1:
                    await asyncio.sleep(sleep_on_429)
                    continue
                if attempt < tries - 1:
                    await asyncio.sleep(1)
                    continue
                raise
            except (httpx.RequestError, Exception) as exc:
                last_exception = exc
                if attempt < tries - 1:
                    await asyncio.sleep(1)
                    continue
                raise

        raise (
            last_exception
            if last_exception
            else RuntimeError("make_request failed unexpectedly")
        )

    def parse_html(
        self, html: str, metadata_primary: bool = False
    ) -> dict[str, Any] | str:
        soup = BeautifulSoup(html, "html.parser")
        has_description_meta = any(
            meta.get("name", "").lower() == "description"
            for meta in soup.find_all("meta")
        )

        for meta in soup.find_all("meta"):
            if meta.get("name", "").lower() != "description":
                meta.decompose()

        result = convert(soup.prettify(), options=ConversionOptions(skip_images=True))

        if metadata_primary:
            content = result["content"] if isinstance(result, dict) else result
            if not isinstance(content, str):
                return ""
            if not has_description_meta:
                return content
            front_matter = self._extract_front_matter(content)
            return front_matter or content

        return result

    async def parse_page(self, url: str) -> str:
        response = await self.make_request(url)
        html = response.text if hasattr(response, "text") else str(response.content)
        result = self.parse_html(html)
        return result

    
