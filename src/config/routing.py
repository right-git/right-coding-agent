"""Per-model provider routing pins for OpenRouter requests.

`provider_pins.json` at the repo root maps model-id prefixes to the
`provider.order` list sent with matching requests. The file exists because
prompt caches are per-endpoint: OpenRouter serves one model from many
endpoints (Bedrock alone has three regions, each an isolated cache), and
load-balanced routing turns cache hits into a lottery of full-price calls
plus cache-write premiums.

Rules: keys are model-id prefixes and the longest match wins, so an exact
model id overrides a family pin; values are OpenRouter provider slugs; an
empty list disables pinning for that prefix; keys starting with "_" are
comments. A missing file falls back to `DEFAULT_PROVIDER_PINS`, and a broken
one is logged and falls back too — a typo must not silently disable the
caching economics. An existing file is taken literally, so `{}` means
"no pins" on purpose.
"""

import json
from pathlib import Path

from src.config.logging import logger

PROVIDER_PINS_FILE = Path(__file__).resolve().parents[2] / "provider_pins.json"
# Mirrors the shipped provider_pins.json; see its _comment keys for why each
# provider was chosen.
DEFAULT_PROVIDER_PINS: dict[str, list[str]] = {
    "anthropic/": ["anthropic"],
    "google/": ["google-vertex/global"],
    "openai/": ["openai"],
}


def load_provider_pins(path: Path | str | None = None) -> dict[str, list[str]]:
    """The pin mapping from `provider_pins.json`, validated entry by entry."""
    file = PROVIDER_PINS_FILE if path is None else Path(path)
    if not file.exists():
        return dict(DEFAULT_PROVIDER_PINS)
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning("Ignoring unreadable provider pins file [{}]: {}", file, error)
        return dict(DEFAULT_PROVIDER_PINS)
    if not isinstance(payload, dict):
        logger.warning("Ignoring provider pins file [{}]: the top level must be an object", file)
        return dict(DEFAULT_PROVIDER_PINS)

    pins: dict[str, list[str]] = {}
    for prefix, order in payload.items():
        if not isinstance(prefix, str) or prefix.startswith("_"):
            continue
        if not isinstance(order, list) or not all(isinstance(slug, str) and slug for slug in order):
            logger.warning(
                "Skipping provider pin [{}] in [{}]: the value must be a list of provider slugs", prefix, file
            )
            continue
        pins[prefix] = list(order)
    return pins


def provider_order_for(model_name: str, pins: dict[str, list[str]]) -> list[str] | None:
    """The pinned provider order for a model; the longest matching prefix wins."""
    best_prefix: str | None = None
    for prefix in pins:
        if model_name.startswith(prefix) and (best_prefix is None or len(prefix) > len(best_prefix)):
            best_prefix = prefix
    if best_prefix is None:
        return None
    order = pins[best_prefix]
    return list(order) if order else None
