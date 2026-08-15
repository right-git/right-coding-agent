"""The LLM layer: client, agents, and everything the model touches.

Layout:
- `client.py` / `agents.py` — provider failover loop and the concrete agents;
- `types.py` — `LLMProvider`, `TurnUsage`, and shared type aliases;
- `providers/` — provider-specific integrations (OpenRouter catalog, ...);
- `middlewares/` — steps that run around every model call;
- `tools/` — one subpackage per concern: `meta/` (the meta-tool surface,
  registry, and sandbox engine), `parser/` and `computer/` (capabilities);
- `statistics/` — accounting and instrumentation (token/cost usage, script
  tool-call counters, future metrics);
- `utils.py` — client-side helpers.

Kept import-light on purpose: `src.main` imports the heavy agent stack
lazily, so submodules are imported explicitly rather than re-exported here.
"""
