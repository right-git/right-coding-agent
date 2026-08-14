# Architecture evaluation

Two agents, same brain, different tool plumbing:

| | meta (production) | direct (control) |
|---|---|---|
| entry | `uv run python -m src.main` | `uv run python -m evaluation.main` |
| tools the model sees | 3 (`search_tools`, `get_tool`, `run_tools`) | all of them (`web_search` + every `screen_*`) |
| tool execution | inside one sandboxed `run_tools` script | one model round-trip per call |
| schema cost | flat, however many tools are registered | grows with every registered tool, paid on **every** call |

Everything else is identical and comes from `src/`: LLM client with
retry/failover, summarization at 40k tokens, screenshot attachment as vision
messages, JSON request logging, the usage footer.

## Running the experiment

1. Start one agent, give it the task, and when it finishes note the footer:

   ```
   ctx … · turn 66,541 in + 674 out ($0.0262) · tools 3 (+7 in scripts) · session 88,124 tokens ($0.0348)
   ```

2. Quit, start the other agent, give it the **same task verbatim**, note its
   footer.

3. Compare the `session` totals (tokens and dollars) and the tool counts.
   For the direct agent every tool call is a model round-trip, so expect
   more `tools N` and a bigger gap between `ctx` and `turn` input; for the
   meta agent inner calls show up as `(+N in scripts)` and cost no
   round-trips.

The direct REPL prints both architectures' estimated schema cost per model
call at startup — that is the fixed overhead the meta layer removes, and it
is also the number that keeps growing as more tools get registered.

Use one task per session (or `/clear`, which in this REPL also resets the
session counters) so the `session` line measures exactly one run. Model
nondeterminism means single runs are indicative, not proof — repeat the task
a few times if the numbers are close.
