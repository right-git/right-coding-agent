# Evaluation prompts

Give each prompt verbatim to both agents, one task per fresh session
(`uv run python -m src.main` vs `uv run python -m evaluation.main`), and
record the footer's `session` and `tools` values. Repeat close results 2–3
times. Predictions name the architecture expected to spend fewer tokens.

## 1. Baseline single call — predicted: direct

> Take a screenshot of my screen and tell me which application is in the
> foreground.

Measures the meta agent's discovery overhead (search_tools / get_tool /
run_tools vs one direct call).

## 2. Big page, small answer — predicted: meta (large margin)

> Fetch https://en.wikipedia.org/wiki/Python_(programming_language) and
> tell me only: how many characters the page is, and its first two
> sentences.

Direct puts the whole page into context and re-bills it every call; meta
computes in-script and returns only the extract.

## 3. Parallel fan-out — predicted: meta

> Fetch https://example.com, https://www.python.org and
> https://www.rust-lang.org. Tell me which page is longest, and give one
> sentence about each.

`parallel()` plus in-script comparison vs three full pages in history.

## 4. Sequential desktop workflow — predicted: meta

Prep: open Notepad.

> In Notepad, click the text area, type "hello from the agent", then take
> a screenshot so you can confirm the text is actually there.

Three round-trips vs one script; screenshot cost identical for both.

## 5. Where is it? — predicted: near-tie

Prep: open any app with a settings button.

> I can't find the settings button in this window. Show me where it is and
> explain what clicking it does.

Single screen_locate with mark=True; control task for overhead on trivial asks.

## 6. Polling — predicted: meta (large margin)

Prep: open an online countdown timer set to ~2 minutes.

> There's a countdown timer on my screen. Watch it and tell me the moment
> it reaches zero.

`while` + `sleep()` in one script vs one billed round-trip per check.

## 7. Enumerate the screen — predicted: meta

> List every clickable button you can find in the current window, each
> with its screen coordinates.

Several locate queries: an in-script loop vs a round-trip per query.


Зайди в 