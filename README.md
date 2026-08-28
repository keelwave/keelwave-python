# keelwave (Python SDK)

Zero-friction tracing for AI agents.

`keelwave` records agent runs, decision steps, tool calls, token/cost totals,
and loop-detection fingerprints, then ships them to a keelwave server that
surfaces **why** an agent failed and **where**. An API key is the only required
config: wrap your LLM client or decorate your agent, and the run is traced.

> **Status: early / MVP (pre-1.0).** The wire protocol and public API may still
> change. The package was previously named `vigil`; it is now `keelwave`.

## Install

```bash
pip install keelwave
```

Requires Python 3.10+. The base install pulls in `httpx` and nothing else.

Provider adapters need their own SDK, available as extras:

```bash
pip install 'keelwave[openai]'        # wrap_openai
pip install 'keelwave[anthropic]'     # wrap_anthropic
pip install 'keelwave[pydantic-ai]'   # adapters.pydantic_ai
pip install 'keelwave[all]'           # all three
```

The SDK needs a running keelwave server to send traces to. Self-host it from the
core repo: [github.com/keelwave/keelwave](https://github.com/keelwave/keelwave).
By default the client points at `http://localhost:8080`.

## Quickstart

Construct a client with your API key and endpoint, then decorate your agent.
`@client.agent` opens and closes a run automatically; `@client.observe` records
each decorated function as a step and links it to the active run.

```python
import os
from keelwave import Keelwave

client = Keelwave(
    api_key=os.environ["KEELWAVE_API_KEY"],
    endpoint=os.environ.get("KEELWAVE_ENDPOINT", "http://localhost:8080"),
)


@client.observe(name="web_search", step_type="tool_call")
def web_search(q: str) -> dict:
    # Replace with a real search implementation.
    return {"results": [f"result for: {q}"]}


@client.agent(name="research-agent")
def run_agent(task: str) -> str:
    results = web_search(q=task)
    return f"Found: {results['results'][0]}"


answer = run_agent("what is keelwave")
print(answer)
```

That run — its input/output, every `web_search` call, step ordering, timings,
and any duplicate-tool-call loop — is now visible on your keelwave server.

Prefer explicit control? Use the run context manager directly:

```python
with client.run("research-agent", input="what is keelwave") as run:
    run.step("think", content="deciding which tool to call")
    run.tool_call("web_search", input={"q": "keelwave"}, output={"hits": 3})
    run.set_output("done")
```

An async equivalent is available via `AsyncKeelwave` / `AsyncRun`.

## What's captured

- **Decision steps** — `think`, `tool_call`, `result`, or any custom `step_type`,
  in order, with content and metadata.
- **Tool calls** — name, input, output, success flag, and latency.
  Since 0.1.2 the pydantic-ai adapter derives the success flag from the real
  outcome: a tool that raises `ModelRetry` (or otherwise produces a retry
  prompt) records a **failed** tool call with `{"error": ...}` as its output.
  Earlier versions recorded every adapter tool call as successful, so success
  rates may drop after upgrading — that is the metric becoming accurate.
  Manual `run.tool_call(..., ok=)` is unchanged (`ok` still defaults to `True`).
- **Loop detection** — each tool call is fingerprinted (SHA-256 of name + input)
  client-side; the first duplicate marks the run as a loop, with no extra code.
- **Tokens & cost** — per-step and per-run totals.
- **Run outcome** — status (`completed` / `failed`), termination reason,
  duration, and output.
- **LLM traces** — model, provider, token usage, latency, and errors, emitted
  automatically by the provider adapters (below).

## Provider adapters

Wrap your provider client once. Every call is recorded to keelwave's `ai_traces`
and auto-linked to the active run:

```python
import anthropic
claude = client.wrap_anthropic(anthropic.Anthropic())

import openai
gpt = client.wrap_openai(openai.OpenAI())
```

Both accept a `provider=` override for OpenAI-/Anthropic-compatible endpoints
(e.g. `groq`, `together`, `deepseek`, `openrouter`). The wrapped client is a
drop-in proxy — call `claude.messages.create(...)` or
`gpt.chat.completions.create(...)` exactly as before.

## Learn more

- keelwave server (self-host): [github.com/keelwave/keelwave](https://github.com/keelwave/keelwave)
- Runnable examples: see [`examples/`](examples/) — decorator agent, multi-turn
  research agent, and a deliberate looping agent that trips loop detection.

## License

MIT. See [LICENSE](LICENSE).
