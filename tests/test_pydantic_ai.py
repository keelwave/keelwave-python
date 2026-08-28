"""Tests for the pydantic-ai adapter.

Uses pydantic-ai's built-in TestModel so no real LLM calls are made.
"""

import os
import pytest

from keelwave import Keelwave
from keelwave.adapters.pydantic_ai import instrument, run_with_steps


@pytest.fixture
def client(_sync_project):
    with Keelwave(
        api_key=_sync_project,
        endpoint=os.getenv("KEELWAVE_ENDPOINT", "http://localhost:8080"),
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_instrument_opens_run_and_returns_result(client):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(), name="test-agent")
    result = await instrument(client, agent, "say hi")
    assert result is not None
    assert result.output is not None


@pytest.mark.asyncio
async def test_instrument_with_tool_emits_steps(client):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    calls: list[str] = []

    agent = Agent(TestModel(), name="tool-agent")

    @agent.tool_plain
    def greet(name: str) -> str:
        calls.append(name)
        return f"Hello, {name}!"

    result = await instrument(client, agent, "greet Alice")
    assert result is not None


@pytest.mark.asyncio
async def test_run_with_steps_reuses_existing_run(client):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(), name="reuse-agent")

    with client.run("outer-run", input="task") as run:
        result = await run_with_steps(run, agent, "inner task")
        assert result is not None
        run_id = run.id

    assert run_id  # run was properly opened


@pytest.mark.asyncio
async def test_tool_return_emits_ok_true(client):
    import unittest.mock as mock

    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(), name="ok-agent")

    @agent.tool_plain
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    with client.run("ok-run", input="greet") as run:
        with mock.patch.object(run, "tool_call", wraps=run.tool_call) as mock_tc:
            await run_with_steps(run, agent, "greet Alice")
            mock_tc.assert_called_once()
            kwargs = mock_tc.call_args.kwargs
            assert kwargs["ok"] is True
            assert "result" in kwargs["output"]


@pytest.mark.asyncio
async def test_model_retry_emits_ok_false(client):
    import unittest.mock as mock

    from pydantic_ai import Agent, ModelRetry
    from pydantic_ai.models.test import TestModel

    attempts: list[int] = []
    agent = Agent(TestModel(), name="retry-agent")

    @agent.tool_plain(retries=2)
    def flaky(q: str) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise ModelRetry("bad args")
        return "recovered"

    with client.run("retry-run", input="flaky") as run:
        with mock.patch.object(run, "tool_call", wraps=run.tool_call) as mock_tc:
            await run_with_steps(run, agent, "call flaky")
            assert mock_tc.call_count == 2
            first = mock_tc.call_args_list[0].kwargs
            second = mock_tc.call_args_list[1].kwargs
            assert first["ok"] is False
            assert "bad args" in first["output"]["error"]
            assert second["ok"] is True


@pytest.mark.asyncio
async def test_dict_output_tool_emits_step(client):
    import unittest.mock as mock

    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(), name="dict-agent")

    @agent.tool_plain
    def search(q: str) -> dict:
        return {"hits": 3}

    with client.run("dict-run", input="search") as run:
        with mock.patch.object(run, "tool_call", wraps=run.tool_call) as mock_tc:
            await run_with_steps(run, agent, "search things")
            mock_tc.assert_called_once()
            kwargs = mock_tc.call_args.kwargs
            assert kwargs["ok"] is True
            assert kwargs["output"] != {}


@pytest.mark.asyncio
async def test_empty_output_stays_ok_true(client):
    import unittest.mock as mock

    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(), name="empty-agent")

    @agent.tool_plain
    def blank(q: str) -> str:
        return ""

    with client.run("empty-run", input="blank") as run:
        with mock.patch.object(run, "tool_call", wraps=run.tool_call) as mock_tc:
            await run_with_steps(run, agent, "call blank")
            mock_tc.assert_called_once()
            kwargs = mock_tc.call_args.kwargs
            assert kwargs["ok"] is True
            assert kwargs["output"] == {}


@pytest.mark.asyncio
async def test_tool_success_false_reaches_ingest(client):
    import unittest.mock as mock

    from pydantic_ai import Agent, ModelRetry
    from pydantic_ai.models.test import TestModel

    attempts: list[int] = []
    agent = Agent(TestModel(), name="wire-agent")

    @agent.tool_plain(retries=2)
    def flaky(q: str) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise ModelRetry("nope")
        return "fine"

    with mock.patch.object(
        client, "ingest_agent_step", wraps=client.ingest_agent_step
    ) as mock_ing:
        with client.run("wire-run", input="flaky") as run:
            await run_with_steps(run, agent, "call flaky")

    successes = [
        c.kwargs["tool_success"]
        for c in mock_ing.call_args_list
        if c.kwargs.get("step_type") == "tool_call"
    ]
    assert False in successes and True in successes


@pytest.mark.asyncio
async def test_instrument_reuses_ambient_run(client):
    """When called inside @agent, instrument() reuses the ambient run."""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    inner_agent = Agent(TestModel(), name="inner")

    @client.agent(name="outer-agent")
    async def outer(task: str) -> str:
        result = await instrument(client, inner_agent, task)
        return str(result.output)

    result = await outer("test task")
    assert result is not None
