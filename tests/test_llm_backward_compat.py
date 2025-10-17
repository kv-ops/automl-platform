"""Regression tests for backward-compatible LLM helpers."""

import asyncio


def test_backward_compatible_import():
    """DataCleaningAgent should remain importable from automl_platform.llm."""

    from automl_platform.llm import DataCleaningAgent

    agent = DataCleaningAgent(None)

    assert agent is not None
    assert agent.uses_real_llm is False


def test_data_cleaning_agent_fallback_event_loop():
    """The fallback provider must support async generate calls without errors."""

    from automl_platform.llm import DataCleaningAgent

    agent = DataCleaningAgent()

    result = asyncio.run(agent.llm.generate("ping"))

    assert result.content.startswith("LLM not configured")
    assert result.metadata.get("fallback") is True
