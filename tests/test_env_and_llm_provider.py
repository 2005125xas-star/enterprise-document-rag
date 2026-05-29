from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.streamlit_app import MISSING_PROVIDER_KEY_MESSAGE, clean_error_message
from src.llm.providers import MockProvider, OpenAICompatibleProvider, ProviderConfigurationError, create_llm_provider


class FakeOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=FakeCompletions())


class FakeCompletions:
    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="Answer from evidence [1].")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_auto_without_environment_keys_uses_mock_provider() -> None:
    provider = create_llm_provider({"provider": "auto"}, environ={})

    assert isinstance(provider, MockProvider)


def test_llm_provider_mock_overrides_config() -> None:
    provider = create_llm_provider(
        {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
        environ={"LLM_PROVIDER": "mock"},
    )

    assert isinstance(provider, MockProvider)


def test_openai_compatible_without_llm_api_key_raises_clean_error() -> None:
    with pytest.raises(ProviderConfigurationError, match="LLM_API_KEY"):
        create_llm_provider(
            {"provider": "auto", "model": "gpt-4o-mini", "temperature": 0.0},
            environ={"LLM_PROVIDER": "openai_compatible"},
            client_factory=FakeOpenAIClient,
        )


def test_openai_compatible_env_creates_provider_without_external_call() -> None:
    captured: dict = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeOpenAIClient(**kwargs)

    provider = create_llm_provider(
        {"provider": "auto", "model": "config-model", "temperature": 0.0},
        environ={
            "LLM_PROVIDER": "openai_compatible",
            "LLM_API_KEY": "fake-test-key",
            "LLM_BASE_URL": "https://api.deepseek.com",
            "LLM_MODEL": "deepseek-v4-flash",
        },
        client_factory=factory,
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "openai-compatible"
    assert provider.model == "deepseek-v4-flash"
    assert provider.backend_label == "api.deepseek.com"
    assert captured == {"api_key": "fake-test-key", "base_url": "https://api.deepseek.com"}


def test_environment_variables_override_config() -> None:
    provider = create_llm_provider(
        {"provider": "mock", "model": "config-model", "temperature": 0.0},
        environ={
            "LLM_PROVIDER": "openai_compatible",
            "LLM_API_KEY": "fake-test-key",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_MODEL": "env-model",
        },
        client_factory=FakeOpenAIClient,
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "env-model"
    assert provider.backend_label == "example.test"


def test_openai_provider_can_use_openai_api_key_without_external_call() -> None:
    provider = create_llm_provider(
        {"provider": "auto", "model": "gpt-4o-mini", "temperature": 0.0},
        environ={
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "fake-openai-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
        client_factory=FakeOpenAIClient,
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.backend_label == "api.openai.com"


def test_streamlit_provider_error_message_hides_raw_traceback() -> None:
    message = clean_error_message(
        ProviderConfigurationError("OPENAI_API_KEY is not set."),
        "Application startup failed.",
    )

    assert message == MISSING_PROVIDER_KEY_MESSAGE
    assert "Traceback" not in message
