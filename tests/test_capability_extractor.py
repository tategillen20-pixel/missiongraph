"""Mocked tests for structured OpenAI capability extraction."""

from unittest.mock import Mock

import pytest

from ai.capability_extractor import (
    CapabilityExtraction,
    CapabilityExtractionError,
    ExtractedCapability,
    extract_capabilities,
)


def _client_with(capabilities: list[ExtractedCapability]) -> Mock:
    """Return a mocked client with one parsed Responses API result."""
    client = Mock()
    client.responses.parse.return_value = Mock(
        output_parsed=CapabilityExtraction(capabilities=capabilities)
    )
    return client


def test_uses_responses_parse_and_keeps_exact_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported output is parsed with Pydantic and keeps direct evidence."""
    monkeypatch.setenv("OPENAI_MODEL", "test-structured-model")
    description = "The contractor shall provide zero trust architecture."
    client = _client_with(
        [
            ExtractedCapability(
                name="Zero Trust Architecture",
                category="cybersecurity",
                evidence_quote="provide zero trust architecture",
                confidence=0.91,
            )
        ]
    )

    extraction = extract_capabilities(
        "Cybersecurity requirement",
        description,
        "NOTICE-123",
        client=client,
    )

    assert extraction["source_record_id"] == "NOTICE-123"
    assert extraction["evidence_type"] == "ai_extracted"
    assert extraction["model"] == "test-structured-model"
    assert extraction["capabilities"][0]["evidence_quote"] == (
        "provide zero trust architecture"
    )
    call = client.responses.parse.call_args
    assert call.kwargs["model"] == "test-structured-model"
    assert call.kwargs["text_format"] is CapabilityExtraction


def test_discards_quote_that_is_not_an_exact_substring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Case or wording differences cause unsupported output to be discarded."""
    monkeypatch.setenv("OPENAI_MODEL", "test-structured-model")
    client = _client_with(
        [
            ExtractedCapability(
                name="Zero Trust Architecture",
                category="cybersecurity",
                evidence_quote="Provide Zero Trust Architecture",
                confidence=0.99,
            )
        ]
    )

    extraction = extract_capabilities(
        "Cybersecurity requirement",
        "The contractor shall provide zero trust architecture.",
        "NOTICE-123",
        client=client,
    )

    assert extraction["capabilities"] == []


def test_permits_empty_capability_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid structured response may contain no capabilities."""
    monkeypatch.setenv("OPENAI_MODEL", "test-structured-model")
    client = _client_with([])

    extraction = extract_capabilities(
        "Office supplies",
        "Deliver paper clips to the listed address.",
        "NOTICE-123",
        client=client,
    )

    assert extraction["capabilities"] == []


def test_empty_description_does_not_call_openai() -> None:
    """No source text produces an empty result without spending API tokens."""
    client = Mock()

    extraction = extract_capabilities(
        "Missing description",
        "",
        "NOTICE-123",
        client=client,
    )

    assert extraction["capabilities"] == []
    assert extraction["model"] is None
    client.responses.parse.assert_not_called()


def test_requires_model_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No undocumented or invented fallback model is used."""
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(CapabilityExtractionError, match="OPENAI_MODEL"):
        extract_capabilities(
            "Requirement",
            "The contractor shall provide software engineering.",
            "NOTICE-123",
            client=Mock(),
        )


def test_requires_api_key_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production client construction reads the API key from the environment."""
    monkeypatch.setenv("OPENAI_MODEL", "test-structured-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(CapabilityExtractionError, match="OPENAI_API_KEY"):
        extract_capabilities(
            "Requirement",
            "The contractor shall provide software engineering.",
            "NOTICE-123",
        )
