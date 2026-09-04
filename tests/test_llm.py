import pytest

from autoresearch.services.llm import LLMError, extract_json_object


def test_extract_json_from_fence() -> None:
    assert extract_json_object('answer\n```json\n{"ok": true}\n```') == {"ok": True}


def test_extract_json_rejects_array() -> None:
    with pytest.raises(LLMError):
        extract_json_object("[1, 2]")

