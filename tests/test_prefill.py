from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import Mock, patch

from benchmarks.prefill import (
    build_candidate_document,
    calibrate_prompt,
    prepare_exact_prompt,
)
from core_runner import ModelClient


@dataclass
class _Tokenization:
    count: int
    tokens: list[int]


class _FakeTokenizerClient:
    """One rendered template token plus one token per four characters."""

    def tokenize_prompt(self, prompt: str) -> _Tokenization:
        count = 1 + len(prompt) // 4
        return _Tokenization(count=count, tokens=list(range(count)))


class PromptCalibrationTests(unittest.TestCase):
    def test_candidate_document_meets_requested_character_length(self) -> None:
        document = build_candidate_document(10_000)
        self.assertGreaterEqual(len(document), 10_000)
        self.assertIn("Field record 1", document)
        self.assertIn("Field record 6", document)

    def test_calibration_counts_the_rendered_prompt_exactly(self) -> None:
        client = _FakeTokenizerClient()
        result = calibrate_prompt(client, "a" * 20_000, 512)
        self.assertTrue(result.exact)
        self.assertEqual(result.requested_tokens, 512)
        self.assertEqual(client.tokenize_prompt(result.text).count, 512)

    def test_prepare_grows_and_calibrates_candidate(self) -> None:
        result = prepare_exact_prompt(
            _FakeTokenizerClient(),
            2_048,
            initial_chars_per_token=1,
        )
        self.assertTrue(result.exact)
        self.assertEqual(result.actual_tokens, 2_048)

    def test_rejects_nonpositive_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_tokens must be positive"):
            calibrate_prompt(_FakeTokenizerClient(), "long enough", 0)


class ModelClientTokenizeTests(unittest.TestCase):
    @patch("core_runner.requests.post")
    def test_chat_tokenization_uses_rendered_messages(self, post: Mock) -> None:
        response = post.return_value
        response.json.return_value = {"count": 3, "tokens": [10, 11, 12]}

        client = ModelClient("http://localhost:8000", "example/model", chat=True)
        result = client.tokenize_prompt("hello")

        self.assertEqual(result.count, 3)
        post.assert_called_once_with(
            "http://localhost:8000/tokenize",
            headers={"Content-Type": "application/json"},
            json={
                "model": "example/model",
                "messages": [{"role": "user", "content": "hello"}],
                "add_generation_prompt": True,
            },
            timeout=300,
        )
        response.raise_for_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
