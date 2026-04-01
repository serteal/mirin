"""Unit tests for request normalization helpers."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from mirin.requests import (
    merge_request_kwargs,
    normalize_request_row,
    normalize_requests,
    request_items,
)

from .helpers import FakeTokenizer, request_contract_cases


class _TokenizerWithoutInputIds:
    def __call__(self, _text: str, *, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"attention_mask": torch.ones((1, 2), dtype=torch.long)}


class _TokenizerWithBadTemplate(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> None:
        del messages, tokenize, add_generation_prompt
        return None


def test_request_items_accept_contract_request_forms() -> None:
    for _name, request, _expected in request_contract_cases(add_generation_prompt=False):
        items = request_items(request)
        assert items is not None
        assert len(items) == 1


def test_normalize_request_row_matches_contract_cases() -> None:
    tokenizer = FakeTokenizer()
    for _name, request, expected in request_contract_cases(add_generation_prompt=False):
        actual = normalize_request_row(
            request,
            tokenizer=tokenizer,
            add_generation_prompt=False,
            owner="Model",
        )
        assert torch.equal(actual["input_ids"], expected["input_ids"])
        assert torch.equal(actual["attention_mask"], expected["attention_mask"])


def test_normalize_requests_batches_variable_lengths() -> None:
    batch = normalize_requests(
        ["hi", "hello"],
        tokenizer=FakeTokenizer(),
        add_generation_prompt=False,
        pad_side="right",
        pad_token_id=0,
        owner="Model",
    )

    assert len(batch.rows) == 2
    assert batch.batch["input_ids"].shape == batch.batch["attention_mask"].shape
    assert batch.batch["input_ids"].shape[0] == 2


def test_merge_request_kwargs_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicate request kwargs"):
        merge_request_kwargs({"input_ids": torch.ones((1, 2), dtype=torch.long)}, {"input_ids": 1})


def test_role_only_mapping_is_rejected_as_a_request() -> None:
    with pytest.raises(TypeError, match="Requests must be strings"):
        normalize_request_row(
            {"role": "user"},
            tokenizer=FakeTokenizer(),
            add_generation_prompt=False,
            owner="Model",
        )


def test_text_request_requires_input_ids_from_tokenizer() -> None:
    with pytest.raises(TypeError, match="input_ids"):
        normalize_request_row(
            "hello",
            tokenizer=_TokenizerWithoutInputIds(),
            add_generation_prompt=False,
            owner="Model",
        )


def test_message_request_requires_string_chat_template() -> None:
    with pytest.raises(TypeError, match="must return a string"):
        normalize_request_row(
            {"messages": [{"role": "user", "content": "hello"}]},
            tokenizer=_TokenizerWithBadTemplate(),
            add_generation_prompt=False,
            owner="Model",
        )
