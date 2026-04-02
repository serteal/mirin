"""Shared request normalization for local and server-backed model APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch


@dataclass(slots=True)
class RequestBatch:
    rows: list[dict[str, torch.Tensor]]
    batch: dict[str, torch.Tensor]


def request_items(value: Any) -> list[Any] | None:
    if _is_message_sequence(value) or isinstance(value, (str, Mapping)):
        return [value]
    if isinstance(value, torch.Tensor):
        return None
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, Mapping, torch.Tensor))
        and not value
    ):
        return []
    if isinstance(value, Sequence) and value and all(_is_request_item(item) for item in value):
        return list(value)
    return None


def normalize_request_row(
    request: Any,
    *,
    tokenizer: Any | None,
    add_generation_prompt: bool,
    owner: str,
) -> dict[str, torch.Tensor]:
    if isinstance(request, str):
        return _encode_text_request(request, tokenizer=tokenizer, owner=owner)
    if isinstance(request, Mapping):
        if "input_ids" in request:
            return _normalize_token_request(request)
        if "text" in request:
            return _encode_text_request(
                str(request["text"]),
                tokenizer=tokenizer,
                owner=owner,
            )
        if "messages" in request:
            return _encode_messages_request(
                request["messages"],
                tokenizer=tokenizer,
                add_generation_prompt=add_generation_prompt,
                owner=owner,
            )
        if _looks_like_message(request):
            return _encode_messages_request(
                [request],
                tokenizer=tokenizer,
                add_generation_prompt=add_generation_prompt,
                owner=owner,
            )
    if _is_message_sequence(request):
        return _encode_messages_request(
            request,
            tokenizer=tokenizer,
            add_generation_prompt=add_generation_prompt,
            owner=owner,
        )
    raise TypeError(
        "Requests must be strings, chat-message lists, or mappings with "
        "`input_ids`, `text`, or `messages`."
    )


def normalize_requests(
    requests: Sequence[Any] | Any,
    *,
    tokenizer: Any | None,
    add_generation_prompt: bool,
    pad_side: str,
    pad_token_id: int,
    owner: str,
) -> RequestBatch:
    items = request_items(requests)
    if items is None:
        raise TypeError("Expected one request or a sequence of requests.")
    if not items:
        raise ValueError("Expected at least one request.")
    rows = [
        normalize_request_row(
            request,
            tokenizer=tokenizer,
            add_generation_prompt=add_generation_prompt,
            owner=owner,
        )
        for request in items
    ]
    devices = {row["input_ids"].device for row in rows}
    if len(devices) != 1:
        raise ValueError("All batched requests must live on the same device.")
    max_len = max(int(row["input_ids"].shape[-1]) for row in rows)
    device = rows[0]["input_ids"].device
    batch_input_ids = torch.full(
        (len(rows), max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    batch_attention = torch.zeros(
        (len(rows), max_len),
        dtype=torch.long,
        device=device,
    )
    for idx, row in enumerate(rows):
        input_ids = row["input_ids"].view(-1)
        attention_mask = row["attention_mask"].view(-1)
        length = int(input_ids.shape[0])
        if pad_side == "left":
            batch_input_ids[idx, max_len - length :] = input_ids
            batch_attention[idx, max_len - length :] = attention_mask
        else:
            batch_input_ids[idx, :length] = input_ids
            batch_attention[idx, :length] = attention_mask
    return RequestBatch(
        rows=rows,
        batch={
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
        },
    )


def merge_request_kwargs(
    row: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    overlap = set(row).intersection(kwargs)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ValueError(f"Duplicate request kwargs: {joined}.")
    return {**row, **kwargs}


def coerce_token_tensor(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor
    raise ValueError(f"{name} must be shape [seq] or [1, seq].")


def _normalize_token_request(request: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    input_ids = coerce_token_tensor(request["input_ids"], name="input_ids")
    attention_value = request.get("attention_mask")
    if attention_value is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = coerce_token_tensor(attention_value, name="attention_mask")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape.")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _encode_text_request(
    text: str,
    *,
    tokenizer: Any | None,
    owner: str,
) -> dict[str, torch.Tensor]:
    tokenizer = _require_tokenizer(tokenizer, owner=owner)
    encoded = tokenizer(text, return_tensors="pt")
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError(f"{owner} tokenizer must return a mapping with input_ids.")
    return _normalize_token_request(cast(Mapping[str, Any], encoded))


def _encode_messages_request(
    messages: Any,
    *,
    tokenizer: Any | None,
    add_generation_prompt: bool,
    owner: str,
) -> dict[str, torch.Tensor]:
    tokenizer = _require_tokenizer(tokenizer, owner=owner)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError(f"{owner} tokenizer does not support chat messages.")
    rendered = apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str):
        raise TypeError(f"{owner} tokenizer.apply_chat_template(...) must return a string.")
    return _encode_text_request(rendered, tokenizer=tokenizer, owner=owner)


def _require_tokenizer(tokenizer: Any | None, *, owner: str) -> Any:
    if tokenizer is None:
        raise TypeError(f"{owner} requires a tokenizer for string or chat-message requests.")
    return tokenizer


def _looks_like_message(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and "role" in value
        and "content" in value
        and "input_ids" not in value
        and "text" not in value
        and "messages" not in value
    )


def _is_message_sequence(value: Any) -> bool:
    return (
        not isinstance(value, (str, Mapping, torch.Tensor))
        and isinstance(value, Sequence)
        and bool(value)
        and all(_looks_like_message(item) for item in value)
    )


def _is_request_item(value: Any) -> bool:
    return isinstance(value, str) or isinstance(value, Mapping) or _is_message_sequence(value)
