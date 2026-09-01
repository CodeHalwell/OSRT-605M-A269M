"""Fail-closed validation for the tokenizer baked into model checkpoints."""

from __future__ import annotations

from typing import Protocol


class _Tokenizer(Protocol):
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None
    unk_token_id: int | None

    def __len__(self) -> int: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


V6_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "<|padding|>": 0,
    "<|begin_of_text|>": 1,
    "<|end_of_text|>": 2,
    "<|unknown|>": 3,
    "<|fim_prefix|>": 4,
    "<|fim_middle|>": 5,
    "<|fim_suffix|>": 6,
    "<|think|>": 7,
    "<|/think|>": 8,
    "<|answer|>": 9,
    "<|/answer|>": 10,
    "<|user|>": 11,
    "<|assistant|>": 12,
    "<|system|>": 13,
    "<|end_turn|>": 14,
    "<|tool_call|>": 15,
    "<|/tool_call|>": 16,
    "<|tool_result|>": 17,
    "<|/tool_result|>": 18,
    "<|image|>": 19,
    "<|audio|>": 20,
    **{f"<|reserved_{token_id}|>": token_id for token_id in range(21, 32)},
}

V6_ROLE_TOKEN_IDS = {
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "unk_token_id": 3,
}


def validate_tokenizer_contract(
    tokenizer: _Tokenizer,
    *,
    expected_vocab_size: int = 65_536,
) -> None:
    """Raise when vocab size or any structural-token ID has drifted."""
    errors: list[str] = []
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != expected_vocab_size:
        errors.append(
            f"vocab size is {actual_vocab_size}, expected {expected_vocab_size}"
        )

    for token, expected_id in V6_SPECIAL_TOKEN_IDS.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            errors.append(f"{token} has id {actual_id}, expected {expected_id}")

    for attribute, expected_id in V6_ROLE_TOKEN_IDS.items():
        actual_id = getattr(tokenizer, attribute, None)
        if actual_id != expected_id:
            errors.append(f"{attribute} is {actual_id}, expected {expected_id}")

    if errors:
        details = "; ".join(errors)
        raise ValueError(
            "Tokenizer contract mismatch; refusing to construct a model with "
            f"incompatible embeddings: {details}"
        )
