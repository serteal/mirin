"""Prebuilt rename packs for cross-model navigation."""

from __future__ import annotations

llm: dict[str, str] = {
    "transformer": "model",
    "gpt_neox": "model",
    "decoder": "model",
    "language_model": "model",
    "h": "layers",
    "blocks": "layers",
    "attn": "self_attn",
    "self_attention": "self_attn",
    "attention": "self_attn",
    "norm_attn_norm": "self_attn",
    "block_sparse_moe": "mlp",
    "ffn": "mlp",
    "ln_f": "ln_final",
    "norm_f": "ln_final",
    "final_layer_norm": "ln_final",
    "norm": "ln_final",
    "embed_out": "lm_head",
    "wte": "embed_tokens",
    "embed_in": "embed_tokens",
    "word_embeddings": "embed_tokens",
}

vision: dict[str, str] = {
    "patch_embed": "embed",
    "patch_embedding": "embed",
    "blocks": "layers",
    "head": "classifier",
    "heads": "classifier",
    "norm": "ln_final",
}

__all__ = ["llm", "vision"]
