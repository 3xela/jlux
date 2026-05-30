"""Smoke test for FluxDoubleStreamBlock at FLUX shapes."""

import jax
import jax.numpy as jnp

from jlux.dit import FluxDoubleStreamBlock
from jlux.dit.layers import build_position_ids

DIM = 3072
NUM_HEADS = 24
S_TEXT = 4
H_P, W_P = 8, 8
S_IMAGE = H_P * W_P


def test_double_block_forward_shapes():
    key = jax.random.PRNGKey(0)
    block = FluxDoubleStreamBlock(dim=DIM, num_heads=NUM_HEADS, key=key)

    img = jnp.ones((S_IMAGE, DIM))
    txt = jnp.ones((S_TEXT, DIM))
    temb = jnp.ones((DIM,))
    pos_ids = build_position_ids(S_TEXT, H_P, W_P)

    img_out, txt_out = block(img, txt, temb, pos_ids)

    assert img_out.shape == img.shape, f"img shape mismatch: {img_out.shape} vs {img.shape}"
    assert txt_out.shape == txt.shape, f"txt shape mismatch: {txt_out.shape} vs {txt.shape}"


def test_double_block_finite():
    """Forward pass produces finite values (no NaN/Inf)."""
    key = jax.random.PRNGKey(42)
    block = FluxDoubleStreamBlock(dim=DIM, num_heads=NUM_HEADS, key=key)

    img = jax.random.normal(jax.random.PRNGKey(1), (S_IMAGE, DIM))
    txt = jax.random.normal(jax.random.PRNGKey(2), (S_TEXT, DIM))
    temb = jax.random.normal(jax.random.PRNGKey(3), (DIM,))
    pos_ids = build_position_ids(S_TEXT, H_P, W_P)

    img_out, txt_out = block(img, txt, temb, pos_ids)

    assert jnp.all(jnp.isfinite(img_out)), "img_out contains NaN or Inf"
    assert jnp.all(jnp.isfinite(txt_out)), "txt_out contains NaN or Inf"


if __name__ == "__main__":
    test_double_block_forward_shapes()
    print("test_double_block_forward_shapes: ok")
    test_double_block_finite()
    print("test_double_block_finite: ok")
