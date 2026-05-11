"""Smoke test for FluxSingleStreamBlock at FLUX shapes."""

import jax
import jax.numpy as jnp

from jlux.dit import FluxSingleStreamBlock
from jlux.dit.layers import build_position_ids


DIM = 3072
NUM_HEADS = 24
S_TEXT = 4
H_P, W_P = 8, 8
S_IMAGE = H_P * W_P
S_JOINT = S_TEXT + S_IMAGE


def test_single_block_forward_shapes():
    key = jax.random.PRNGKey(0)
    block = FluxSingleStreamBlock(dim=DIM, num_heads=NUM_HEADS, key=key)

    x = jnp.ones((S_JOINT, DIM))
    temb = jnp.ones((DIM,))
    pos_ids = build_position_ids(S_TEXT, H_P, W_P)

    out = block(x, temb, pos_ids)

    assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"


def test_single_block_finite():
    """Forward pass produces finite values (no NaN/Inf)."""
    key = jax.random.PRNGKey(42)
    block = FluxSingleStreamBlock(dim=DIM, num_heads=NUM_HEADS, key=key)

    x = jax.random.normal(jax.random.PRNGKey(1), (S_JOINT, DIM))
    temb = jax.random.normal(jax.random.PRNGKey(2), (DIM,))
    pos_ids = build_position_ids(S_TEXT, H_P, W_P)

    out = block(x, temb, pos_ids)

    assert jnp.all(jnp.isfinite(out)), "out contains NaN or Inf"


if __name__ == "__main__":
    test_single_block_forward_shapes()
    print("test_single_block_forward_shapes: ok")
    test_single_block_finite()
    print("test_single_block_finite: ok")