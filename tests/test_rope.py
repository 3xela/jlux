import jax
import jax.numpy as jnp

from jlux.dit.layers.rope import (
    RoPE,
    build_position_ids,
)  # adjust import path to your file structure


def test_identity_at_zero():
    """RoPE with all-zero positions should be identity."""
    H, T, head_dim = 4, 10, 128
    rope = RoPE()

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (H, T, head_dim))
    position_ids = jnp.zeros((T, 3), dtype=jnp.int32)

    out = rope(x, position_ids)

    max_diff = jnp.max(jnp.abs(out - x))
    print(f"[identity_at_zero] max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Expected identity, got max diff {max_diff}"
    print("  PASSED")


def test_magnitude_preservation():
    """RoPE should preserve per-token magnitudes (rotations are isometries)."""
    H, T, head_dim = 4, 10, 128
    rope = RoPE()

    key = jax.random.PRNGKey(1)
    key_x, key_pos = jax.random.split(key)
    x = jax.random.normal(key_x, (H, T, head_dim))

    # Use varied positions: not all zero, so rotation is non-trivial
    position_ids = jax.random.randint(key_pos, (T, 3), 0, 50)

    out = rope(x, position_ids)

    # Per-token magnitudes (sum of squares along head_dim)
    mag_in = jnp.sum(x**2, axis=-1)  # shape (H, T)
    mag_out = jnp.sum(out**2, axis=-1)

    max_diff = jnp.max(jnp.abs(mag_in - mag_out))
    print(f"[magnitude_preservation] max diff: {max_diff:.2e}")
    assert max_diff < 1e-3, f"Expected magnitudes preserved, max diff {max_diff}"
    print("  PASSED")


def test_per_axis_independence():
    """Setting only axis 1 nonzero should leave axis-0 and axis-2 slices unchanged."""
    H, T, head_dim = 4, 10, 128
    rope = RoPE()

    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (H, T, head_dim))

    # Only axis 1 has non-zero positions
    position_ids = jnp.zeros((T, 3), dtype=jnp.int32)
    position_ids = position_ids.at[:, 1].set(jnp.arange(T))

    out = rope(x, position_ids)

    # First 16 dims (axis 0) should be unchanged
    diff_axis0 = jnp.max(jnp.abs(out[..., :16] - x[..., :16]))
    # Middle 56 dims (axis 1) should change
    diff_axis1 = jnp.max(jnp.abs(out[..., 16:72] - x[..., 16:72]))
    # Last 56 dims (axis 2) should be unchanged
    diff_axis2 = jnp.max(jnp.abs(out[..., 72:] - x[..., 72:]))

    print("[per_axis_independence]")
    print(f"  axis-0 slice diff (should be ~0): {diff_axis0:.2e}")
    print(f"  axis-1 slice diff (should be >0): {diff_axis1:.2e}")
    print(f"  axis-2 slice diff (should be ~0): {diff_axis2:.2e}")

    assert diff_axis0 < 1e-5, f"axis-0 slice should be unchanged, diff {diff_axis0}"
    assert diff_axis1 > 1e-2, f"axis-1 slice should change, diff {diff_axis1}"
    assert diff_axis2 < 1e-5, f"axis-2 slice should be unchanged, diff {diff_axis2}"
    print("  PASSED")


def test_build_position_ids():
    """Sanity check the position ID structure."""
    s_text, H_p, W_p = 3, 2, 3
    pos = build_position_ids(s_text, H_p, W_p)

    print("[build_position_ids]")
    print(f"  shape: {pos.shape} (expected ({s_text + H_p * W_p}, 3))")
    print("  full output:")
    print(pos)

    assert pos.shape == (s_text + H_p * W_p, 3)

    # Text rows all zero
    assert jnp.all(pos[:s_text] == 0), "Text positions should be all zero"

    # Image rows: row-major (h, w) with leading zero
    expected_image = jnp.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
        ]
    )
    assert jnp.all(pos[s_text:] == expected_image), "Image positions wrong"
    print("  PASSED")


def test_shape_preservation():
    """Output shape should equal input shape."""
    H, T, head_dim = 4, 10, 128
    rope = RoPE()

    key = jax.random.PRNGKey(3)
    x = jax.random.normal(key, (H, T, head_dim))
    position_ids = jnp.zeros((T, 3), dtype=jnp.int32)

    out = rope(x, position_ids)

    print(f"[shape_preservation] in: {x.shape}, out: {out.shape}")
    assert out.shape == x.shape
    print("  PASSED")


if __name__ == "__main__":
    print("Testing RoPE implementation\n")

    test_build_position_ids()
    print()
    test_shape_preservation()
    print()
    test_identity_at_zero()
    print()
    test_magnitude_preservation()
    print()
    test_per_axis_independence()
    print()

    print("All tests passed.")
