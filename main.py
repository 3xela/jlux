# scripts/smoke_test_flux.py
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp

from jlux.dit.layers.rope import build_position_ids
from jlux.model.flux import Flux, FluxParams


@eqx.filter_jit
def forward(model, img, img_ids, txt, txt_ids, t, y, guidance):
    return model(img, img_ids, txt, txt_ids, t, y, guidance)


def main():
    # Tiny config for fast iteration.
    # hidden_size=128, num_heads=8 → head_dim=16. RoPE is hardcoded to axes
    # (16, 56, 56) summing to 128, so head_dim must be 128 for now.
    # Keeping hidden_size=128, num_heads=1 → head_dim=128 satisfies RoPE.
    base_cfg = FluxParams()
    cfg = replace(
        base_cfg,
        hidden_size=128,
        num_heads=1,  # head_dim = 128, matches RoPE axes_dim sum
        depth=2,  # 2 double blocks
        depth_single_blocks=2,  # 2 single blocks
    )

    key = jax.random.PRNGKey(0)
    model_key, img_key, txt_key, y_key = jax.random.split(key, 4)

    model = Flux(cfg=FluxParams(), key=model_key)

    # Dummy inputs.
    H_p, W_p = 8, 8  # 8x8 patch grid → N = 64 image tokens
    L = 16  # 16 text tokens
    N = H_p * W_p

    img = jax.random.normal(img_key, (N, cfg.in_channels))
    txt = jax.random.normal(txt_key, (L, cfg.context_in_dim))
    y = jax.random.normal(y_key, (cfg.vec_in_dim,))

    # Position ids: build_position_ids returns (L+N, 3) with txt-then-img.
    # Slice it back into separate (L,3) and (N,3) since Flux.__call__
    # expects them separately and concats internally.
    all_ids = build_position_ids(s_text=L, H_p=H_p, W_p=W_p)
    txt_ids = all_ids[:L]
    img_ids = all_ids[L:]

    timesteps = jnp.array(0.5)
    guidance = jnp.array(3.5)

    print("Running forward pass...")
    out = forward(model, img, img_ids, txt, txt_ids, timesteps, y, guidance)

    arrays_only = eqx.filter(model, eqx.is_array)
    leaves, _ = jax.tree_util.tree_flatten(arrays_only)
    print(len(leaves))

    expected_shape = (N, cfg.in_channels)  # patch_size=1 → out is (N, in_channels)
    print(f"Output shape:   {out.shape}")
    print(f"Expected shape: {expected_shape}")
    assert out.shape == expected_shape, "shape mismatch"

    # Sanity: no NaNs.
    assert not jnp.any(jnp.isnan(out)), "output contains NaNs"
    print("Forward pass OK, no NaNs.")


if __name__ == "__main__":
    main()
