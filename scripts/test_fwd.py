# scripts/test_forward.py
"""
Test forward pass through loaded FLUX.
Verifies the loaded model traces, doesn't NaN, and produces reasonable output.
"""

import time
import jax
import jax.numpy as jnp
import equinox as eqx
from huggingface_hub import hf_hub_download

from jlux import FluxParams
from jlux import load_flux
from jlux.dit.layers.rope import build_position_ids


@eqx.filter_jit
def forward(model, img, img_ids, txt, txt_ids, t, y, guidance):
    return model(img, img_ids, txt, txt_ids, t, y, guidance)


def main():
    # --- Load weights ---
    path = hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev.safetensors",
    )
    cfg = FluxParams()
    print("Loading FLUX...")
    t0 = time.time()
    model = load_flux(cfg, path, dtype=jnp.bfloat16)
    jax.block_until_ready(model.img_in.weight)
    print(f"  → loaded in {time.time() - t0:.1f}s")

    # --- Build dummy inputs (small, to fit comfortably) ---
    # 256x256 effective image → 16x16 patch grid → 256 image tokens.
    # Keeps attention matrices small for first-run testing.
    H_p, W_p = 16, 16
    N = H_p * W_p  # 256 image tokens
    L = 64  # 64 text tokens (T5 typically 256+, smaller for testing)

    key = jax.random.PRNGKey(0)
    k_img, k_txt, k_y = jax.random.split(key, 3)

    img = jax.random.normal(k_img, (N, cfg.in_channels), dtype=jnp.bfloat16)
    txt = jax.random.normal(k_txt, (L, cfg.context_in_dim), dtype=jnp.bfloat16)
    y = jax.random.normal(k_y, (cfg.vec_in_dim,), dtype=jnp.bfloat16)

    all_ids = build_position_ids(s_text=L, H_p=H_p, W_p=W_p)
    txt_ids = all_ids[:L]
    img_ids = all_ids[L:]

    timesteps = jnp.array(0.5, dtype=jnp.bfloat16)
    guidance = jnp.array(3.5, dtype=jnp.bfloat16)

    # --- Compile + first run ---
    print(f"\nFirst forward pass (compiling, {L + N} tokens, ~57 blocks)...")
    t0 = time.time()
    out = forward(model, img, img_ids, txt, txt_ids, timesteps, y, guidance)
    out = jax.block_until_ready(out)
    print(f"  → compile + run: {time.time() - t0:.1f}s")

    # --- Second run (warm, just execution) ---
    print("\nSecond forward pass (cached)...")
    t0 = time.time()
    out2 = forward(model, img, img_ids, txt, txt_ids, timesteps, y, guidance)
    out2 = jax.block_until_ready(out2)
    print(f"  → execution: {time.time() - t0:.2f}s")

    # --- Validate output ---
    print("\nOutput diagnostics:")
    print(f"  shape: {out.shape}  (expected ({N}, {cfg.in_channels}))")
    print(f"  dtype: {out.dtype}")

    out_fp32 = out.astype(jnp.float32)
    has_nan = bool(jnp.any(jnp.isnan(out_fp32)))
    has_inf = bool(jnp.any(jnp.isinf(out_fp32)))
    mean = float(jnp.mean(out_fp32))
    std = float(jnp.std(out_fp32))
    abs_max = float(jnp.max(jnp.abs(out_fp32)))

    print(f"  NaN: {has_nan}")
    print(f"  Inf: {has_inf}")
    print(f"  mean: {mean:+.4e}")
    print(f"  std:  {std:.4e}")
    print(f"  max |x|: {abs_max:.4e}")

    # --- Verdict ---
    print()
    if has_nan or has_inf:
        print("✗ Output contains NaN or Inf — something is wrong.")
    elif abs_max > 1e3:
        print("⚠ Output magnitudes are large — possibly exploding.")
    elif abs_max < 1e-6:
        print("⚠ Output magnitudes are tiny — possibly all-zero.")
    else:
        print("✓ Forward pass completed cleanly with reasonable output.")


if __name__ == "__main__":
    main()
