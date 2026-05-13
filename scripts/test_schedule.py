# scripts/test_sampler_realistic.py
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from huggingface_hub import hf_hub_download

from jlux import FluxParams
from jlux import load_flux
from jlux.dit.layers.rope import build_position_ids
from jlux.sampler import euler_sample, build_schedule


def main():
    # Realistic FLUX dev config
    H_p, W_p = 64, 64  # 1024×1024 → 4096 image tokens
    N = H_p * W_p
    L = 256  # T5 sequence length
    num_steps = 28  # canonical FLUX dev step count

    # Load
    path = hf_hub_download("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors")
    cfg = FluxParams()
    print("Loading FLUX...")
    model = load_flux(cfg, path, dtype=jnp.bfloat16)
    jax.block_until_ready(model.img_in.weight)

    # Inputs
    key = jax.random.PRNGKey(0)
    k_x, k_txt, k_y = jax.random.split(key, 3)
    x_init = jax.random.normal(k_x, (N, cfg.in_channels), dtype=jnp.bfloat16)
    txt = jax.random.normal(k_txt, (L, cfg.context_in_dim), dtype=jnp.bfloat16)
    y = jax.random.normal(k_y, (cfg.vec_in_dim,), dtype=jnp.bfloat16)

    all_ids = build_position_ids(s_text=L, H_p=H_p, W_p=W_p)
    txt_ids, img_ids = all_ids[:L], all_ids[L:]
    guidance = jnp.array(3.5, dtype=jnp.bfloat16)
    timesteps = build_schedule(num_steps=num_steps, N=N)

    # Jit the model call
    jitted_model = eqx.filter_jit(model)

    # Warmup: compile by running a single step
    print(f"\nCompiling at {N} img tokens + {L} txt tokens...")
    t0 = time.time()
    _ = jitted_model(x_init, img_ids, txt, txt_ids, timesteps[0], y, guidance)
    jax.block_until_ready(_)
    print(f"  compile + first step: {time.time() - t0:.1f}s")

    # Real sampling run
    print(f"\nSampling {num_steps} steps...")
    t0 = time.time()
    out = euler_sample(
        jitted_model, x_init, img_ids, txt, txt_ids, y, guidance, timesteps
    )
    jax.block_until_ready(out)
    elapsed = time.time() - t0
    print(f"  total: {elapsed:.2f}s  ({elapsed / num_steps * 1000:.0f}ms/step)")

    # Sanity
    out32 = out.astype(jnp.float32)
    print(
        f"\nshape: {out.shape}, NaN: {bool(jnp.any(jnp.isnan(out32)))}, "
        f"std: {float(jnp.std(out32)):.3f}"
    )


if __name__ == "__main__":
    main()
