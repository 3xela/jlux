# scripts/test_loader.py
"""
Test the FLUX safetensors loader end-to-end.
Loads weights from the HuggingFace cache, prints diagnostics.
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
from huggingface_hub import hf_hub_download

from jlux import Flux, FluxParams
from jlux import load_flux  # adjust import to wherever you put it


def main():
    # Resolve cached safetensors path (no download if already cached).
    print("Locating flux1-dev.safetensors in HF cache...")
    path = hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev.safetensors",
    )
    print(f"  → {path}")

    # Load.
    cfg = FluxParams()
    print(f"\nLoading FLUX with cfg.hidden_size={cfg.hidden_size}, "
          f"depth={cfg.depth}, depth_single={cfg.depth_single_blocks}...")
    t0 = time.time()
    model = load_flux(cfg, path, dtype=jnp.bfloat16)
    jax.block_until_ready(model.img_in.weight)  # force materialization
    t1 = time.time()
    print(f"  → loaded in {t1 - t0:.1f}s")

    # Pull out only the array leaves for stats.
    arrays = eqx.filter(model, eqx.is_array)
    leaves, _ = jax.tree_util.tree_flatten(arrays)
    total_params = sum(int(jnp.prod(jnp.array(l.shape))) for l in leaves)
    print(f"\nLeaf count: {len(leaves)} (expected 780)")
    print(f"Total params: {total_params / 1e9:.2f}B (expected ~11.9B)")

    # Spot-check a few specific weights.
    print("\nSpot checks:")
    checks = [
        ("img_in.weight", model.img_in.weight, (3072, 64)),
        ("txt_in.weight", model.txt_in.weight, (3072, 4096)),
        ("double_blocks[0].img_attn.qkv.weight",
         model.double_blocks[0].img_attn.qkv.weight, (9216, 3072)),
        ("single_blocks[0].linear1.weight",
         model.single_blocks[0].linear1.weight, (21504, 3072)),
        ("final_layer.linear.weight", model.final_layer.linear.weight, (64, 3072)),
    ]
    for name, tensor, expected_shape in checks:
        ok = tensor.shape == expected_shape
        dtype_ok = tensor.dtype == jnp.bfloat16
        nan = bool(jnp.any(jnp.isnan(tensor)))
        print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
              f"nan={nan}, ok={ok and dtype_ok and not nan}")

    # Cheap statistical sanity: weights should be small, centered, no NaNs.
    print("\nGlobal weight stats:")
    sample = model.img_in.weight.astype(jnp.float32)
    print(f"  img_in.weight  mean={float(jnp.mean(sample)):+.4e}  "
          f"std={float(jnp.std(sample)):.4e}")

    sample = model.double_blocks[0].img_attn.qkv.weight.astype(jnp.float32)
    print(f"  block0.qkv     mean={float(jnp.mean(sample)):+.4e}  "
          f"std={float(jnp.std(sample)):.4e}")

    print("\n✓ Loader test complete.")


if __name__ == "__main__":
    main()