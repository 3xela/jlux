import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from jlux.model.pipeline import FluxPipeline
from jlux.model import FluxParams


def run_call(pipe, prompts, height, width, steps, guidance, key, label):
    t0 = time.time()
    img = pipe(prompts, height, width, steps, guidance, key)
    img.block_until_ready()
    elapsed = time.time() - t0
    print(f"  {label}: {elapsed:.2f}s")
    return img, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a photo of a cat")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out.png")
    parser.add_argument("--n_runs", type=int, default=2,
                        help="timed runs after warmup")
    parser.add_argument("--profile_dir", type=str, default=None,
                        help="jax.profiler.trace dir for the final run")
    args = parser.parse_args()

    flux_path = args.flux_path or hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev.safetensors",
    )

    print("loading pipeline...")
    t0 = time.time()
    pipe = FluxPipeline(load_cfg=FluxParams(), flux_path=flux_path)
    print(f"  loaded in {time.time() - t0:.1f}s")

    prompts = [args.prompt]
    key = jax.random.PRNGKey(args.seed)
    common = (prompts, args.height, args.width, args.steps, args.guidance, key)

    print(f"sampling: {args.height}x{args.width}, {args.steps} steps, "
          f"guidance={args.guidance}, B={len(prompts)}")

    # Warmup: first call includes compilation
    img, warmup_t = run_call(pipe, *common, label="warmup (compile + run)")

    # Steady-state runs
    times = []
    for i in range(args.n_runs):
        if args.profile_dir and i == args.n_runs - 1:
            with jax.profiler.trace(args.profile_dir):
                img, t = run_call(pipe, *common, label=f"run {i+1} (profiled)")
        else:
            img, t = run_call(pipe, *common, label=f"run {i+1}")
        times.append(t)

    avg = sum(times) / len(times)
    print(f"\nsteady-state: {avg:.2f}s avg ({len(times)} runs)")
    print(f"compile overhead: {warmup_t - avg:.2f}s")

    print(f"\nshape: {img.shape}, dtype: {img.dtype}")
    print(f"range: [{float(jnp.min(img)):.3f}, {float(jnp.max(img)):.3f}]")
    assert not jnp.any(jnp.isnan(img)), "NaN in output"
    assert not jnp.any(jnp.isinf(img)), "Inf in output"

    img_np = np.asarray(img[0]).transpose(1, 2, 0)
    img_np = np.clip((img_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(args.out)
    print(f"saved to {args.out}")

    if args.profile_dir:
        print(f"profile: {args.profile_dir}")
        print(f"view with: tensorboard --logdir={args.profile_dir}")


if __name__ == "__main__":
    main()