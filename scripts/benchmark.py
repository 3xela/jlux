import argparse
import platform
import time

import jax
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from jlux.model import FluxParams
from jlux.model.pipeline import FluxPipeline


def print_env():
    print("=" * 60)
    print("jlux benchmark")
    print("=" * 60)
    devices = jax.devices()
    print(f"jax:    {jax.__version__}")
    print(f"device: {devices[0].device_kind if devices else 'unknown'}")
    print(f"python: {platform.python_version()}")
    print()


def fmt_stats(times, label):
    mean = sum(times) / len(times)
    print(f"  {label:7s} mean={mean:6.2f}s  min={min(times):6.2f}s  max={max(times):6.2f}s")
    return mean


def main():
    parser = argparse.ArgumentParser(description="Benchmark jlux end-to-end.")
    parser.add_argument("--prompt", type=str, default="a photo of a cat")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=3, help="timed runs after warmup")
    parser.add_argument(
        "--save", type=str, default=None, help="optional output path for the generated image"
    )
    parser.add_argument("--flux_path", type=str, default=None)
    args = parser.parse_args()

    print_env()

    flux_path = args.flux_path or hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev.safetensors",
    )

    print("loading pipeline...")
    t0 = time.perf_counter()
    pipe = FluxPipeline(load_cfg=FluxParams(), flux_path=flux_path)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s\n")

    prompts = [args.prompt]
    key = jax.random.PRNGKey(args.seed)
    B = len(prompts)

    print(
        f"config: {args.height}x{args.width}, {args.steps} steps, "
        f"guidance={args.guidance}, B={B}, n_runs={args.n_runs}"
    )
    print(f"prompt: {args.prompt!r}\n")

    # ---- Warmup (absorbs JAX compile) ----
    print("warmup (includes one-time compile)...")
    t0 = time.perf_counter()
    pooled, seq = pipe.encode(prompts)
    pooled.block_until_ready()
    seq.block_until_ready()
    latent = pipe.sample(pooled, seq, B, args.height, args.width, args.steps, args.guidance, key)
    latent.block_until_ready()
    img = pipe.decode(latent, args.height, args.width)
    warmup_t = time.perf_counter() - t0
    print(f"  done in {warmup_t:.1f}s\n")

    # ---- Timed runs ----
    encode_t, sample_t, decode_t = [], [], []
    for i in range(args.n_runs):
        t0 = time.perf_counter()
        pooled, seq = pipe.encode(prompts)
        pooled.block_until_ready()
        seq.block_until_ready()
        encode_t.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        latent = pipe.sample(
            pooled, seq, B, args.height, args.width, args.steps, args.guidance, key
        )
        latent.block_until_ready()
        sample_t.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        img = pipe.decode(latent, args.height, args.width)
        decode_t.append(time.perf_counter() - t0)

    print(f"steady-state timings (n={args.n_runs}):")
    e = fmt_stats(encode_t, "encode")
    s = fmt_stats(sample_t, "sample")
    d = fmt_stats(decode_t, "decode")
    total = e + s + d
    print(f"  {'total':7s} {total:6.2f}s")
    print(f"  {'compile':7s} {warmup_t - total:6.2f}s  (warmup - steady-state)")
    print()

    if args.save:
        img_np = np.asarray(img[0]).transpose(1, 2, 0)
        img_np = np.clip((img_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(args.save)
        print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
