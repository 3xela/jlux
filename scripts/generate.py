import argparse

import jax
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from jlux.model import FluxParams
from jlux.model.pipeline import FluxPipeline


def main():
    parser = argparse.ArgumentParser(description="Generate an image with jlux.")
    parser.add_argument("--prompt", type=str, default="a photo of a cat")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="out.png")
    parser.add_argument(
        "--flux_path",
        type=str,
        default=None,
        help="local safetensors path; if unset, downloads from HF",
    )
    args = parser.parse_args()

    flux_path = args.flux_path or hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev.safetensors",
    )

    print("loading pipeline...")
    pipe = FluxPipeline(load_cfg=FluxParams(), flux_path=flux_path)

    key = jax.random.PRNGKey(args.seed)
    print(f"generating: {args.height}x{args.width}, {args.steps} steps, guidance={args.guidance}")
    print(f"prompt: {args.prompt!r}")

    img = pipe([args.prompt], args.height, args.width, args.steps, args.guidance, key)
    img.block_until_ready()

    img_np = np.asarray(img[0]).transpose(1, 2, 0)
    img_np = np.clip((img_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
