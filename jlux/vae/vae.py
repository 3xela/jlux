import jax.numpy as jnp
import numpy as np
import torch
from diffusers import AutoencoderKL


class VAEWrapper:
    def __init__(self, path="black-forest-labs/FLUX.1-dev"):
        self._path = path
        self.device = "cuda"
        self._vae = (
            AutoencoderKL.from_pretrained(self._path, subfolder="vae").to(self.device).eval()
        )
        self.scale = self._vae.config.scaling_factor
        self.shift = self._vae.config.shift_factor

    def encode(self, img: jnp.ndarray, sample: bool = False):
        img_t = torch.from_numpy(np.asarray(img).copy()).to(self.device)
        with torch.no_grad():
            posterior = self._vae.encode(img_t).latent_dist
            z = posterior.sample() if sample else posterior.mode()
            z = (z - self.shift) * self.scale
            return jnp.asarray(z.float().detach().cpu().numpy())

    def decode(self, latent: jnp.ndarray):
        latent = latent.astype(jnp.float32)
        z = torch.from_numpy(np.asarray(latent)).to(self.device)

        z = z / self.scale + self.shift
        with torch.no_grad():
            img = self._vae.decode(z).sample
            return jnp.asarray(img.float().detach().cpu().numpy())


def main():
    vae = VAEWrapper()

    B, H, W = 2, 256, 256
    img = jnp.asarray(np.random.uniform(-1, 1, size=(B, 3, H, W)).astype(np.float32))
    print(f"input  shape: {img.shape}")

    latent = vae.encode(img)
    print(f"latent shape: {latent.shape}")
    assert latent.shape == (B, 16, H // 8, W // 8)

    recon = vae.decode(latent)
    print(f"recon  shape: {recon.shape}")
    assert recon.shape == (B, 3, H, W)

    print("round-trip OK")


if __name__ == "__main__":
    main()
