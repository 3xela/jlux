import einops
import jax
import jax.numpy as jnp

from ..dit import build_position_ids
from ..model import FluxParams, load_flux
from ..sampler import build_schedule, euler_sample
from ..text import CLIPWrapper, T5Wrapper
from ..vae import VAEWrapper


class FluxPipeline:
    def __init__(self, load_cfg: FluxParams, flux_path: "str"):
        self.clip = CLIPWrapper()
        self.t5 = T5Wrapper()
        self.vae = VAEWrapper()
        self.dit = load_flux(cfg=load_cfg, path=flux_path)

        self.batched_dit = jax.vmap(self.dit, in_axes=(0, None, 0, None, None, 0, None))

    def encode(self, prompts: list[str]):
        pooled = self.clip(prompts)
        seq = self.t5(prompts)
        self.clip.unload()
        self.t5.unload()

        return pooled, seq

    def sample(self, pooled, seq, B, height, width, num_steps, guidance, key):
        assert height % 16 == 0 and width % 16 == 0, "image dims must be multiples of 16"
        H_p, W_p = height // 16, width // 16
        guidance = jnp.asarray(guidance, dtype=jnp.bfloat16)
        latent = jax.random.normal(key, (B, 16, height // 8, width // 8)).astype(jnp.bfloat16)
        latent = einops.rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        txt_ids, img_ids = build_position_ids(s_text=512, H_p=height // 16, W_p=width // 16)
        timesteps = build_schedule(num_steps=num_steps, N=W_p * H_p)
        latent = euler_sample(
            model=self.batched_dit,
            img_ids=img_ids,
            x_init=latent,
            txt=seq,
            txt_ids=txt_ids,
            y=pooled,
            guidance=guidance,
            timesteps=timesteps,
        )

        return latent

    def decode(self, latent, height, width):
        H_p, W_p = height // 16, width // 16
        latent = einops.rearrange(
            latent, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=H_p, w=W_p, ph=2, pw=2
        )
        img_out = self.vae.decode(latent)
        return img_out

    def __call__(self, prompts: list[str], height, width, num_steps, guidance, key) -> jnp.ndarray:
        pooled, seq = self.encode(prompts)
        latent = self.sample(
            pooled,
            seq,
            B=len(prompts),
            height=height,
            width=width,
            num_steps=num_steps,
            guidance=guidance,
            key=key,
        )
        img_out = self.decode(latent, height, width)
        return img_out
