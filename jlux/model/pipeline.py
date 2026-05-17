from text import CLIPWrapper, T5Wrapper
from model import Flux, FluxParams, load_model
from vae import VAEWrapper
from sampler import build_schedule, euler_sample
import jax 
import jax.numpy as jnp
import einops
from dit import build_position_ids


class FluxPipeline:
    clip : CLIPWrapper
    t5: T5Wrapper
    vae: VAEWrapper
    dit: Flux

    def __init__(self, load_cfg: FluxParams, flux_path : "str"):
        self.clip = CLIPWrapper()
        self.t5 = T5Wrapper()
        self.vae = VAEWrapper()
        self.dit = load_model(cfg = load_cfg, path = flux_path)

    def __call__(self, prompts : list[str], height, width, num_steps, guidance, key) -> jnp.ndarray:
        assert height % 16 == 0 and width % 16 == 0, "image dims must be multiples of 16"
        H_p, W_p = height // 16, width // 16
        pooled = self.clip(prompts)
        seq = self.t5(prompts)
        latent = jax.random.normal(key, (1,16,height//8, width //8))
        latent = einops.rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        txt_ids, img_ids = build_position_ids(s_text=512, H_p = height//16, W_p = width//16)
        timesteps = jnp.linspace(1.0, 0.0, num_steps + 1)
        latent = euler_sample(model = self.dit, img_ids = img_ids, x_init= latent, txt = seq, txt_ids=txt_ids, y = pooled, guidance = guidance, timesteps = timesteps)
        latent = einops.rearrange(latent, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=H_p, w=W_p, ph=2, pw=2)
    