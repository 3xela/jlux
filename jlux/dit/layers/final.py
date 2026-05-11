import jax
import jax.numpy as jnp
import equinox as eqx
from .norms import LayerNorm


class FinalLayer(eqx.Module):
    norm_final : LayerNorm
    linear: eqx.nn.Linear
    adaLN_modulation: eqx.nn.Linear

    dim: int


    def __init__(self, dim, patch_size, out_channels, key):
        self.dim = dim
        keys = jax.random.split(key, 2)
        self.norm_final = LayerNorm(dim, use_affine=False)
        self.adaLN_modulation = eqx.nn.Linear(in_features=dim, out_features= 2 * dim, key=keys[0])
        self.linear = eqx.nn.Linear(in_features=dim, out_features=patch_size**2  * out_channels, key=keys[1]) 

    def __call__(self, x, temb):
        mod = self.adaLN_modulation(jax.nn.silu(temb))
        shift, scale = jnp.split(mod, [self.dim], axis = -1)

        x = (1+scale) * self.norm_final(x) + shift
        x = jax.vmap(self.linear)(x)

        return x