import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from .norms import LayerNorm


class AdaLN(eqx.Module):
    norm : LayerNorm
    dim : int
    def __init__(self, dim):
        self.dim = dim
        self.norm = LayerNorm(self.dim, use_affine=False)
    def __call__(self, x, scale, shift):
        # x.shape = (... , D)
        # scale.shape = (... , D)
        # shift.shape = (... , D)
        return (1 + scale ) * self.norm(x) + shift
