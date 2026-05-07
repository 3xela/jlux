import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from .modulation import Modulation6D
from .norms import LayerNorm
from .attention import FluxAttention
from .mlp import FluxMLP

class FluxDoubleStreamBlock(eqx.Module):
    image_mod: Modulation6D
    text_mod: Modulation6D
    image_norm1: LayerNorm
    text_norm1: LayerNorm
    image_norm2: LayerNorm
    text_norm2: LayerNorm
    image_attn: FluxAttention
    text_attn: FluxAttention
    image_mlp: FluxMLP
    text_mlp: FluxMLP

    dim: int
    num_heads: int

    def __init__(self, dim, num_heads, key):
        self.dim = dim
        self.int = num_heads
        jax.random.split(key, 8)

