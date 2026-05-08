import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from .modules.modulation import Modulation6D
from .modules.norms import LayerNorm
from .modules.attention import FluxSelfAttention
from .modules.mlp import FluxMLP
from .modules.rope import RoPE

class FluxSingleStreamBlock(eqx.Module):
    passå