import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class Modulation6D(eqx.Module):
    linear : eqx.nn.Linear

    def __init__(self, dim, key):
        self.linear  = eqx.nn.Linear(in_features=dim, out_features= 6 * dim, key = key)

    def __call__(self, temb):
        #temb.shape (D, )
        x = jax.nn.silu(temb)
        out = self.linear(x)
        #out.shape = (6D, ), split along each D
        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = jnp.split(out, 6, axis = -1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp



class Modulation3d(eqx.Module):
    linear : eqx.nn.Linear

    def __init__(self, dim, key):
        self.linear  = eqx.nn.Linear(in_features=dim, out_features= 3 * dim, key = key)

    def __call__(self, temb):
        #temb.shape (D, )
        x = jax.nn.silu(temb)
        out = self.linear(x)
        #out.shape = (3D, ), split along each D
        shift, scale, gate = jnp.split(out, 3, axis = -1)
        return shift, scale, gate

