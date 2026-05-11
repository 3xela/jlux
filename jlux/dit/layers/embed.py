import jax
import jax.numpy as jnp
import equinox as eqx


class MLPEmbedder(eqx.Module):
    in_layer: eqx.nn.Linear
    out_layer: eqx.nn.Linear

    def __init__(self, in_dim, hidden_dim, key):
        keys = jax.random.split(key , 2)
        self.in_layer = eqx.nn.Linear(in_features=in_dim, out_features=hidden_dim, key = keys[0])
        self.out_layer = eqx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, key = keys[1])
    def __call__(self, x):
        return self.out_layer(jax.nn.silu(self.in_layer(x)))


def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    #TODO this assuumes dim even. make it work for odd too by adding an extra 0
    half = dim // 2

    freqs = jnp.exp( -jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32)/half)
    args = (time_factor * t) * freqs
    cos = jnp.cos(args)
    sin = jnp.sin(args)
    out = jnp.concat([cos,sin], axis = 0)
    return out