import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class LayerNorm(eqx.Module):
    gamma: Float[Array, "dim"] | None
    beta: Float[Array, "dim"] | None
    eps: Float = eqx.field(static=True)
    use_affine: bool = eqx.field(static=True)

    def __init__(self, dim, use_affine=True):
        if use_affine:
            self.gamma = jnp.ones(dim)
            self.beta = jnp.zeros(dim)
        else:
            self.gamma = None
            self.beta = None
        self.eps = 1e-5
        self.use_affine = use_affine

    def __call__(self, x):
        # x.shape = (...,  D)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / (jnp.sqrt(var + self.eps))

        result = self.gamma * normalized + self.beta if self.use_affine else normalized

        return result.astype(x.dtype)


class RMSNorm(eqx.Module):
    scale: Float[Array, "dim"]
    eps: Float = eqx.field(static=True)

    def __init__(self, dim):
        self.scale = jnp.ones(dim)
        self.eps = 1e-5

    def __call__(self, x):
        # x.shape = (..., D)
        ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normalized = x / (jnp.sqrt(ms + self.eps))
        return (self.scale * normalized).astype(x.dtype)


class QKNorm(eqx.Module):
    query_norm: RMSNorm
    key_norm: RMSNorm

    def __init__(self, dim):
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def __call__(self, Q, K, V):
        return self.query_norm(Q), self.key_norm(K), V
