import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

class LayerNorm(eqx.Module):
    gamma : Float[Array, "dim"] | None
    beta : Float[Array, "dim"] | None
    eps : Float
    use_affine : bool

    def __init__(self, dim, use_affine = True):
        self.gamma = jnp.ones(dim)
        self.beta = jnp.zeros(dim)
        self.eps = 1e-6
        self.use_affine = use_affine

    def __call__(self, x):
        # x.shape = (...,  D)
        mean = jnp.mean(x, axis = -1, keepdims = True)
        var = jnp.var(x, axis = -1, keepdims = True)
        normalized = (x - mean) / (jnp.sqrt(var + self.eps))

        return self.gamma * normalized + self.beta if self.use_affine else normalized
    

class RMSNorm(eqx.Module):
    gamma : Float[Array, "dim"]
    eps: Float

    def __init__(self, dim):
        self.gamma = jnp.ones(dim)
        self.eps = 1e-6

    def __call__(self, x):
        #x.shape = (..., D)
        ms = jnp.mean(jnp.square(x), axis = -1, keepdims = True)
        normalized = x / (jnp.sqrt(ms + self.eps))
        return self.gamma * normalized
    
class QKNorm(eqx.Module):
    q_norm: RMSNorm
    k_norm: RMSNorm
    def __init__(self, dim):
        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)
    def __call__(self, Q,K,V):
        return self.q_norm(Q), self.k_norm(K), V