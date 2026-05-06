import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from .rope import RoPE


class FluxAttention(eqx.Module):
    W_q : eqx.nn.Linear # (D, D)
    W_k : eqx.nn.Linear # (D, D)
    W_v : eqx.nn.Linear # (D, D)
    W_o : eqx.nn.Linear # (D, D)
    rope : eqx.Module
    dim : int
    num_heads : int
    head_dim : int
    
    def __init__(self, dim , num_heads , key):
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        key_q, key_k, key_v, key_o = jax.random.split(key, 4)
        self.W_q = eqx.nn.Linear(in_features = dim, out_features = dim, key = key_q, use_bias = False)
        self.W_k = eqx.nn.Linear(in_features = dim, out_features = dim, key = key_k, use_bias = False)
        self.W_v = eqx.nn.Linear(in_features = dim, out_features = dim, key = key_v, use_bias = False)
        self.W_o = eqx.nn.Linear(in_features = dim, out_features = dim, key = key_o, use_bias = False)
        self.rope = RoPE()

    def _split_head(self, x):
        #extract from w_proj(x) , reshape across head him
        B, T, _ = x.shape
        x = jnp.reshape(x, (B, T, self.num_heads, self.head_dim))
        # swap axes to (B, H, T, d_H)
        return jnp.swapaxes(x, 1, 2)

    def _merge_heads(self, x):
        B, H, T, d_H = x.shape
        #undo axes swap
        x = jnp.swapaxes(x, 1,2)
        #reshape back to (B, T , D)
        x = jnp.reshape(x, (B, T, H * d_H))
        return x

    def __call__(self, x: Float[Array , "batch seq_len dim"]):
        q_proj = jax.vmap(jax.vmap(self.W_q))
        k_proj = jax.vmap(jax.vmap(self.W_k))
        v_proj = jax.vmap(jax.vmap(self.W_v))
        o_proj = jax.vmap(jax.vmap(self.W_o))

        # (B, H, T, d_H)
        Q = self._split_head(q_proj(x)) 
        K = self._split_head(k_proj(x))
        V = self._split_head(v_proj(x))

        Q = self.rope(Q)
        K = self.rope(K)

        scores = Q @ jnp.swapaxes(K, -1 , -2) / jnp.sqrt(self.head_dim)
        weights = jax.nn.softmax(scores, axis = -1)
        context = weights @ V

        context = self._merge_heads(context)
        out = o_proj(context)
        return out
