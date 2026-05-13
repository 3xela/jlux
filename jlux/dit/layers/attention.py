import jax
import jax.numpy as jnp
import equinox as eqx
from .norms import QKNorm


class FluxSelfAttention(eqx.Module):
    qkv: eqx.nn.Linear  # (3D, D)
    proj: eqx.nn.Linear
    norm: QKNorm

    num_heads: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, dim, num_heads, key):
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        key_qkv, key_proj = jax.random.split(key, 2)

        self.proj = eqx.nn.Linear(in_features=dim, out_features=dim, key=key_proj)
        self.qkv = eqx.nn.Linear(
            in_features=self.dim, out_features=3 * self.dim, key=key_qkv
        )
        self.norm = QKNorm(self.head_dim)

    def qkv_proj(self, x):
        QKV = jax.vmap(self.qkv)(x)
        q_proj, k_proj, v_proj = jnp.split(QKV, [self.dim, 2 * self.dim], axis=-1)
        Q = split_head(q_proj, self.num_heads)
        K = split_head(k_proj, self.num_heads)
        V = split_head(v_proj, self.num_heads)
        return self.norm(Q, K, V)

    def out_proj(self, context):
        x = merge_heads(context)
        return jax.vmap(self.proj)(x)


def split_head(x, num_heads):
    # extract from w_proj(x) , reshape across head him
    T, D = x.shape
    x = jnp.reshape(x, (T, num_heads, D // num_heads))
    # swap axes to (H, T, d_H)
    return jnp.swapaxes(x, 0, 1)


def merge_heads(x):
    H, T, d_H = x.shape
    # undo axes swap
    x = jnp.swapaxes(x, 0, 1)
    # reshape back to (T , D)
    x = jnp.reshape(x, (T, H * d_H))
    return x
