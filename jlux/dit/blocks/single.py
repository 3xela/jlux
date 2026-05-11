import jax
import jax.numpy as jnp
import equinox as eqx
from ..layers.modulation import Modulation3D
from ..layers.norms import LayerNorm, QKNorm
from ..layers.attention import merge_heads, split_head
from ..layers.rope import RoPE


class FluxSingleStreamBlock(eqx.Module):
    modulation: Modulation3D
    linear1: eqx.nn.Linear  # (D, 7D)
    linear2: eqx.nn.Linear  # (5D, D)

    rope: RoPE
    norm: QKNorm
    pre_norm: LayerNorm

    dim: int
    head_dim: int
    num_heads: int

    def __init__(self, dim, num_heads, key):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        keys = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(
            in_features=self.dim, out_features=7 * dim, key=keys[0]
        )
        self.linear2 = eqx.nn.Linear(
            in_features=5 * self.dim, out_features=self.dim, key=keys[1]
        )
        self.modulation = Modulation3D(dim=dim, key=keys[2])

        self.rope = RoPE()
        self.norm = QKNorm(dim=self.head_dim)
        self.pre_norm = LayerNorm(dim=self.dim, use_affine=False)

    def __call__(self, x, temb, pos_ids):
        shift, scale, gate = self.modulation(temb)
        x_mod = (1 + scale) * self.pre_norm(x) + shift
        fused = jax.vmap(self.linear1)(x_mod)
        QKV, mlp_in = jnp.split(fused, [3 * self.dim], axis=1)
        Q, K, V = jnp.split(QKV, [self.dim, 2 * self.dim], axis=1)
        Q, K, V = (
            split_head(Q, self.num_heads),
            split_head(K, self.num_heads),
            split_head(V, self.num_heads),
        )
        Q, K, V = self.norm(Q, K, V)
        Q = self.rope(Q, pos_ids)
        K = self.rope(K, pos_ids)
        scores = Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(self.head_dim)
        attn = jax.nn.softmax(scores, axis=-1) @ V
        context = merge_heads(attn)
        mlp_activated = jax.nn.gelu(mlp_in, approximate=True)
        concat = jnp.concat([context, mlp_activated], axis=-1)
        output = jax.vmap(self.linear2)(concat)
        x = x + gate * output
        return x
