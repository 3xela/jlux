import jax
import jax.numpy as jnp
import equinox as eqx
from ..layers.modulation import Modulation6D
from ..layers.norms import LayerNorm
from ..layers.attention import FluxSelfAttention
from ..layers.mlp import FluxMLP
from ..layers.rope import RoPE


class FluxDoubleStreamBlock(eqx.Module):
    img_mod: Modulation6D
    txt_mod: Modulation6D
    img_norm1: LayerNorm
    txt_norm1: LayerNorm
    img_norm2: LayerNorm
    txt_norm2: LayerNorm
    img_attn: FluxSelfAttention
    txt_attn: FluxSelfAttention
    img_mlp: FluxMLP
    txt_mlp: FluxMLP

    rope: RoPE

    dim: int
    num_heads: int
    head_dim: int

    def __init__(self, dim, num_heads, key):
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0
        self.head_dim = self.dim // self.num_heads

        self.rope = RoPE()
        keys = jax.random.split(key, 6)

        self.img_mod = Modulation6D(dim=self.dim, key=keys[0])
        self.txt_mod = Modulation6D(dim=self.dim, key=keys[1])

        self.img_norm1 = LayerNorm(dim=self.dim, use_affine=False)
        self.img_norm2 = LayerNorm(dim=self.dim, use_affine=False)

        self.txt_norm1 = LayerNorm(dim=self.dim, use_affine=False)
        self.txt_norm2 = LayerNorm(dim=self.dim, use_affine=False)

        self.img_attn = FluxSelfAttention(
            dim=self.dim, num_heads=self.num_heads, key=keys[2]
        )
        self.txt_attn = FluxSelfAttention(
            dim=self.dim, num_heads=self.num_heads, key=keys[3]
        )

        self.txt_mlp = FluxMLP(in_dim=self.dim, key=keys[4])
        self.img_mlp = FluxMLP(in_dim=self.dim, key=keys[5])

    def __call__(self, img, txt, temb, pos_ids):
        (
            img_shift_msa,
            img_scale_msa,
            img_gate_msa,
            img_shift_mlp,
            img_scale_mlp,
            img_gate_mlp,
        ) = self.img_mod(temb)
        (
            txt_shift_msa,
            txt_scale_msa,
            txt_gate_msa,
            txt_shift_mlp,
            txt_scale_mlp,
            txt_gate_mlp,
        ) = self.txt_mod(temb)

        img_modulated = (1 + img_scale_msa) * self.img_norm1(img) + img_shift_msa
        txt_modulated = (1 + txt_scale_msa) * self.txt_norm1(txt) + txt_shift_msa

        img_Q, img_K, img_V = self.img_attn.qkv_proj(img_modulated)
        txt_Q, txt_K, txt_V = self.txt_attn.qkv_proj(txt_modulated)

        s_text = txt_Q.shape[1]

        Q = jnp.concat([txt_Q, img_Q], axis=1)
        K = jnp.concat([txt_K, img_K], axis=1)
        V = jnp.concat([txt_V, img_V], axis=1)

        Q = self.rope(Q, pos_ids)
        K = self.rope(K, pos_ids)

        scores = Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(self.head_dim)
        attn = jax.nn.softmax(scores, axis=-1) @ V

        txt_ctx, img_ctx = jnp.split(attn, [s_text], axis=1)

        txt_proj = self.txt_attn.out_proj(txt_ctx)
        img_proj = self.img_attn.out_proj(img_ctx)

        txt_residual = txt + txt_gate_msa * txt_proj
        img_residual = img + img_gate_msa * img_proj

        txt_pre_mlp = (1 + txt_scale_mlp) * self.txt_norm2(txt_residual) + txt_shift_mlp
        img_pre_mlp = (1 + img_scale_mlp) * self.img_norm2(img_residual) + img_shift_mlp

        txt_mlp_out = jax.vmap(self.txt_mlp)(txt_pre_mlp)
        img_mlp_out = jax.vmap(self.img_mlp)(img_pre_mlp)

        txt_out = txt_residual + txt_mlp_out * txt_gate_mlp
        img_out = img_residual + img_mlp_out * img_gate_mlp

        return img_out, txt_out
