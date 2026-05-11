from dit.blocks import FluxDoubleStreamBlock, FluxSingleStreamBlock
from dit.layers import MLPEmbedder, FinalLayer
import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass

@dataclass(frozen=True)
class FluxParams:
    in_channels: int = 64
    vec_in_dim : int = 768 #pooled CLIP dim
    context_in_dim : int = 4096 # T5 dim
    hidden_size : int = 3072 # dim
    mlp_ratio : float = 4.0
    num_heads: int = 24
    depth: int = 19 #double blocks
    depth_single_blocks: int = 38
    guidance_embed : bool = True #True for dev, false for schnell
    theta: float = 10000.0
    axes_dim: tuple = (16,56,56)

class Flux(eqx.Module):
    img_in : eqx.nn.Linear
    txt_in: eqx.nn.Linear
    time_in : MLPEmbedder
    vector_in : MLPEmbedder
    guidance_in : MLPEmbedder | None

    double_blocks : list[FluxDoubleStreamBlock]
    single_blocks : list[FluxSingleStreamBlock]

    final_layer : FinalLayer

    cfg : FluxParams

    def __init__(self, cfg : FluxParams, key):
        self.cfg = cfg
        emb_keys, double_keys, single_keys, final_key = jax.random.split(key, 4)

        emb_keys = jax.random.split(emb_keys, 5)
        double_keys = jax.random.split(double_keys, cfg.depth)
        single_keys = jax.random.split(single_keys, cfg.depth_single_blocks)

        self.img_in = eqx.nn.Linear(in_features=cfg.in_channels, out_features=cfg.hidden_size, key = emb_keys[0])
        self.txt_in = eqx.nn.Linear(in_features = cfg.context_in_dim, out_features=cfg.hidden_size, key = emb_keys[1])
        self.time_in = MLPEmbedder(in_dim = 256, hidden_dim= cfg.hidden_size, key = emb_keys[2])
        self.vector_in = MLPEmbedder(in_dim=cfg.vec_in_dim, hidden_dim=cfg.hidden_size, key = emb_keys[3])
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=cfg.hidden_size, key = emb_keys[4])

        self.double_blocks = [FluxDoubleStreamBlock(dim = cfg.hidden_size, num_heads = cfg.num_heads, key = k) for k in double_keys]
        self.single_blocks = [FluxSingleStreamBlock(dim = cfg.hidden_size, num_heads=cfg.num_heads, key = k) for k in single_keys]

        self.final_layer = FinalLayer(dim = cfg.hidden_size, patch_size=1, out_channels=cfg.in_channels, key = final_key)