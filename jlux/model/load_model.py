from .flux import FluxParams, Flux
import jax
from jax.tree_util import GetAttrKey, SequenceKey
import jax.tree_util as jtu
import jax.numpy as jnp
import equinox as eqx
from safetensors import safe_open


def path_to_key(path: tuple):
    """
    Render a pytree path tuple as a safetensors-style dot-separated key string. Applies BFL's Sequential-indexing rewrites for img_mlp, txt_mlp, and adaLN_modulation
    """
    out = []
    for entry in path:
        if isinstance(entry, GetAttrKey):
            out.append(entry.name)
        elif isinstance(entry, SequenceKey):
            out.append(str(entry.idx))

    key = ".".join(out)
    key = key.replace("img_mlp.linear1", "img_mlp.0")
    key = key.replace("img_mlp.linear2", "img_mlp.2")
    key = key.replace("txt_mlp.linear1", "txt_mlp.0")
    key = key.replace("txt_mlp.linear2", "txt_mlp.2")
    key = key.replace("final_layer.adaLN_modulation", "final_layer.adaLN_modulation.1")
    return key


def load_flux(cfg: FluxParams, path: str, dtype=jnp.bfloat16) -> Flux:
    template = eqx.filter_eval_shape(Flux, cfg, key=jax.random.PRNGKey(0))
    with safe_open(path, framework="flax") as f:
        tensors = {k: f.get_tensor(k).astype(dtype) for k in f.keys()}

    arrays_template = eqx.filter(
        template, lambda x: isinstance(x, jax.ShapeDtypeStruct)
    )
    leaves_with_paths, treedef = jtu.tree_flatten_with_path(arrays_template)
    new_leaves = []
    used_keys = set()

    for p, leaf in leaves_with_paths:
        key = path_to_key(p)

        if key not in tensors:
            raise KeyError(f"safetensors missing key '{key}' for template path {p}")

        tensor = tensors[key]

        if tensor.shape != leaf.shape:
            raise ValueError(
                f"shape mismatch at '{key}': "
                f"template wants {leaf.shape}, safetensors has {tensor.shape}"
            )

        new_leaves.append(tensor)
        used_keys.add(key)
    unused = set(tensors.keys()) - used_keys
    if unused:
        raise ValueError(
            f"{len(unused)} unused safetensors keys, e.g. {sorted(unused)[:5]}"
        )
    new_arrays_template = jtu.tree_unflatten(treedef, new_leaves)
    static_template = eqx.filter(
        template, lambda x: not isinstance(x, jax.ShapeDtypeStruct)
    )
    model = eqx.combine(new_arrays_template, static_template)
    return model
