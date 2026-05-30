# scripts/dump_namespaces.py
"""
Dump the safetensors key namespace from FLUX.1-dev alongside the pytree
paths from the jlux Flux model. Writes three files:
  - flux_dev_keys.txt   : safetensors keys (sorted)
  - jlux_paths.txt      : jlux pytree paths (sorted)
  - namespace_diff.txt  : side-by-side diff for renaming reference
"""

import equinox as eqx
import jax
import jax.tree_util as jtu
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from jlux.model.flux import Flux, FluxParams

# ---------- 1. Get the checkpoint ----------

print("Locating flux1-dev.safetensors (downloads on first run, ~23GB)...")
ckpt_path = hf_hub_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    filename="flux1-dev.safetensors",
)
print(f"Checkpoint at: {ckpt_path}")


# ---------- 2. Dump safetensors keys ----------

with safe_open(ckpt_path, framework="pt") as f:
    st_entries = sorted([(k, tuple(f.get_slice(k).get_shape())) for k in f.keys()])

with open("flux_dev_keys.txt", "w") as out:
    for k, shape in st_entries:
        out.write(f"{k}\t{shape}\n")

print(f"safetensors keys: {len(st_entries)} (written to flux_dev_keys.txt)")


# ---------- 3. Dump jlux pytree paths ----------


def path_to_str(path):
    parts = []
    for p in path:
        if isinstance(p, jtu.GetAttrKey):
            parts.append(p.name)
        elif isinstance(p, jtu.SequenceKey):
            parts.append(str(p.idx))
        elif isinstance(p, jtu.DictKey):
            parts.append(str(p.key))
        else:
            parts.append(str(p))
    return ".".join(parts)


model = Flux(FluxParams(), key=jax.random.PRNGKey(0))
arrays_only = eqx.filter(model, eqx.is_array)
leaves_with_paths, _ = jtu.tree_flatten_with_path(arrays_only)

jlux_entries = sorted((path_to_str(path), tuple(leaf.shape)) for path, leaf in leaves_with_paths)

with open("jlux_paths.txt", "w") as out:
    for p, shape in jlux_entries:
        out.write(f"{p}\t{shape}\n")

print(f"jlux pytree leaves: {len(jlux_entries)} (written to jlux_paths.txt)")


# ---------- 4. Side-by-side diff ----------

# Align by sorted position. This won't be a "correct" diff in any semantic
# sense — both lists are sorted alphabetically and shown side-by-side so you
# can scan for where they diverge. For real alignment you'll do it by hand
# once you see the structure.
max_len = max(len(st_entries), len(jlux_entries))
with open("namespace_diff.txt", "w") as out:
    out.write(f"{'SAFETENSORS':<70} | {'JLUX':<70}\n")
    out.write(f"{'-' * 70} | {'-' * 70}\n")
    for i in range(max_len):
        st = f"{st_entries[i][0]} {st_entries[i][1]}" if i < len(st_entries) else ""
        jl = f"{jlux_entries[i][0]} {jlux_entries[i][1]}" if i < len(jlux_entries) else ""
        out.write(f"{st:<70} | {jl:<70}\n")

print("Side-by-side written to namespace_diff.txt")


# ---------- 5. Quick summary ----------

print()
print(f"Leaf count match: {len(st_entries) == len(jlux_entries)}")
if len(st_entries) != len(jlux_entries):
    print(f"  safetensors: {len(st_entries)}")
    print(f"  jlux:        {len(jlux_entries)}")
    print("  → structural mismatch, expected for first pass")
