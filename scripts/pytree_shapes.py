from jlux import Flux, FluxParams
import jax
import equinox as eqx

test_flux = eqx.filter_eval_shape(Flux, FluxParams(), key=jax.random.PRNGKey(0))

arrays_only = eqx.filter(test_flux, lambda x: isinstance(x, jax.ShapeDtypeStruct))

leaves, _ = jax.tree_util.tree_flatten(arrays_only)
print(f"leaves: {len(leaves)}")