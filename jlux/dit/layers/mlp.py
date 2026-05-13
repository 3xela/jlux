import jax
import equinox as eqx


class FluxMLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    in_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(self, in_dim: int, key):
        self.in_dim = in_dim
        self.hidden_dim = 4 * self.in_dim
        key_1, key_2 = jax.random.split(key, 2)
        self.linear1 = eqx.nn.Linear(
            in_features=self.in_dim, out_features=self.hidden_dim, key=key_1
        )
        self.linear2 = eqx.nn.Linear(
            in_features=self.hidden_dim, out_features=self.in_dim, key=key_2
        )

    def __call__(self, x):
        # x.shape (D, )
        x = jax.nn.gelu(self.linear1(x), approximate=True)
        return self.linear2(x)
