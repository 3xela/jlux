import equinox as eqx
import jax
from jlux.dit.attention import FluxAttention
from jlux.dit.rope import build_position_ids

def main():
    key = jax.random.PRNGKey(0)

    pos_ids = build_position_ids(10,4,4)
    key, subkey = jax.random.split(key)

    x = jax.random.normal(subkey, (2, 26, 3072))
    attn = FluxAttention(dim = 3072, num_heads = 24, key = key)
    out = attn(x, pos_ids)
    print(f"shape of x:{x.shape}, shape of attn(x):{out.shape}")

if __name__ == "__main__":
    main()
