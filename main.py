import equinox as eqx
import jax
from jlux.dit.attention import FluxAttention

def main():
    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)

    x = jax.random.normal(subkey, (2, 16, 512))
    attn = FluxAttention(dim = 512, num_heads = 8, key = key)
    out = attn(x)
    print(f"shape of x:{x.shape}, shape of attn(x):{out.shape}")

if __name__ == "__main__":
    main()
