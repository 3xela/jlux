import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

def build_position_ids(s_text : int, H_p : int, W_p : int):
    hw_indices = jnp.indices((H_p, W_p))
    image_grid = jnp.transpose(hw_indices, (1,2,0))

    zeros = jnp.zeros((H_p, W_p, 1))

    hw_grid = jnp.concat([zeros, image_grid], axis =-1)
    hw_grid = jnp.reshape(hw_grid, (H_p * W_p, 3))

    text_pos= jnp.zeros(shape=(s_text, 3))
    out = jnp.concat([text_pos, hw_grid] , axis = 0)
    return out

def rotate(x, position_ids, theta):
        D = x.shape[-1]
        assert D % 2 == 0

        t = position_ids.astype(jnp.float32)

        pairs = jnp.arange(D//2)
        omega = 1.0 / (theta ** (2 * pairs / D))

        angles = t[:, None] * omega[None, :]

        cos = jnp.cos(angles)
        sin = jnp.sin(angles)

        x_real = x[..., 0::2]
        x_imag  = x[..., 1::2]

        x_real_new = x_real * cos - x_imag * sin
        x_imag_new = x_real * sin + x_imag * cos

        stacked = jnp.stack([x_real_new, x_imag_new], axis = -1)
        stacked = stacked.reshape(x.shape)
        return stacked

class RoPE(eqx.Module):
    axis_dim: tuple
    theta: float
    def __init__(self, axis_dim = (16,56,56), theta = 10000.0):
        self.axis_dim = axis_dim
        self.theta = theta

    def __call__(self, x, position_ids):
        #x.shape(H, T, head_dim)
        # Split into axis-0, axis-1, axis-2 slices: sizes 16, 56, 56 for Flux head_dim=128
        first, second, third = jnp.split(x, [16,72], axis = -1)

        first = rotate(first, position_ids[:, 0], self.theta)
        second = rotate(second, position_ids[:, 1], self.theta)
        third = rotate(third, position_ids[:, 2], self.theta)

        out = jnp.concat([first, second,third], axis = -1)
        return out
    
