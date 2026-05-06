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
    

    
# TODO: Rework RoPE for Flux's multi-axis structure
#
# Current state: standard 1D RoPE, assumes arange(T) positions, single rotation
# per token. Won't work for Flux's 2D image grid + text concatenation.
#
# Target: Flux uses 3-axis RoPE with axes_dim = [16, 56, 56] for head_dim=128.
# - Axis 0: "constant" axis, always position 0 (effectively unused but allocated)
# - Axis 1: height position in patch grid
# - Axis 2: width position in patch grid
# Text tokens get position (0, 0, 0) -> identity rotation -> pass through.
# Image tokens get (0, h, w) for their patch grid position.
#
# Required changes:
#
# [ ] Accept explicit position IDs of shape (B, T, 3) instead of using arange(T)
#
# [ ] Refactor current rotation math into a rope_1d(pos, dim, theta) helper that
#     takes explicit position values and an allocated dim count for that axis.
#     The existing real/imag rotation logic is correct and recyclable.
#
# [ ] Wrap rope_1d to handle the 3-axis split:
#       - Split position IDs along last axis into per-axis position arrays
#       - Call rope_1d for each axis with its axes_dim[i] allocation
#       - Concatenate the three rotated slices along the head_dim axis
#
# [ ] Make axes_dim and theta configurable (Flux uses [16, 56, 56], theta=10000)
#
# [ ] Verify: identity rotation when all positions are zero (text token check)
#
# [ ] Verify: rotation preserves vector magnitude (|rope(x)| == |x|)
#
# [ ] Add a build_position_ids helper that constructs (B, T, 3) given:
#       - batch size B
#       - text sequence length S_txt (gets all zeros)
#       - image patch grid (H_patches, W_patches) -> (0, h, w) per patch
#     Output ordering: text tokens first, then image tokens (matches Flux concat order)
#
# Notes:
# - eps for trig stability not needed; cos/sin are bounded
# - cos/sin tables can be precomputed once per forward pass and shared across
#   all 57 attention layers (don't recompute per-block)
# - The reference packages cos/sin as (B, T, head_dim/2, 2, 2) rotation matrices;
#   the real/imag pair approach you have is mathematically equivalent
