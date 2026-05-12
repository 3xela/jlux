import jax.numpy as jnp


def mu(N):
    (m_min, m_max) = (0.5, 1.15)
    (N_min, N_max) = (256, 4096)

    return m_min + (N - N_min) / (N_max - N_min) * (m_max - m_min)


def schedule(t, N):
    return (mu(N) * t) / (1 + (mu(N) - 1) * t)


def build_schedule(num_steps, N):
    array = jnp.linspace(1.0, 0.0, num_steps + 1, dtype=jnp.bfloat16)
    return schedule(array, N)


def euler_sample(model, x_init, img_ids, txt, txt_ids, y, guidance, timesteps):
    x = x_init
    for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
        dt = t_next - t_curr
        v = model(x, img_ids, txt, txt_ids, t_curr, y, guidance)
        x = x + dt * v
    return x
