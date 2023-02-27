import jax.numpy as jnp

# ds functions
def scaled_f(w):
    return jnp.abs(w) * (1 - jnp.abs(w / jnp.pi) ** 3 / 4.)

# ds functions
def scaled_g(w):
    return w * (1 - 3. / 2. * jnp.abs(w / jnp.pi) + 1. / 2. * jnp.abs(w / jnp.pi) ** 2)

# ds functions
def nu_distance(xea, a, b, has_sym=True):
    w = jnp.einsum('...ijk,lk->...ijl', xea, b)
    mod = (w + jnp.pi) // (2 * jnp.pi)
    w = (w - mod * 2 * jnp.pi)
    r1 = (jnp.linalg.norm(a, axis=-1) * scaled_f(w)) ** 2
    sg = scaled_g(w)
    rel = jnp.einsum('...i,ij->...j', sg, a)
    if has_sym:
        sf = scaled_f(w)
        rel_f = jnp.einsum('...i,ij->...j', sf, a)
        rel = jnp.concatenate([rel, rel_f], axis=-1)
    r2 = jnp.einsum('ij,kj->ik', a, a) * (sg[..., :, None] * sg[..., None, :])
    result = jnp.sum(r1, axis=-1) + jnp.sum(r2 * (jnp.ones(r2.shape[-2:]) - jnp.eye(r2.shape[-1])), axis=[-1, -2])
    sd = result ** 0.5
    return sd, rel

