import numpy as np
import jax.numpy as jnp

def apply_pbc(
        rr,
        lattice,
        rec_lattice,
):
    shape = rr.shape
    sr = jnp.matmul(rr.reshape([-1,3]), rec_lattice)
    sr = jnp.mod(sr, 1.0)
    return jnp.matmul(sr, lattice).reshape(shape)

def apply_nearest_neighbor(
        rij,
        lattice,
        rec_lattice,
):
    srij = jnp.matmul(rij, rec_lattice)
    srij = jnp.mod(srij+0.5, 1.0) - 0.5
    rij = jnp.matmul(srij, lattice)
    return rij

def auto_nearest_neighbor(
        lattice,
        rc,
) -> bool:
    vol = np.linalg.det(lattice)
    tofacedist = np.cross(lattice[[1,2,0],:], lattice[[2,0,1],:])
    tofacedist = vol * np.reciprocal(np.linalg.norm(tofacedist, axis=1))
    return (rc <= 0.5 * tofacedist[0]) and (rc <= 0.5 * tofacedist[1]) and (rc <= 0.5 * tofacedist[2])


def spline_func(xx, rc, rc_smth):
    uu = (xx - rc_smth) / (rc - rc_smth)
    return uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1

def switch_func_poly(
        xx,
        rc = 3.0,
        rc_smth = 0.2,
):
    ret = \
        1.0 * (xx < rc_smth) + \
        spline_func(xx, rc, rc_smth) * jnp.logical_and(xx >= rc_smth, xx < rc) + \
        0.0 * (xx >= rc)
    return ret
