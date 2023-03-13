import jax
import jax.numpy as jnp
from ferminet import networks


# implementation of deep splid `local_kinetic_energy_real_imag`
def local_kinetic_energy_ds(f: networks.LogFermiNetLike) -> networks.LogFermiNetLike:
    def _lapl_over_f(params, x):
        # 这里取-1位置，如果后面再vmap有没有影响
        # x.shape[-1]应当是ndim，应该取nelectron？
        ne = x.shape[0]
        eye = jnp.eye(ne)
        grad_f_real = jax.grad(lambda p, y: f(p, y).real, argnums=1)
        grad_f_imag = jax.grad(lambda p, y: f(p, y).imag, argnums=1)
        grad_f_real_closure = lambda y: grad_f_real(params, y)
        grad_f_imag_closure = lambda y: grad_f_imag(params, y)

        # loop body
        def _body_fun(i, val):
            primal_real, tangent_real = jax.jvp(grad_f_real_closure, (x,), (eye[i],))
            primal_imag, tangent_imag = jax.jvp(grad_f_imag_closure, (x,), (eye[i],))
            kine_real = val[0] + tangent_real[i] + primal_real[i] ** 2 - primal_imag[i] ** 2
            kine_imag = val[1] + tangent_imag[i] + 2 * primal_real[i] * primal_imag[i]
            return [kine_real, kine_imag]

        # result
        result = jax.lax.fori_loop(0, ne, _body_fun, [0.0, 0.0])
        result = -0.5 * (result[0] + 1j * result[1])
        return result

    _lapl_over_f_vmap = jax.vmap(_lapl_over_f, in_axes=(None, -1), out_axes=-1)

    return _lapl_over_f_vmap


def local_kinetic_energy(f: networks.LogFermiNetLike) -> networks.LogFermiNetLike:
    def _lapl_over_f(params, x):
        ne = x.shape[0]
        eye = jnp.eye(ne)
        grad_f_real = jax.grad(lambda p, y: f(p, y).real, argnums=1)
        grad_f_imag = jax.grad(lambda p, y: f(p, y).imag, argnums=1)
        grad_f_real_closure = lambda y: grad_f_real(params, y)
        grad_f_imag_closure = lambda y: grad_f_imag(params, y)
        primal_real, dgrad_f_real = jax.linearize(grad_f_real_closure, x)
        primal_imag, dgrad_f_imag = jax.linearize(grad_f_imag_closure, x)

        # loop body
        def _body_fun(i, val):
            incr_real = dgrad_f_real(eye[i])[i]
            incr_imag = dgrad_f_imag(eye[i])[i]
            return [val[0] + incr_real, val[1] + incr_imag]

        # result
        result = jax.lax.fori_loop(0, ne, _body_fun, [0.0, 0.0])
        result[0] += jnp.sum(primal_real ** 2) - jnp.sum(primal_imag ** 2)
        result[1] += 2. * jnp.sum(primal_real * primal_imag)
        result = -0.5 * (result[0] + 1j * result[1])
        return result

    _lapl_over_f_vmap = jax.vmap(_lapl_over_f, in_axes=(None, -1), out_axes=-1)

    return _lapl_over_f_vmap
