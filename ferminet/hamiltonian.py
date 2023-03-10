# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluating the Hamiltonian on a wavefunction."""

from typing import Any, Sequence

import chex
from ferminet import networks
import jax
from jax import lax
import jax.numpy as jnp
from typing_extensions import Protocol
from ferminet.cmplx import hamiltonian as cmplx_h


class MomentLocalEnergy(Protocol):

    def __call__(self, params: networks.ParamTree, key: chex.PRNGKey,
                 data: jnp.ndarray) -> jnp.ndarray:
        """Returns the local energy of a Hamiltonian at a configuration.
    返回的是某时刻的local energy，需要提供mcmc采样的data，以及某个时刻(总时间离散的时间网格).

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
      这里时间是放进data，还是单独提出来呢
    """


class MakeMomentLocalEnergy(Protocol):

    def __call__(self,
                 f: networks.FermiNetLike,
                 atoms: jnp.ndarray,
                 charges: jnp.ndarray,
                 nspins: Sequence[int],
                 use_scan: bool = False,
                 **kwargs: Any) -> MomentLocalEnergy:
        """Builds the LocalEnergy function.

    Args:
      f: Callable which evaluates the sign and log of the magnitude of the
        wavefunction.
      atoms: atomic positions.
      charges: nuclear charges.
      nspins: Number of particles of each spin.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      **kwargs: additional kwargs to use for creating the specific Hamiltonian.
    """


def local_kinetic_energy(
        f: networks.LogFermiNetLike,
        use_scan: bool = False) -> networks.LogFermiNetLike:
    """Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the log of the magnitude of the wavefunction.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """

    def _lapl_over_f(params, data):
        n = data.shape[0]
        eye = jnp.eye(n)
        grad_f = jax.grad(f, argnums=1)
        grad_f_closure = lambda x: grad_f(params, x)
        primal, dgrad_f = jax.linearize(grad_f_closure, data)
        # 这里的f是LogFermiNetLike(Params, electrons, time) -> log magnitude of function f
        # grad_f是对electrons求的，用closure完成对参数params和时刻moment的固定
        # 因为data最后一个维度是ntimestep，所以在laplacian外面直接用vmap

        if use_scan:
            _, diagonal = lax.scan(
                lambda i, _: (i + 1, dgrad_f(eye[i])[i]), 0, None, length=n)
            result = -0.5 * jnp.sum(diagonal)
        else:
            result = -0.5 * lax.fori_loop(
                0, n, lambda i, val: val + dgrad_f(eye[i])[i], 0.0)
        return result - 0.5 * jnp.sum(primal ** 2)

    _lapl_over_f_vmap = jax.vmap(_lapl_over_f, in_axes=(None, -1), out_axes=-1)

    return _lapl_over_f_vmap


# 利用vmap完成对data的ntimestep部分的批操作


def potential_electron_electron(r_ee: jnp.ndarray) -> jnp.ndarray:
    """Returns the electron-electron potential.

  Args:
    r_ee: Shape (neletrons, nelectrons, :, ntimestep). r_ee[i,j,0,t] gives the distance
      between electrons i and j at time t. Other elements in the final axes are not
      required.
  """

    def pee_for_notime(x):
        result = jnp.sum(jnp.triu(1 / x[..., 0], k=1))
        return result

    return jax.vmap(pee_for_notime, in_axes=-1, out_axes=-1)(r_ee)


# 利用vmap完成对data的ntimestep部分的批操作


def potential_electron_nuclear(charges: jnp.ndarray,
                               r_ae: jnp.ndarray) -> jnp.ndarray:
    """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
  """

    def pen_for_time(x):
        result = -jnp.sum(charges / x[..., 0])
        return result

    return jax.vmap(pen_for_time, in_axes=-1, out_axes=-1)(r_ae)


# 利用vmap完成对data的ntimestep部分的批操作


def potential_nuclear_nuclear(charges: jnp.ndarray,
                              atoms: jnp.ndarray) -> jnp.ndarray:
    """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    atoms: Shape (natoms, ndim). Positions of the atoms.
  """
    r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
    return jnp.sum(
        jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


# 这个是对atom的操作，不需要对时间批处理，需要扩充时间维度


def potential_energy(r_ae: jnp.ndarray, r_ee: jnp.ndarray, atoms: jnp.ndarray,
                     charges: jnp.ndarray) -> jnp.ndarray:
    """Returns the potential energy for this electron configuration.

  Args:
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
  """
    return (potential_electron_electron(r_ee) +
            potential_electron_nuclear(charges, r_ae) +
            potential_nuclear_nuclear(charges, atoms)[..., None])


# pnn的时间维度，在此处扩充，应该是复制了ntimestep份


def local_energy(f: networks.FermiNetLike,
                 atoms: jnp.ndarray,
                 charges: jnp.ndarray,
                 nspins: Sequence[int],
                 use_scan: bool = False,
                 do_complex: bool = False,
                 ) -> MomentLocalEnergy:
    """Creates the function to evaluate the local energy.维度是(ntimestep)

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    do_complex: 是否用复数形式

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
    del nspins
    log_abs_f = lambda *args, **kwargs: f(*args, **kwargs)[1]
    # already take the value of the network
    # log_abs_f = f
    if do_complex:
        ke = cmplx_h.local_kinetic_energy(log_abs_f)
    else:
        ke = local_kinetic_energy(log_abs_f, use_scan=use_scan)

    def _e_l(params: networks.ParamTree, key: chex.PRNGKey,
             data: jnp.ndarray) -> jnp.ndarray:
        """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
        del key  # unused
        _, _, r_ae, r_ee = networks.construct_input_features(data, atoms)
        potential = potential_energy(r_ae, r_ee, atoms, charges)
        kinetic = ke(params, data)
        return potential + kinetic

    return _e_l
# 对时间的数据维度怎么办
