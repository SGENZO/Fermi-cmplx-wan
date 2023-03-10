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

    def _lapl_over_f(params, data, moment):
        n = data.shape[0]
        eye = jnp.eye(n)
        grad_f = jax.grad(f, argnums=1)
        grad_f_closure = lambda x: grad_f(params, x, moment)
        primal, dgrad_f = jax.linearize(grad_f_closure, data)
        # 这里的f是LogFermiNetLike(Params, electrons, time) -> log magnitude of function f
        # grad_f是对electrons求的，用closure完成对参数params和时刻moment的固定

        if use_scan:
            _, diagonal = lax.scan(
                lambda i, _: (i + 1, dgrad_f(eye[i])[i]), 0, None, length=n)
            result = -0.5 * jnp.sum(diagonal)
        else:
            result = -0.5 * lax.fori_loop(
                0, n, lambda i, val: val + dgrad_f(eye[i])[i], 0.0)
        return result - 0.5 * jnp.sum(primal ** 2)

    return _lapl_over_f
# 完成时刻t的增加


def potential_electron_electron(r_ee: jnp.ndarray) -> jnp.ndarray:
    """Returns the electron-electron potential.

  Args:
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
  """
    return jnp.sum(jnp.triu(1 / r_ee[..., 0], k=1))


def potential_electron_nuclear(charges: jnp.ndarray,
                               r_ae: jnp.ndarray) -> jnp.ndarray:
    """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
  """
    return -jnp.sum(charges / r_ae[..., 0])


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
            potential_nuclear_nuclear(charges, atoms))


def local_energy(f: networks.FermiNetLike,
                 atoms: jnp.ndarray,
                 charges: jnp.ndarray,
                 nspins: Sequence[int],
                 use_scan: bool = False,
                 do_complex: bool = False,
                 ) -> MomentLocalEnergy:
    """Creates the function to evaluate the local energy.

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
    # log_abs_f = lambda *args, **kwargs: f(*args, **kwargs)[1]
    # already take the value of the network
    log_abs_f = f
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
