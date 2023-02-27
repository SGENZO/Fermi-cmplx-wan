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

"""Utilities for pretraining and importing PySCF models."""

from typing import Callable, Optional, Sequence, Tuple, Union

from absl import logging
import chex
import ferminet
from ferminet import constants
from ferminet import envelopes
from ferminet import mcmc
from ferminet import networks
from ferminet.utils import scf
from ferminet.pbc import scf as pbc_scf
from ferminet.utils import system
import jax
from jax import numpy as jnp
import kfac_jax
import numpy as np
import optax
import pyscf


# Given the parameters and electron positions, return arrays(s) of the orbitals.
# See networks.fermi_net_orbitals. (Note only the orbitals, and not envelope
# parameters, are required.)
FermiNetOrbitals = Callable[[networks.ParamTree, jnp.ndarray],
                            Sequence[jnp.ndarray]]


def get_hf(molecule: Optional[Sequence[system.Atom]] = None,
           nspins: Optional[Tuple[int, int]] = None,
           basis: Optional[str] = 'sto-3g',
           pyscf_mol: Optional[pyscf.gto.Mole] = None,
           restricted: Optional[bool] = False,
           lattice : Optional[np.ndarray] = None,
) -> scf.Scf:
  if lattice is None:
    scf_approx = get_hf_mol(
      molecule=molecule, nspins=nspins, basis=basis,
      pyscf_mol=pyscf_mol, restricted=restricted,
    )
  else:
    if pyscf_mol is None:
      # raise RuntimeError("the pyscf molecule should be provided when using pbc")
      return None
    if not isinstance(pyscf_mol, pyscf.pbc.gto.Cell):
      raise RuntimeError("the pyscf molecule should be of type pyscf.pbc.gto.Cell")
    scf_approx = pbc_scf.Scf(pyscf_mol, restricted=restricted)
    scf_approx.run()
  return scf_approx

def get_hf_mol(molecule: Optional[Sequence[system.Atom]] = None,
           nspins: Optional[Tuple[int, int]] = None,
           basis: Optional[str] = 'sto-3g',
           pyscf_mol: Optional[pyscf.gto.Mole] = None,
           restricted: Optional[bool] = False) -> scf.Scf:
  """Returns an Scf object with the Hartree-Fock solution to the system.

  Args:
    molecule: the molecule in internal format.
    nspins: tuple with number of spin up and spin down electrons.
    basis: basis set to use in Hartree-Fock calculation.
    pyscf_mol: pyscf Mole object defining the molecule. If supplied,
      molecule, nspins and basis are ignored.
    restricted: If true, perform a restricted Hartree-Fock calculation,
      otherwise perform an unrestricted Hartree-Fock calculation.
  """
  if pyscf_mol:
    scf_approx = scf.Scf(pyscf_mol=pyscf_mol, restricted=restricted)
  else:
    scf_approx = scf.Scf(
        molecule, nelectrons=nspins, basis=basis, restricted=restricted)
  scf_approx.run()
  return scf_approx


def eval_orbitals(scf_approx: scf.Scf, pos: Union[np.ndarray, jnp.ndarray],
                  nspins: Tuple[int, int], use_geminal: bool=False) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates SCF orbitals from PySCF at a set of positions.

  Args:
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    pos: an array of electron positions to evaluate the orbitals at, of shape
      (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
      number of electrons and the spin up electrons are ordered before the spin
      down electrons.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with matrices of orbitals for spin up and spin down electrons, with
    the same leading dimensions as in pos.
  """
  if not isinstance(pos, np.ndarray):  # works even with JAX array
    try:
      pos = pos.copy()
    except AttributeError as exc:
      raise ValueError('Input must be either NumPy or JAX array.') from exc
  leading_dims = pos.shape[:-1]
  # split into separate electrons
  pos = np.reshape(pos, [-1, 3])  # (batch*nelec, 3)
  mos = scf_approx.eval_mos(pos)  # (batch*nelec, nbasis), (batch*nelec, nbasis)
  # Reshape into (batch, nelec, nbasis) for each spin channel.
  mos = [np.reshape(mo, leading_dims + (sum(nspins), -1)) for mo in mos]
  # Return (using Aufbau principle) the matrices for the occupied alpha and
  # beta orbitals. Number of alpha electrons given by nspins[0].
  if use_geminal:
    # nb x nele0 x nmo
    alpha_spin = mos[0][..., :nspins[0], :]
    # nb x nele1 x nmo
    beta_spin = mos[1][..., nspins[0]:, :]
  else:
    alpha_spin = mos[0][..., :nspins[0], :nspins[0]]
    beta_spin = mos[1][..., nspins[0]:, :nspins[1]]
  return alpha_spin, beta_spin


def eval_slater(scf_approx: scf.Scf, pos: Union[jnp.ndarray, np.ndarray],
                nspins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates the Slater determinant.

  Args:
    scf_approx: an object that contains the result of a PySCF calculation.
    pos: an array of electron positions to evaluate the orbitals at.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with sign and log absolute value of Slater determinant.
  """
  matrices = eval_orbitals(scf_approx, pos, nspins)
  slogdets = [np.linalg.slogdet(elem) for elem in matrices]
  sign_alpha, sign_beta = [elem[0] for elem in slogdets]
  log_abs_wf_alpha, log_abs_wf_beta = [elem[1] for elem in slogdets]
  log_abs_slater_determinant = log_abs_wf_alpha + log_abs_wf_beta
  sign = sign_alpha * sign_beta
  return sign, log_abs_slater_determinant


def make_pretrain_step(batch_envelope_fn,
                       batch_orbitals: FermiNetOrbitals,
                       batch_network: networks.LogFermiNetLike,
                       optimizer_update: optax.TransformUpdateFn,
                       full_det: bool = False,
                       lattice: np.ndarray = None,
                       ):
  """Creates function for performing one step of Hartre-Fock pretraining.

  Args:
    batch_envelope_fn: callable with signature f(params, data) which, given a
      batch of electron positions and the tree of envelope network parameters,
      returns the multiplicative envelope to apply to the orbitals. See envelope
      functions in networks for details. Only required if the envelope is not
      included in batch_orbitals.
    batch_orbitals: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the orbitals in
      the network evaluated at those positions.
    batch_network: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the log of the
      magnitude of the (wavefunction) network  evaluated at those positions.
    optimizer_update: callable for transforming the gradients into an update (ie
      conforms to the optax API).
    full_det: If true, evaluate all electrons in a single determinant.
      Otherwise, evaluate products of alpha- and beta-spin determinants.

  Returns:
    Callable for performing a single pretraining optimisation step.
  """

  def pretrain_step(data, target, params, state, key, logprob):
    """One iteration of pretraining to match HF."""
    n = jnp.array([tgt.shape[-1] for tgt in target]).sum()

    def loss_fn(p, x, target):
      env = jnp.exp(batch_envelope_fn(p['envelope'], x) / n)
      env = jnp.reshape(env, [env.shape[-1], 1, 1, 1])
      if full_det:
        ndet = target[0].shape[0]
        na = target[0].shape[1]
        nb = target[1].shape[1]
        target = jnp.concatenate(
            (jnp.concatenate((target[0], jnp.zeros((ndet, na, nb))), axis=-1),
             jnp.concatenate((jnp.zeros((ndet, nb, na)), target[1]), axis=-1)),
            axis=-2)
        result = jnp.mean(
            (target[:, None, ...] - env * batch_orbitals(p, x)[0])**2)
      else:
        result = jnp.array([
            jnp.mean((t[:, None, ...] - env * o)**2)
            for t, o in zip(target, batch_orbitals(p, x))
        ]).sum()
      return constants.pmean(result)

    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, search_direction = val_and_grad(params, data, target)
    search_direction = constants.pmean(search_direction)
    updates, state = optimizer_update(search_direction, state, params)
    params = optax.apply_updates(params, updates)
    data, key, logprob, _ = mcmc.mh_update(params, batch_network, data, key,
                                           logprob, 0,
                                           lattice=lattice,
                                           )
    return data, params, state, loss_val, logprob

  return pretrain_step


def make_gemi_pretrain_step(batch_envelope_fn,
                       batch_orbitals: FermiNetOrbitals,
                       batch_network: networks.LogFermiNetLike,
                       optimizer_update: optax.TransformUpdateFn,
                       full_det: bool = True):
  """Creates function for performing one step of Hartre-Fock pretraining.

  Args:
    batch_envelope_fn: callable with signature f(params, data) which, given a
      batch of electron positions and the tree of envelope network parameters,
      returns the multiplicative envelope to apply to the orbitals. See envelope
      functions in networks for details. Only required if the envelope is not
      included in batch_orbitals.
    batch_orbitals: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the orbitals in
      the network evaluated at those positions.
    batch_network: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the log of the
      magnitude of the (wavefunction) network  evaluated at those positions.
    optimizer_update: callable for transforming the gradients into an update (ie
      conforms to the optax API).
    full_det: If true, evaluate all electrons in a single determinant.
      Otherwise, evaluate products of alpha- and beta-spin determinants.

  Returns:
    Callable for performing a single pretraining optimisation step.
  """
  del full_det

  def pretrain_step(data, target, params, state, key, logprob):

    def loss_fn(p, x, target):
      # nb x nele0/nele1 x nele
      gemi_a, gemi_b = target
      nb = gemi_a.shape[0]
      nmo = gemi_a.shape[-1]
      nele0 = gemi_a.shape[1]
      nele1 = gemi_b.shape[1]
      numb_pad = nele0 - nele1

      env = jnp.exp(batch_envelope_fn(p['envelope'], x) / nele0)
      env = jnp.reshape(env, [env.shape[-1], 1, 1, 1])
      # nb x nele0/nele1 x (nmo-pad_num)
      gemi_a_cut = jax.lax.dynamic_slice(gemi_a, (0,0,0), (nb, nele0, nmo-numb_pad))
      gemi_b_cut = jax.lax.dynamic_slice(gemi_b, (0,0,0), (nb, nele1, nmo-numb_pad))
      # nb x nele0 x nele1
      target_mat = jnp.einsum('ijk,ilk ->ijl', gemi_a_cut, gemi_b_cut)
      if numb_pad>0:
        # nb x nele0 x numb_pad
        pad_mat = jax.lax.dynamic_slice(gemi_a, (0,0,nmo-numb_pad), (nb, nele0, numb_pad))
        # nb x nele0 x nele0
        target_mat = jnp.concatenate([target_mat, pad_mat], axis = -1)

      # nb x ndet x nele0 x nele0
      result = jnp.mean(
          (target_mat[:, None, ...] - env * batch_orbitals(p, x)[0])**2)
      return constants.pmean(result)

    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, search_direction = val_and_grad(params, data, target)
    search_direction = constants.pmean(search_direction)
    updates, state = optimizer_update(search_direction, state, params)
    params = optax.apply_updates(params, updates)
    data, key, logprob, _ = mcmc.mh_update(params, batch_network, data, key,
                                           logprob, 0)
    return data, params, state, loss_val, logprob

  return pretrain_step

def pretrain_hartree_fock(
    *,
    params: networks.ParamTree,
    data: jnp.ndarray,
    batch_network: networks.FermiNetLike,
    batch_orbitals: FermiNetOrbitals,
    network_options: networks.FermiNetOptions,
    sharded_key: chex.PRNGKey,
    atoms: jnp.ndarray,
    electrons: Tuple[int, int],
    scf_approx: scf.Scf,
    iterations: int = 1000,
    logger: Optional[Callable[[int, float], None]] = None,
    use_geminal: bool = False,
    lattice : Optional[np.ndarray] = None,
    heg : Optional[bool] = False,
):
  """Performs training to match initialization as closely as possible to HF.

  Args:
    params: Network parameters.
    data: MCMC configurations.
    batch_network: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the log of the
      magnitude of the (wavefunction) network  evaluated at those positions.
    batch_orbitals: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the orbitals in
      the network evaluated at those positions.
    network_options: FermiNet network options.
    sharded_key: JAX RNG state (sharded) per device.
    atoms: (natom, 3) array of atom positions.
    electrons: tuple of number of electrons of each spin.
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    iterations: number of pretraining iterations to perform.
    logger: Callable with signature (step, value) which externally logs the
      pretraining loss.

  Returns:
    params, data: Updated network parameters and MCMC configurations such that
    the orbitals in the network closely match Hartree-Foch and the MCMC
    configurations are drawn from the log probability of the network.
  """
  # Pretraining is slow on larger systems (very low GPU utilization) because the
  # Hartree-Fock orbitals are evaluated on CPU and only on a single host.
  # Implementing the basis set in JAX would enable using GPUs and allow
  # eval_orbitals to be pmapped.

  optimizer = optax.adam(3.e-4)
  opt_state_pt = constants.pmap(optimizer.init)(params)

  if (network_options.envelope.apply_type ==
      envelopes.EnvelopeType.POST_DETERMINANT):

    def envelope_fn(params, x):
      ae, r_ae, _, r_ee = networks.construct_input_features(x, atoms)
      return network_options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params)
  else:
    envelope_fn = lambda p, x: 0.0
  batch_envelope_fn = jax.vmap(envelope_fn, (None, 0))

  if use_geminal:
    get_pretrain_step = make_gemi_pretrain_step
  else:
    get_pretrain_step = make_pretrain_step
  if heg :
    from ferminet.pbc.pretrain import make_eval_orbitals    
    pt_eval_orb = make_eval_orbitals(lattice, electrons, do_complex=False)
  else:
    pt_eval_orb = eval_orbitals

  pretrain_step = get_pretrain_step(
      batch_envelope_fn,
      batch_orbitals,
      batch_network,
      optimizer.update,
      full_det=network_options.full_det,
      lattice=lattice,
  )
  pretrain_step = constants.pmap(pretrain_step)
  pnetwork = constants.pmap(batch_network)
  logprob = 2. * pnetwork(params, data)

  for t in range(iterations):
    data_ = np.array(data, dtype=np.float64)
    target = pt_eval_orb(scf_approx, data_, electrons, use_geminal=use_geminal)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    data, params, opt_state_pt, loss, logprob = pretrain_step(
        data, target, params, opt_state_pt, subkeys, logprob)
    logging.info('Pretrain iter %05d: %g', t, loss[0])
    if logger:
      logger(t, loss[0])
  return params, data
