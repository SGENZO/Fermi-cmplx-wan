# Copyright 2022 DeepMind Technologies Limited.
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
# limitations under the License

"""Multiplicative envelopes appropriate for periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

import itertools
from typing import Mapping, Optional, Sequence, Tuple

import ferminet
from ferminet import envelopes
from ferminet.utils import scf

import jax
import jax.numpy as jnp


def make_multiwave_envelope(kpoints: jnp.ndarray) -> envelopes.Envelope:
  """Returns an oscillatory envelope.

  Envelope consists of a sum of truncated 3D Fourier series, one centered on
  each atom, with Fourier frequencies given by kpoints:

    sigma_{2i}*cos(kpoints_i.r_{ae}) + sigma_{2i+1}*sin(kpoints_i.r_{ae})

  Initialization sets the coefficient of the first term in each
  series to 1, and all other coefficients to 0. This corresponds to the
  cosine of the first entry in kpoints. If this is [0, 0, 0], the envelope
  will evaluate to unity at the beginning of training.

  Args:
    kpoints: Reciprocal lattice vectors of terms included in the Fourier
      series. Shape (nkpoints, ndim) (Note that ndim=3 is currently
      a hard-coded default).

  Returns:
    An instance of ferminet.envelopes.Envelope with apply_type
    envelopes.EnvelopeType.PRE_DETERMINANT
  """

  def init(natom: int,
           output_dims: Sequence[int],
           hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Sequence[Mapping[str, jnp.ndarray]]:
    """See ferminet.envelopes.EnvelopeInit."""
    del hf, natom, ndim  # unused
    params = []
    nk = kpoints.shape[0]
    for output_dim in output_dims:
      params.append({'sigma': jnp.zeros((2 * nk, output_dim))})
      params[-1]['sigma'] = params[-1]['sigma'].at[0, :].set(1.0)
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            sigma: jnp.ndarray) -> jnp.ndarray:
    """See ferminet.envelopes.EnvelopeApply."""
    del r_ae, r_ee  # unused
    phase_coords = ae @ kpoints.T
    waves = jnp.concatenate((jnp.cos(phase_coords), jnp.sin(phase_coords)),
                            axis=2)
    env = waves @ sigma
    return jnp.sum(env, axis=1)

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_kpoints(lattice: jnp.ndarray,
                 spins: Tuple[int, int],
                 min_kpoints: Optional[int] = None) -> jnp.ndarray:
  """Generates an array of reciprocal lattice vectors.

  Args:
    lattice: Matrix whose columns are the primitive lattice vectors of the
      system, shape (ndim, ndim). (Note that ndim=3 is currently
      a hard-coded default).
    spins: Tuple of the number of spin-up and spin-down electrons.
    min_kpoints: If specified, the number of kpoints which must be included in
      the output. The number of kpoints returned will be the
      first filled shell which is larger than this value. Defaults to None,
      which results in min_kpoints == sum(spins).

  Raises:
    ValueError: Fewer kpoints requested by min_kpoints than number of
      electrons in the system.

  Returns:
    jnp.ndarray, shape (nkpoints, ndim), an array of reciprocal lattice
      vectors sorted in ascending order according to length.
  """
  rec_lattice = 2 * jnp.pi * jnp.linalg.inv(lattice)
  # Calculate required no. of k points
  if min_kpoints is None:
    min_kpoints = sum(spins)
  elif min_kpoints < sum(spins):
    raise ValueError(
        'Number of kpoints must be equal or greater than number of electrons')

  dk = 1 + 1e-5
  # Generate ordinals of the lowest min_kpoints kpoints
  max_k = int(jnp.ceil(min_kpoints * dk)**(1 / 3.))
  ordinals = sorted(range(-max_k, max_k+1), key=abs)
  ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=3)))

  kpoints = ordinals @ rec_lattice.T
  kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
  k_norms = jnp.linalg.norm(kpoints, axis=1)

  return kpoints[k_norms <= k_norms[min_kpoints - 1] * dk]


def make_pbc_full(
    lattice: jnp.ndarray,
    rc : float,
    rc_smth : float,
    nearest_nei : bool = False,
) -> envelopes.Envelope:
  """Returns a full envelope for pbc systems.
  """
  shift_vec = ferminet.dp.compute_shift_vec(rc, lattice)
  nshift = shift_vec.shape[0]
  batch_apply_covar = jax.vmap(ferminet.envelopes._apply_covariance, in_axes=[0,None])
  batch_multiply = jax.vmap(jnp.multiply, in_axes=[0,None])
  if nearest_nei:
    vol = jnp.linalg.det(lattice)
    tofacedist = jnp.cross(lattice[[1,2,0],:], lattice[[2,0,1],:])
    tofacedist = vol * jnp.reciprocal(jnp.linalg.norm(tofacedist, axis=1))
    if 2. * rc > tofacedist[0] or 2. * rc > tofacedist[1] or 2. * rc > tofacedist[2]:
      raise RuntimeError('the rc should be no larger than half box length')

  def init(natom: int, output_dims: Sequence[int], hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Sequence[Mapping[str, jnp.ndarray]]:
    del hf  # unused
    eye = jnp.eye(ndim)
    params = []
    for output_dim in output_dims:
      params.append({
          'pi': jnp.ones(shape=(natom, output_dim)),
          'sigma': jnp.tile(eye[..., None, None], [1, 1, natom, output_dim])
      })
    return params

  def decayed_exp(rr):
    sw = ferminet.dp.switch_func_poly(rr, rc, rc_smth)
    exp = jnp.exp(-rr)
    return sw * exp

  def get_nearest(rr):
    # rr shape nnx3
    idx = jnp.argsort(jnp.linalg.norm(rr, axis=1))
    idx_nei = jnp.split(idx, [1])[0]
    return rr[idx_nei,:]

  def find_nearest(ae_):
    (ns, nele, nion, _) = ae_.shape
    # (nele x nion) x ns x 3    
    ae = ae_.reshape((ns, nele*nion, 3)).transpose((1, 0, 2))
    # (nele x nion) x 1 x 3
    ae = jax.vmap(get_nearest, in_axes=[0])(ae)
    # 1 x nele x nion x 3
    ae = ae.transpose((1, 0, 2)).reshape((1, nele, nion, 3))
    return ae

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes a fully anisotropic exponentially-decaying envelope."""
    del r_ae, r_ee  # unused
    nele = ae.shape[0]
    nion = ae.shape[1]
    orb_dim = sigma.shape[-1]
    # nshift x nele x nion x 3
    bg_ae = ferminet.dp.compute_background_coords(ae.reshape([-1,3]), lattice, shift_vec)\
      .reshape([nshift, nele, nion, 3])
    if nearest_nei:
      bg_ae = find_nearest(bg_ae)
      ns = 1
    else:
      ns = nshift
    # ns x nele x nion x 3 x orb_dim
    ae_sigma = batch_apply_covar(bg_ae, sigma)    
    bg_ae = jnp.reshape(bg_ae, [ns*nele, nion, 3])
    ae_sigma = jnp.reshape(ae_sigma, [ns*nele, nion, 3, orb_dim])
    # (ns x nele) x nion x 3 x orb_dim
    ae_sigma = ferminet.curvature_tags_and_blocks.register_qmc(
        ae_sigma, bg_ae, sigma, type='full')
    # (ns x nele) x nion x orb_dim
    dr_ae_sigma = decayed_exp(jnp.linalg.norm(ae_sigma, axis=2))
    # (ns x nele) x nion x orb_dim
    pi_dr_ae_sigma = batch_multiply(dr_ae_sigma, pi)
    # nele x (ns x nion) x orb_dim
    pi_dr_ae_sigma = pi_dr_ae_sigma.reshape([ns, nele, nion, orb_dim])\
                                   .transpose((1, 0, 2, 3))\
                                   .reshape([nele, ns*nion, orb_dim])
    # # nele x orb_dim
    return jnp.sum(pi_dr_ae_sigma, axis=1)

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)



def make_pbc_full_nn(
    lattice: jnp.ndarray,
    rc : float,
    rc_smth : float,
) -> envelopes.Envelope:
  """Returns a full envelope for pbc systems. only search for the nearest neighbors
  """
  shift_vec = ferminet.dp.compute_shift_vec(rc, lattice)
  nshift = shift_vec.shape[0]
  batch_apply_covar = jax.vmap(ferminet.envelopes._apply_covariance, in_axes=[0,None])
  batch_multiply = jax.vmap(jnp.multiply, in_axes=[0,None])
  rec_lattice = jnp.linalg.inv(lattice)
  if not ferminet.dp.auto_nearest_neighbor(lattice, rc):
    raise RuntimeError('the rc should be no larger than half box length')

  def init(natom: int, output_dims: Sequence[int], hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Sequence[Mapping[str, jnp.ndarray]]:
    del hf  # unused
    eye = jnp.eye(ndim)
    params = []
    for output_dim in output_dims:
      params.append({
          'pi': jnp.ones(shape=(natom, output_dim)),
          'sigma': jnp.tile(eye[..., None, None], [1, 1, natom, output_dim])
      })
    return params

  def decayed_exp(xx, rr):
    sw = ferminet.dp.switch_func_poly(rr, rc, rc_smth)
    exp = jnp.exp(-xx)
    return sw * exp

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes a fully anisotropic exponentially-decaying envelope."""
    del r_ae, r_ee  # unused
    # nele x nion x 1
    ae = ferminet.dp.apply_nearest_neighbor(ae, lattice, rec_lattice)
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    ae_sigma = ferminet.envelopes._apply_covariance(ae, sigma)
    # nele x nion x 3 x orb_dim
    ae_sigma = ferminet.curvature_tags_and_blocks.register_qmc(
        ae_sigma, ae, sigma, type='full')
    # nele x nion x orb_dim
    r_ae_sigma = jnp.linalg.norm(ae_sigma, axis=2)
    # # nele x orb_dim
    return jnp.sum(decayed_exp(r_ae_sigma, r_ae) * pi, axis=1)

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)
