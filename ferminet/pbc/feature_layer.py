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

"""Feature layer for periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

from typing import Optional, Tuple
import jax.numpy as jnp

from ferminet import networks, dp
from ferminet.pbc import ds


def make_pbc_feature_layer(charges: Optional[jnp.ndarray] = None,
                           nspins: Optional[Tuple[int, ...]] = None,
                           ndim: int = 3,
                           lattice: Optional[jnp.ndarray] = None,
                           include_r_ae: bool = True) -> networks.FeatureLayer:
  """Returns the init and apply functions for periodic features.

  Args:
      charges: (natom) array of atom nuclear charges.
      nspins: tuple of the number of spin-up and spin-down electrons.
      ndim: dimension of the system.
      lattice: Matrix whose columns are the primitive lattice vectors of the
        system, shape (ndim, ndim).
      include_r_ae: Flag to enable electron-atom distance features. Set to False
        to avoid cusps with ghost atoms in, e.g., homogeneous electron gas.
  """

  del charges, nspins

  if lattice is None:
    lattice = jnp.eye(ndim)

  # Calculate reciprocal vectors, factor 2pi omitted
  reciprocal_vecs = jnp.linalg.inv(lattice)
  lattice_metric = lattice.T @ lattice

  def periodic_norm(vec, metric):
    a = (1 - jnp.cos(2 * jnp.pi * vec))
    b = jnp.sin(2 * jnp.pi * vec)
    # i,j = nelectron, natom for ae
    cos_term = jnp.einsum('ijm,mn,ijn->ij', a, metric, a)
    sin_term = jnp.einsum('ijm,mn,ijn->ij', b, metric, b)
    return (1 / (2 * jnp.pi)) * jnp.sqrt(cos_term + sin_term)

  def init() -> Tuple[Tuple[int, int], networks.Param]:
    if include_r_ae:
      return (2 * ndim + 1, 2 * ndim + 1), {}
    else:
      return (2 * ndim, 2 * ndim + 1), {}

  def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # One e features in phase coordinates, (s_ae)_i = k_i . ae
    s_ae = jnp.einsum('il,jkl->jki', reciprocal_vecs, ae)
    # Two e features in phase coordinates
    s_ee = jnp.einsum('il,jkl->jki', reciprocal_vecs, ee)
    # Periodized features
    ae = jnp.concatenate(
        (jnp.sin(2 * jnp.pi * s_ae), jnp.cos(2 * jnp.pi * s_ae)), axis=-1)
    ee = jnp.concatenate(
        (jnp.sin(2 * jnp.pi * s_ee), jnp.cos(2 * jnp.pi * s_ee)), axis=-1)
    # Distance features defined on orthonormal projections
    r_ae = periodic_norm(s_ae, lattice_metric)
    # Don't take gradients through |0|
    n = ee.shape[0]
    s_ee += jnp.eye(n)[..., None]
    r_ee = periodic_norm(s_ee, lattice_metric) * (1.0 - jnp.eye(n))

    if include_r_ae:
      ae_features = jnp.concatenate((r_ae[..., None], ae), axis=2)
    else:
      ae_features = ae
    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    ee_features = jnp.concatenate((r_ee[..., None], ee), axis=2)
    return ae_features, ee_features

  return networks.FeatureLayer(init=init, apply=apply)


def make_ferminet_decaying_features(
    charges: Optional[jnp.ndarray] = None,
    nspins: Optional[Tuple[int, ...]] = None,
    ndim: int = 3,
    rc : float = 3.0,
    rc_smth : float = 0.5,
    lr : Optional[bool] = False,
    lattice : Optional[jnp.ndarray] = None,
) -> networks.FeatureLayer:
  """Returns the init and apply functions for the decaying features."""

  if lr:
    pbc_feat = make_pbc_feature_layer(
      charges, nspins, ndim, 
      lattice=lattice, 
      include_r_ae=True,
    )

  def init() -> Tuple[Tuple[int, int], networks.Param]:
    if lr:
      return (pbc_feat.init()[0][0] + ndim + 1, pbc_feat.init()[0][1] + ndim + 1), {}
    else:
      return (ndim + 1, ndim + 1), {}

  def make_pref(rr):
    """
              sw(r)
    pref = -----------
             (r+1)^2
    """
    sw = dp.switch_func_poly(rr, rc, rc_smth)
    return sw / ((rr+1.0) * (rr+1.0))

  def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ae_pref = make_pref(r_ae)
    ee_pref = make_pref(r_ee)
    ae_features = jnp.concatenate((r_ae, ae), axis=2)
    ee_features = jnp.concatenate((r_ee, ee), axis=2)
    ae_features = ae_pref * ae_features
    ee_features = ee_pref * ee_features
    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    if lr:
      ae_pbc_feat, ee_pbc_feat = pbc_feat.apply(ae, r_ae, ee, r_ee)
      ae_features = jnp.concatenate([ae_features, ae_pbc_feat], axis=-1)
      ee_features = jnp.concatenate([ee_features, ee_pbc_feat], axis=-1)
    return ae_features, ee_features

  return networks.FeatureLayer(init=init, apply=apply)


def make_ds_features(
    charges: Optional[jnp.ndarray] = None,
    nspins: Optional[Tuple[int, ...]] = None,
    ndim: int = 3,
    lattice : Optional[jnp.ndarray] = None,
    has_cos : Optional[bool] = False,
    has_sym : Optional[bool] = False,
):
  if lattice is not None:
    org_lattice = lattice / (2. * jnp.pi)
    rec_lattice = jnp.linalg.inv(org_lattice)
    inv_lattice = jnp.linalg.inv(lattice)
  else :
    org_lattice = None
    rec_lattice = None
    inv_lattice = None

  def init() -> Tuple[Tuple[int, int], networks.Param]:
    dim0 = ndim + 1
    dim1 = ndim + 1
    if has_cos:
      dim0 += ndim
      dim1 += ndim
    if has_sym:
      dim0 += ndim
      dim1 += ndim
    return (dim0, dim1), {}

  def apply_(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
    del r_ae, r_ee
    
    n = ee.shape[0]
    prim_periodic_sea, prim_periodic_xea = ds.nu_distance(
      ae, org_lattice, rec_lattice, has_sym=has_sym)
    prim_periodic_sea = prim_periodic_sea[..., None]
    # different ee convention, so use -ee
    sim_periodic_see, sim_periodic_xee = ds.nu_distance(
      -ee + jnp.eye(n)[..., None], org_lattice, rec_lattice, has_sym=has_sym)
    sim_periodic_see = sim_periodic_see * (1.0 - jnp.eye(n))
    sim_periodic_see = sim_periodic_see[..., None]
    sim_periodic_xee = sim_periodic_xee * (1.0 - jnp.eye(n))[..., None]

    ae_features = jnp.concatenate([prim_periodic_sea, prim_periodic_xea], axis=-1)
    ee_features = jnp.concatenate([sim_periodic_see, sim_periodic_xee], axis=-1)

    if has_cos:
      def add_cos_feat(_ae_features, _ae):        
        s_ae = jnp.matmul(_ae, inv_lattice)
        cos__ae_feat = jnp.cos(2. * jnp.pi * s_ae)
        _ae_features = jnp.concatenate([_ae_features, cos__ae_feat], axis=-1)
        return _ae_features
      ae_features = add_cos_feat(ae_features, ae)
      ee_features = add_cos_feat(ee_features, ee)

    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    # return ae_features, ee_features, prim_periodic_sea
    return ae_features, ee_features

  return networks.FeatureLayer(init=init, apply=apply_)

