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

"""Tests for ferminet.networks."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from ferminet import envelopes
from ferminet import networks
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
# enable double if the precision is not enough
# from jax.config import config
# config.update ('jax_disable_jit', True)
import ferminet
from ferminet.mcmc import (
  apply_pbc,
)
from ferminet.dp import (
  compute_ncopy,
  compute_copy_idx,
  compute_shift_vec,
  compute_background_coords,
)
from ferminet.pretrain import (
  eval_orbitals,
)
import ferminet.pbc.envelopes
import ferminet.pbc.feature_layer

def rand_default():
  randn = np.random.RandomState(0).randn
  def generator(shape, dtype):
    return randn(*shape).astype(dtype)
  return generator


def _antisymmtry_options():
  for envelope,geminal in itertools.product(envelopes.EnvelopeLabel, [True,False]):
    # skip unsupported envelopes
    if envelope == envelopes.EnvelopeLabel.STO or\
       envelope == envelopes.EnvelopeLabel.STO_POLY:
      continue
    yield {
        'testcase_name': f'_dtype={dtype}',
        'dtype': np.float32,
    }

class DPOptions:
  type_embd = True
  cntr_type = False
  n_conv_layers = 4
  nfeat = 8
  nodim = 8
  embd = [4]
  fitt = [4]
  feat = [4]
  power = 1.0
  rinv_shift = 1.0
  geminal = False
  avg_env_mat = [0.5, 0., 0., 0.]
  std_env_mat = [4., 5., 5., 5.]
  onehot_scale = 1.0
  cat_sum_embd = True
  nele_sel = 5
  nion_sel = 5
  rc_mode = 'poly'
  rc_cntr = 3.9
  rc_sprd = 0.2
  only_ele = True
  use_ae_feat = False
  def __init__(self, rc = 8.):
    rc_cntr = rc
  nearest_neighbor = None

def _network_options():
  """Yields the set of all combinations of options to pass into test_fermi_net.

  Example output:
  {
    'vmap': True,
    'envelope': envelopes.EnvelopeLabel.ISOTROPIC,
    'bias_orbitals': False,
    'full_det': True,
    'use_last_layer': False,
    'hidden_dims': ((32, 8), (32, 8)),
  }
  """
  # Key for each option and corresponding values to test.
  all_options = {
      'vmap': [True, False],
      'bias_orbitals': [True, False],
      'full_det': [True, False],
      'use_last_layer': [False],
      'hidden_dims' : [DPOptions(rc=3.9), DPOptions(rc=6.), DPOptions(rc=8.)],
  }
  # Create the product of all options.
  for options in itertools.product(*all_options.values()):
    # Yield dict of the current combination of options.
    yield dict(zip(all_options.keys(), options))


class TestPBC(parameterized.TestCase):
  def test_apply_pbc(self):
    lattice = np.array([[2., 0., 0.],
                        [0.5, np.sqrt(3.)*0.5, 0.],
                        [0., 0., 3.], ])
    xx = jnp.array([[-5., -4., -9.],
                    [ 1., -4.,  8.],])
    xx = apply_pbc(xx, lattice)
    expected_xx = jnp.array([[1.5       , 0.33012702, 0.        ],
                             [1.5       , 0.33012702, 2.        ],])
    np.testing.assert_allclose(xx, expected_xx, rtol=1e-5, atol=1e-5)

  def test_compute_ncopy(self):
    rc=6.
    lattice = np.array([[2., 0., 0.],
                        [0.5, np.sqrt(3.)*0.5, 0.],
                        [0., 0., 3.], ])
    expected_ncopy = np.array([4, 7, 3])
    ncopy = compute_ncopy(rc, lattice)
    np.testing.assert_equal(ncopy, expected_ncopy)

  def _make_set(self, l0):
    set0 = set()
    for ii in l0:
      set0.add(tuple(ii))
    return set0

  def test_compute_copy_idx(self):
    rc=6.
    lattice = 6.1 * np.eye(3)
    idx = compute_copy_idx(rc, lattice)
    idx = [list(ii) for ii in list(idx)]
    expected_idx = [
      [0, 0, 0],
      [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], 
      [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
      [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1], 
      [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
      [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], 
      [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],]    
    self.assertEqual(len(idx), 27)
    self.assertEqual(self._make_set(idx[:1]), self._make_set(expected_idx[:1]))
    self.assertEqual(self._make_set(idx[1:7]), self._make_set(expected_idx[1:7]))
    self.assertEqual(self._make_set(idx[7:19]), self._make_set(expected_idx[7:19]))
    self.assertEqual(self._make_set(idx[19:27]), self._make_set(expected_idx[19:27]))

  def test_compute_background_coord(self):
    rc=6.
    lattice = 6.0 * np.eye(3)
    lattice[0][0] = 6.1
    lattice[1][1] = 6.2
    lattice[2][2] = 6.3
    xx = jnp.array([-12.2, 0., 0.])
    xx = apply_pbc(xx, lattice)
    ss = compute_shift_vec(rc, lattice)
    bk_xx = compute_background_coords(xx, lattice, ss)
    bk_xx = np.reshape(bk_xx, [-1,3])
    bk_xx = [list(np.array(ii)) for ii in list(bk_xx)]
    expected_bk_xx = [
      [0., 0., 0.],
      [6.1, 0., 0.], [-6.1, 0., 0.],
      [0., 6.2, 0.], [0., -6.2, 0.],
      [0., 0., 6.3], [0., 0., -6.3],
      [6.1, 6.2, 0.], [6.1, -6.2, 0.], [-6.1, 6.2, 0.], [-6.1, -6.2, 0.], 
      [6.1, 0., 6.3], [6.1, 0., -6.3], [-6.1, 0., 6.3], [-6.1, 0., -6.3],
      [0., 6.2, 6.3], [0., 6.2, -6.3], [0., -6.2, 6.3], [0., -6.2, -6.3], 
      [6.1, 6.2, 6.3], [6.1, 6.2, -6.3], [6.1, -6.2, 6.3], [6.1, -6.2, -6.3], 
      [-6.1, 6.2, 6.3], [-6.1, 6.2, -6.3], [-6.1, -6.2, 6.3], [-6.1, -6.2, -6.3], 
    ]
    # convert floats to ints, so the set of tuples are comparable
    bk_xx = [ [int(100*(jj+1e-5)) for jj in ii] for ii in bk_xx]
    expected_bk_xx = [ [int(100*(jj+1e-5)) for jj in ii] for ii in expected_bk_xx]
    self.assertEqual(len(bk_xx), 27)
    self.assertEqual(len(expected_bk_xx), 27)
    self.assertEqual(self._make_set(bk_xx[:1]), self._make_set(expected_bk_xx[:1]))
    self.assertEqual(self._make_set(bk_xx[1:7]), self._make_set(expected_bk_xx[1:7]))
    self.assertEqual(self._make_set(bk_xx[7:19]), self._make_set(expected_bk_xx[7:19]))
    self.assertEqual(self._make_set(bk_xx[19:27]), self._make_set(expected_bk_xx[19:27]))


  def test_compute_background_coord_2(self):
    rc=6.
    lattice = 6.0 * np.eye(3)
    lattice[0][0] = 6.1
    lattice[1][1] = 6.2
    lattice[2][2] = 6.3
    xx = jnp.array([-12.2, 0., 0., -12.2, 0., 0.1])
    xx = apply_pbc(xx, lattice)
    ss = compute_shift_vec(rc, lattice)
    bk_xx = compute_background_coords(xx, lattice, ss)
    bk_xx = np.reshape(bk_xx, [-1,6])
    bk_xx = [list(np.array(ii)) for ii in list(bk_xx)]
    expected_bk_xx_0 = np.asarray([
      [0., 0., 0.],
      [6.1, 0., 0.], [-6.1, 0., 0.],
      [0., 6.2, 0.], [0., -6.2, 0.],
      [0., 0., 6.3], [0., 0., -6.3],
      [6.1, 6.2, 0.], [6.1, -6.2, 0.], [-6.1, 6.2, 0.], [-6.1, -6.2, 0.], 
      [6.1, 0., 6.3], [6.1, 0., -6.3], [-6.1, 0., 6.3], [-6.1, 0., -6.3],
      [0., 6.2, 6.3], [0., 6.2, -6.3], [0., -6.2, 6.3], [0., -6.2, -6.3], 
      [6.1, 6.2, 6.3], [6.1, 6.2, -6.3], [6.1, -6.2, 6.3], [6.1, -6.2, -6.3], 
      [-6.1, 6.2, 6.3], [-6.1, 6.2, -6.3], [-6.1, -6.2, 6.3], [-6.1, -6.2, -6.3], 
    ])
    expected_bk_xx_1 = expected_bk_xx_0.copy()
    expected_bk_xx_1 = expected_bk_xx_1 + np.asarray([0., 0., .1])
    expected_bk_xx_0 = expected_bk_xx_0.reshape([-1, 1, 3])
    expected_bk_xx_1 = expected_bk_xx_1.reshape([-1, 1, 3])
    expected_bk_xx = np.concatenate([expected_bk_xx_0, expected_bk_xx_1], axis=1).reshape([-1,6])
    expected_bk_xx = [list(np.array(ii)) for ii in list(expected_bk_xx)]
    # convert floats to ints, so the set of tuples are comparable
    bk_xx = [ [int(100*(jj+1e-5)) for jj in ii] for ii in bk_xx]
    expected_bk_xx = [ [int(100*(jj+1e-5)) for jj in ii] for ii in expected_bk_xx]
    self.assertEqual(len(bk_xx), 27)
    self.assertEqual(len(expected_bk_xx), 27)
    self.assertEqual(self._make_set(bk_xx[:1]), self._make_set(expected_bk_xx[:1]))
    self.assertEqual(self._make_set(bk_xx[1:7]), self._make_set(expected_bk_xx[1:7]))
    self.assertEqual(self._make_set(bk_xx[7:19]), self._make_set(expected_bk_xx[7:19]))
    self.assertEqual(self._make_set(bk_xx[19:27]), self._make_set(expected_bk_xx[19:27]))

  def test_compute_rij_nnei(self):
    rc=3.0
    lattice = 6.0 * np.eye(3)
    lattice[0][0] = 6.1
    lattice[1][1] = 6.2
    lattice[2][2] = 6.3
    ss = compute_shift_vec(rc, lattice)
    xx = jnp.array([0., 0., 0., 0., 0., 0.1]).reshape([-1,3])
    yy = jnp.array([0., 6.1, 0., 0., 0., 6.2]).reshape([-1,3])
    diffnn = ferminet.dp.compute_rij_2_nnei(xx, yy, lattice, jnp.linalg.inv(lattice))
    expected_diff = jnp.array([
      [[0., -0.1, 0.], [0., -0.0, -0.1]], 
      [[0., -0.1, -0.1], [0., 0., -0.2]], 
    ])
    np.testing.assert_allclose(diffnn, expected_diff, rtol=1e-5, atol=1e-5)

  def test_apply_nearest_neighbor(self):
    lattice = 6.0 * np.eye(3)
    lattice[0][0] = 6.1
    lattice[1][1] = 6.2
    lattice[2][2] = 6.3
    self.assertTrue(ferminet.dp.auto_nearest_neighbor(lattice, 3.))
    self.assertFalse(ferminet.dp.auto_nearest_neighbor(lattice, 4.))


def make_atoms(
    nh,
    bond_length,
):
  """make atom pyscf style coords
  """
  atom_strs = [ f'H {ii*bond_length} 0. 0.' for ii in range(nh) ]
  return ';'.join(atom_strs)

def make_cell(
    spins,
    bond_length,
    basis = 'sto-3g', # 'ccpvdz'
):
  nh = sum(spins)
  from pyscf.pbc import gto
  cell = gto.Cell()
  cell.atom = make_atoms(nh, bond_length)
  cell.basis = basis
  cell.a = np.eye(3) * 100
  cell.a[0][0] = bond_length * nh
  cell.unit = "B"
  cell.spin = spins[0] - spins[1]
  cell.verbose = 0
  cell.exp_to_discard = 0.1
  cell.build()
  return cell


class TestPretrain(parameterized.TestCase):
  def test_eval_pretrain_target(self):
    cell = make_cell((1,1), 2.0, basis='sto-3g')
    scf_approx = ferminet.pbc.scf.Scf(cell, twist=np.zeros(3), restricted=False)
    scf_approx.run()
    xx = np.arange(12, dtype=np.float64).reshape([2,6])
    target = eval_orbitals(scf_approx, xx, cell.nelec)
    expected_0 = np.asarray([4.39782949e-02, 4.48837430e-10])
    expected_1 = np.asarray([8.55188828e-05, 5.39668697e-18])
    # do compare
    self.assertEqual(target[0].shape, (2, 1, 1))
    self.assertEqual(target[1].shape, (2, 1, 1))
    np.testing.assert_allclose(target[0].reshape([-1]), expected_0, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(target[1].reshape([-1]), expected_1, atol=1e-8, rtol=1e-6)


def set_fixed_parameters(params, hidden_dims):
  for kk in ['embd', 'feat']:
    for ll_idx in range(hidden_dims.n_conv_layers):
      for ii in range(len(params[kk][ll_idx])):
        keys = params[kk][ll_idx][ii].keys()
        for sub_kk in keys:
          # sub_kk = ['w', 'b']
          tmp_shape = params[kk][ll_idx][ii][sub_kk].shape
          tmp_len = np.prod(np.array(tmp_shape))
          params[kk][ll_idx][ii][sub_kk] = \
                  jnp.reshape(jnp.array([(j+1)*0.001*(ii+1)*(ll_idx+1) for j in range(tmp_len)]),  tmp_shape)
  for kk in ['orbital', 'fitt']:
    norb = len(params[kk])
    for jj in range(norb):
      ob_keys = params[kk][jj].keys()
      for sub_kk in ob_keys:
        tmp_shape = params[kk][jj][sub_kk].shape
        tmp_len = np.prod(np.array(tmp_shape))
        params[kk][jj][sub_kk] = \
                jnp.reshape(jnp.array([j*0.001*(jj+1) for j in range(tmp_len)]),  tmp_shape)
  for i in range(len(params['envelope'])):
    if params['envelope'][i]:
      params['envelope'][i]['sigma'] = jnp.ones(
        params['envelope'][i]['sigma'].shape)
  return params


class TestNetworkPBC(parameterized.TestCase):
  @parameterized.parameters(_network_options())
  def test_pbc(self, vmap, **network_options):
    key = random.PRNGKey(42)
    nspins = (2, 1)
    atoms = jnp.asarray([[0, 0, 0]], dtype=jnp.float32)
    charges = jnp.asarray([0.0], dtype=jnp.float32)
    lattice = jnp.asarray([[4., 0, 0], [0, 5., 0], [0, 0, 6.]], dtype=jnp.float32)
    kpoints = ferminet.pbc.envelopes.make_kpoints(lattice, nspins)
    feature_layer = ferminet.pbc.feature_layer.make_pbc_feature_layer(
      charges,
      nspins,
      3,
      lattice=lattice,
      include_r_ae=False,
    )
    envelope = ferminet.pbc.envelopes.make_multiwave_envelope(
      kpoints=kpoints,
    )
    
    init, fermi_net, _ = networks.make_fermi_net(
        atoms,
        nspins,
        charges,
        envelope=envelope,
        feature_layer=feature_layer,
        bias_orbitals=network_options['bias_orbitals'],
        use_last_layer=network_options['use_last_layer'],
        full_det=network_options['full_det'],
        determinants=4,
        hidden_dims=network_options['hidden_dims'],
        lattice=lattice,
    )
    fermi_net = jax.vmap(fermi_net, in_axes=(None, 0))

    key, subkey = jax.random.split(key)
    # 0 0 0
    xs0 = jnp.array([[0., 0., 0., 1., 0., 0., 2., 0., 0.]])
    shift = jnp.array([[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 1]])
    nshift = shift.shape[0]
    shift_vec = jnp.matmul(shift, lattice.T)    
    shift_vec = jnp.tile(shift_vec, [1, 3])
    xs1 = jnp.tile(xs0, [nshift, 1]) + shift_vec

    params = init(subkey)
    sign_psi0, log_psi0 = fermi_net(params, xs0)
    sign_psi1, log_psi1 = fermi_net(params, xs1)
    
    log_psi0 = jnp.tile(log_psi0, [nshift])
    np.testing.assert_allclose(log_psi0, log_psi1, atol=1e-5, rtol=1e-5)


  def test_value_heg_net(self):
    from jax.config import config
    config.update("jax_enable_x64", True) 
    network_options = {
        'bias_orbitals': False,
        'full_det': True,
        'use_last_layer': False,
        'hidden_dims' : DPOptions(rc=5.),
    }
    key = random.PRNGKey(42)
    nspins = (2, 2)
    atoms = jnp.asarray([[0, 0, 0],], dtype=jnp.float32)
    charges = jnp.asarray([0.0], dtype=jnp.float32)
    lattice = jnp.asarray([[4., 0, 0], [0, 5., 0], [0, 0, 6.]], dtype=jnp.float32)
    kpoints = ferminet.pbc.envelopes.make_kpoints(lattice, nspins)
    feature_layer = ferminet.pbc.feature_layer.make_pbc_feature_layer(
      charges,
      nspins,
      3,
      lattice=lattice,
      include_r_ae=False,
    )
    envelope = ferminet.pbc.envelopes.make_multiwave_envelope(
      kpoints=kpoints,
    )
    
    init, fermi_net, _ = networks.make_fermi_net(
        atoms,
        nspins,
        charges,
        envelope=envelope,
        feature_layer=feature_layer,
        bias_orbitals=network_options['bias_orbitals'],
        use_last_layer=network_options['use_last_layer'],
        full_det=network_options['full_det'],
        determinants=4,
        hidden_dims=network_options['hidden_dims'],
        lattice=lattice,
    )
    fermi_net = jax.vmap(fermi_net, in_axes=(None, 0))

    key, subkey = jax.random.split(key)
    # 0 0 0
    xs0 = jnp.array([
      [0., 0., 0.,  2., 0., 0.,  0., 4., 0.,  1.5, 3., 0.,],
      [0., 0., 0.,  2., 0., 0.,  0., 4., 0.,  1.5, 3., 0.,],
    ])

    params = set_fixed_parameters(init(subkey), network_options['hidden_dims'])
    sign_psi0, log_psi0 = fermi_net(params, xs0)
    
    np.testing.assert_allclose(log_psi0, [-94.5753, -94.5753], atol=1e-5, rtol=1e-5)


  def test_value_be_net(self):
    from jax.config import config
    config.update("jax_enable_x64", True) 
    network_options = {
        'bias_orbitals': False,
        'full_det': True,
        'use_last_layer': False,
        'hidden_dims' : DPOptions(rc=5.),
    }
    key = random.PRNGKey(42)
    nspins = (2, 2)
    atoms = jnp.asarray([[0, 0, 0.5],], dtype=jnp.float32)
    charges = jnp.asarray([4.0], dtype=jnp.float32)
    lattice = jnp.asarray([[4., 0, 0], [0, 5., 0], [0, 0, 6.]], dtype=jnp.float32)
    kpoints = ferminet.pbc.envelopes.make_kpoints(lattice, nspins)
    feature_layer = ferminet.pbc.feature_layer.make_pbc_feature_layer(
      charges,
      nspins,
      3,
      lattice=lattice,
      include_r_ae=False,
    )
    envelope = ferminet.pbc.envelopes.make_multiwave_envelope(
      kpoints=kpoints,
    )
    
    init, fermi_net, _ = networks.make_fermi_net(
        atoms,
        nspins,
        charges,
        envelope=envelope,
        feature_layer=feature_layer,
        bias_orbitals=network_options['bias_orbitals'],
        use_last_layer=network_options['use_last_layer'],
        full_det=network_options['full_det'],
        determinants=4,
        hidden_dims=network_options['hidden_dims'],
        lattice=lattice,
    )
    fermi_net = jax.vmap(fermi_net, in_axes=(None, 0))

    key, subkey = jax.random.split(key)
    # 0 0 0
    xs0 = jnp.array([
      [0., 0., 0.,  2., 0., 0.,  0., 4., 0.,  1.5, 3., 0.,],
      [0., 0., 0.,  2., 0., 0.,  0., 4., 0.,  1.5, 3., 0.,],
    ])

    params = set_fixed_parameters(init(subkey), network_options['hidden_dims'])
    sign_psi0, log_psi0 = fermi_net(params, xs0)
    
    np.testing.assert_allclose(log_psi0, [-95.436053, -95.436053], atol=1e-5, rtol=1e-5)


class TestPBCEnvelope(parameterized.TestCase):
  def test_pbc_full_envelope_close(self):
    lattice = jnp.eye(3) * 100
    rc = 10.
    rc_smth = 2.
    e_pbc_full = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth,
    )
    e_pbc_full_n = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth, nearest_nei=True,
    )
    e_pbc_full_nn = ferminet.pbc.envelopes.make_pbc_full_nn(
      lattice, rc, rc_smth,
    )
    e_full = ferminet.envelopes.make_full_envelope()
    
    # nele = 2  nion = 1
    nele = 2
    nion = 1
    odim = [4]
    p_pbc_full = e_pbc_full.init(nion, odim)
    p_full = e_full.init(nion, odim)

    ae = jnp.array(
      [[2., 0., 0.],
       [0., 0., 2.],]
    ).reshape([nele, nion, 3])
    r_ae = jnp.linalg.norm(ae, axis=2)

    ret_pbc = e_pbc_full.apply(
      ae=ae, 
      r_ae=r_ae,
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_n = e_pbc_full_n.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_nn = e_pbc_full_nn.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret = e_full.apply(
      ae=ae,
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_full[0]['pi'], 
      sigma=p_full[0]['sigma'],
    )
    np.testing.assert_allclose(ret_pbc, ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ret_pbc_n, ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ret_pbc_nn, ret, atol=1e-5, rtol=1e-5)

  def test_pbc_full_envelope_pbc_close(self):
    lattice = jnp.eye(3) * 100
    rec_lattice = jnp.linalg.inv(lattice)
    rc = 10.
    rc_smth = 2.
    e_pbc_full = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth,
    )
    e_pbc_full_n = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth, nearest_nei=True,
    )
    e_pbc_full_nn = ferminet.pbc.envelopes.make_pbc_full_nn(
      lattice, rc, rc_smth,
    )
    e_full = ferminet.envelopes.make_full_envelope()
    
    # nele = 2  nion = 1
    nele = 2
    nion = 1
    odim = [4]
    p_pbc_full = e_pbc_full.init(nion, odim)
    p_full = e_full.init(nion, odim)

    ae_pbc = jnp.array(
      [[-98., 0., 0.],
       [0., 0., 102.],]
    ).reshape([nele, nion, 3])
    r_ae_pbc = jnp.linalg.norm(
      ferminet.dp.apply_nearest_neighbor(ae_pbc, lattice, rec_lattice),
      axis=2,
    )
    ae = jnp.array(
      [[2., 0., 0.],
       [0., 0., 2.],]
    ).reshape([nele, nion, 3])
    r_ae = jnp.linalg.norm(ae, axis=2)

    ret_pbc = e_pbc_full.apply(
      ae=ae_pbc, 
      r_ae=r_ae_pbc, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_n = e_pbc_full_n.apply(
      ae=ae_pbc, 
      r_ae=r_ae_pbc, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_nn = e_pbc_full_nn.apply(
      ae=ae_pbc, 
      r_ae=r_ae_pbc, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret = e_full.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_full[0]['pi'], 
      sigma=p_full[0]['sigma'],
    )
    np.testing.assert_allclose(ret_pbc, ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ret_pbc_n, ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ret_pbc_nn, ret, atol=1e-5, rtol=1e-5)

  def test_pbc_full_envelope_mid(self):
    lattice = jnp.eye(3) * 100
    rc = 5.
    rc_smth = 3.
    e_pbc_full = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth,
    )
    e_pbc_full_n = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth, nearest_nei=True,
    )
    e_pbc_full_nn = ferminet.pbc.envelopes.make_pbc_full_nn(
      lattice, rc, rc_smth,
    )
    e_full = ferminet.envelopes.make_full_envelope()
    
    # nele = 2  nion = 1
    nele = 2
    nion = 1
    odim = [4]
    p_pbc_full = e_pbc_full.init(nion, odim)
    p_full = e_full.init(nion, odim)

    ae = jnp.array(
      [[4., 0., 0.],
       [0., 0., 4.],]
    ).reshape([nele, nion, 3])
    r_ae = jnp.linalg.norm(ae, axis=2)

    ret_pbc = e_pbc_full.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_n = e_pbc_full_n.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_nn = e_pbc_full_nn.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret = e_full.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_full[0]['pi'], 
      sigma=p_full[0]['sigma'],
    )
    np.testing.assert_allclose(2.0 * ret_pbc, ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(2.0 * ret_pbc_n, ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(2.0 * ret_pbc_nn, ret, atol=1e-5, rtol=1e-5)

  def test_pbc_full_envelope_75(self):
    lattice = jnp.eye(3) * 100
    rc = 5.
    rc_smth = 3.
    e_pbc_full = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth,
    )
    e_pbc_full_n = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth, nearest_nei=True,
    )
    e_pbc_full_nn = ferminet.pbc.envelopes.make_pbc_full_nn(
      lattice, rc, rc_smth,
    )
    e_full = ferminet.envelopes.make_full_envelope()
    
    # nele = 2  nion = 1
    nele = 2
    nion = 1
    odim = [4]
    p_pbc_full = e_pbc_full.init(nion, odim)
    p_full = e_full.init(nion, odim)

    ae = jnp.array(
      [[4.5, 0., 0.],
       [0., 0., 4.5],]
    ).reshape([nele, nion, 3])
    r_ae = jnp.linalg.norm(ae, axis=2)

    ret_pbc = e_pbc_full.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_n = e_pbc_full_n.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_nn = e_pbc_full_nn.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret = e_full.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_full[0]['pi'], 
      sigma=p_full[0]['sigma'],
    )
    # uu=0.75    1./(uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1)
    np.testing.assert_allclose(9.660377358490566 * ret_pbc, ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(9.660377358490566 * ret_pbc_n, ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(9.660377358490566 * ret_pbc_nn, ret, atol=1e-5, rtol=1e-5)

  def test_pbc_full_envelope_75_from_input_feature(self):
    lattice = jnp.eye(3) * 100
    rec_lattice = jnp.linalg.inv(lattice)
    rc = 5.
    rc_smth = 3.
    e_pbc_full = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth,
    )
    e_pbc_full_n = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth, nearest_nei=True,
    )
    e_pbc_full_nn = ferminet.pbc.envelopes.make_pbc_full_nn(
      lattice, rc, rc_smth,
    )
    e_full = ferminet.envelopes.make_full_envelope()
    
    # nele = 2  nion = 1
    nele = 2
    nion = 1
    odim = [4]
    p_pbc_full = e_pbc_full.init(nion, odim)
    p_full = e_full.init(nion, odim)

    pos = jnp.array(
      [[4.5, 0., 0.],
       [0., 0., 4.5],]
    ).reshape([nele*3])
    atoms = jnp.zeros([3]).reshape([nion, 3])    
    ae, ee, r_ae, r_ee = ferminet.dp.construct_input_features(pos, atoms)
    ret = e_full.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_full[0]['pi'], 
      sigma=p_full[0]['sigma'],
    )
    pos = jnp.array(
      [[100-4.5, 0., 0.],
       [0., 0., 4.5+100],]
    ).reshape([nele*3])
    atoms = jnp.zeros([3]).reshape([nion, 3])    
    ae, ee, r_ae, r_ee = ferminet.dp.construct_input_features(pos, atoms)
    r_ae = jnp.linalg.norm(
      ferminet.dp.apply_nearest_neighbor(ae, lattice, rec_lattice),
      axis=2,
    )    

    ret_pbc = e_pbc_full.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_n = e_pbc_full_n.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_nn = e_pbc_full_nn.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    # uu=0.75    1./(uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1)
    np.testing.assert_allclose(9.660377358490566 * ret_pbc, ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(9.660377358490566 * ret_pbc_n, ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(9.660377358490566 * ret_pbc_nn, ret, atol=1e-5, rtol=1e-5)


  def test_pbc_full_envelope_far(self):
    lattice = jnp.eye(3) * 100
    rc = 4.
    rc_smth = 3.
    e_pbc_full = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth,
    )
    e_pbc_full_n = ferminet.pbc.envelopes.make_pbc_full(
      lattice, rc, rc_smth, nearest_nei=True,
    )
    e_pbc_full_nn = ferminet.pbc.envelopes.make_pbc_full_nn(
      lattice, rc, rc_smth,
    )
    e_full = ferminet.envelopes.make_full_envelope()
    
    # nele = 2  nion = 1
    nele = 2
    nion = 1
    odim = [4]
    p_pbc_full = e_pbc_full.init(nion, odim)
    p_full = e_full.init(nion, odim)

    ae = jnp.array(
      [[4., 0., 0.],
       [0., 0., 4.],]
    ).reshape([nele, nion, 3])
    r_ae = jnp.linalg.norm(ae, axis=2)

    ret_pbc = e_pbc_full.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_n = e_pbc_full_n.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret_pbc_nn = e_pbc_full_nn.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_pbc_full[0]['pi'], 
      sigma=p_pbc_full[0]['sigma'],
    )
    ret = e_full.apply(
      ae=ae, 
      r_ae=r_ae, 
      r_ee=None, 
      pi=p_full[0]['pi'], 
      sigma=p_full[0]['sigma'],
    )
    np.testing.assert_allclose(ret_pbc, 0.0 * ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ret_pbc_n, 0.0 * ret, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ret_pbc_nn, 0.0 * ret, atol=1e-5, rtol=1e-5)



if __name__ == '__main__':
  absltest.main()
