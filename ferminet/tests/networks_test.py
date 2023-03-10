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
from ferminet.pbc import envelopes as pbc_env
from ferminet.pbc import feature_layer as pbc_feat
from ferminet import networks
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def rand_default():
  randn = np.random.RandomState(0).randn
  def generator(shape, dtype):
    return randn(*shape).astype(dtype)
  return generator


def _antisymmtry_options():
  for envelope,nn,mm,dc in \
      itertools.product(
        envelopes.EnvelopeLabel,
        [None, 9],
        ['orig', 'zinv'],
        [False, True],
      ):
    yield {
        'testcase_name': f'_envelope={envelope}_detnlayer={nn}_model={mm}_complex={dc}',
        'envelope_label': envelope,
        'dtype': np.float32,
      'det_nlayer' : nn,
      'model' : mm,
      'do_complex' : dc,
    }

def _pbc_antisymmtry_options():
  for has_cos,do_complex in \
      itertools.product(
        [True, False],
        [True, False],
      ):
    yield {
        'testcase_name': f'_has_cos={has_cos}_dc={do_complex}',
        'has_cos': has_cos,
        'do_complex': do_complex,
    }


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
      'envelope_label': list(envelopes.EnvelopeLabel),
      'bias_orbitals': [True, False],
      'full_det': [True, False],
      'use_last_layer': [True, False],
      'hidden_dims': [((32, 8), (32, 8))],
  }
  # Create the product of all options.
  for options in itertools.product(*all_options.values()):
    # Yield dict of the current combination of options.
    yield dict(zip(all_options.keys(), options))


class NetworksTest(parameterized.TestCase):

  @parameterized.named_parameters(_antisymmtry_options())
  def test_antisymmetry(self, envelope_label, dtype, det_nlayer, model, do_complex):
    """Check that the Fermi Net is symmetric."""
    del dtype  # unused

    key = random.PRNGKey(42)

    key, *subkeys = random.split(key, num=3)
    atoms = random.normal(subkeys[0], shape=(4, 3))
    charges = random.normal(subkeys[1], shape=(4,))
    nspins = (3, 4)

    key, subkey = random.split(key)
    data1 = random.normal(subkey, shape=(sum(nspins)*3,))
    data2 = jnp.concatenate((data1[3:6], data1[:3], data1[6:]))
    data3 = jnp.concatenate((data1[:9], data1[12:15], data1[9:12], data1[15:]))
    key, subkey = random.split(key)
    kwargs = {}
    if envelope_label == envelopes.EnvelopeLabel.EXACT_CUSP:
      kwargs.update({'charges': charges, 'nspins': nspins})
    feature_layer = networks.make_ferminet_features(
        charges,
        nspins,
        ndim=3,
    )
    hidden_dims = ((16, 16), (16, 16))
    if model == 'orig':
      make_model_fn = networks.make_fermi_net_model
    elif model == 'zinv':
      make_model_fn = networks.make_fermi_net_model_zinv
    else:
      raise RuntimeError(f'unknown model type {model}')
    options = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs),
        det_nlayer=det_nlayer,
        ferminet_model=make_model_fn(
          atoms, nspins, feature_layer, hidden_dims, False),
        do_complex=do_complex,
    )

    params = networks.init_fermi_net_params(
        subkey,
        atoms=atoms,
        nspins=nspins,
        options=options,
    )

    # Randomize parameters of envelope
    if isinstance(params['envelope'], list):
      for i in range(len(params['envelope'])):
        if params['envelope'][i]:
          key, *subkeys = random.split(key, num=3)
          params['envelope'][i]['sigma'] = random.normal(
              subkeys[0], params['envelope'][i]['sigma'].shape)
          params['envelope'][i]['pi'] = random.normal(
              subkeys[1], params['envelope'][i]['pi'].shape)
    else:
      key, *subkeys = random.split(key, num=3)
      params['envelope']['sigma'] = random.normal(
          subkeys[0], params['envelope']['sigma'].shape)
      params['envelope']['pi'] = random.normal(
          subkeys[1], params['envelope']['pi'].shape)

    out1 = networks.fermi_net(params, data1, atoms, nspins, options)

    out2 = networks.fermi_net(params, data2, atoms, nspins, options)
    np.testing.assert_allclose(out1[1], out2[1], atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(out1[0], -1*out2[0], atol=1E-5, rtol=1E-5)

    out3 = networks.fermi_net(params, data3, atoms, nspins, options)
    np.testing.assert_allclose(out1[1], out3[1], atol=3E-5, rtol=3E-5)
    np.testing.assert_allclose(out1[0], -1*out3[0], atol=3E-5, rtol=3E-5)

  @parameterized.named_parameters(_pbc_antisymmtry_options())
  def test_pbc_antisymmetry(self, has_cos, do_complex):
    """Check that the Fermi Net is symmetric."""

    # from jax.config import config
    # config.update("jax_enable_x64", True) 
    key = random.PRNGKey(42)

    key, *subkeys = random.split(key, num=3)
    atoms = random.normal(subkeys[0], shape=(4, 3))
    charges = random.normal(subkeys[1], shape=(4,))
    nspins = (3, 3)
    lattice = jnp.diag(jnp.array([3., 4., 5.]))
    rec_lattice = jnp.linalg.inv(lattice)

    key, subkey = random.split(key)
    data = random.normal(subkey, shape=(sum(nspins)*3,))
    data = jnp.asarray(data)
    data_p0 = jnp.concatenate((data[3:6], data[:3], data[6:]))
    data_p1 = jnp.concatenate((data[:9], data[12:15], data[9:12], data[15:]))
    x_trans = np.random.randint(-5, 6, size=(sum(nspins), 3))
    x_trans = jnp.matmul(x_trans, lattice)
    data_t0 = data + x_trans.reshape([-1])
    idx = np.random.permutation(4)
    atoms1 = atoms[idx,:]

    key, subkey = random.split(key)
    kwargs = {}
    envelope = pbc_env.make_ds_isotropic_envelope(lattice)
    feature_layer = pbc_feat.make_ds_features(
        charges,
        nspins,
        ndim=3,
        lattice=lattice,
        has_cos=has_cos,
    )
    hidden_dims = ((8, 3), (8, 3))
    make_model_fn = networks.make_fermi_net_model_zinv

    options = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelope,
        det_nlayer=0,
        feature_layer=feature_layer,
        ferminet_model=make_model_fn(
          atoms, nspins, feature_layer, hidden_dims, False),
        do_complex=do_complex,
    )

    params = networks.init_fermi_net_params(
        subkey,
        atoms=atoms,
        nspins=nspins,
        options=options,
    )

    # Randomize parameters of envelope
    if isinstance(params['envelope'], list):
      for i in range(len(params['envelope'])):
        if params['envelope'][i]:
          key, *subkeys = random.split(key, num=3)
          params['envelope'][i]['sigma'] = random.normal(
              subkeys[0], params['envelope'][i]['sigma'].shape)
          params['envelope'][i]['pi'] = random.normal(
              subkeys[1], params['envelope'][i]['pi'].shape)
    else:
      key, *subkeys = random.split(key, num=3)
      params['envelope']['sigma'] = random.normal(
          subkeys[0], params['envelope']['sigma'].shape)
      params['envelope']['pi'] = random.normal(
          subkeys[1], params['envelope']['pi'].shape)

    # check permutation
    out = networks.fermi_net(params, data, atoms, nspins, options)
    out_p0 = networks.fermi_net(params, data_p0, atoms, nspins, options)
    np.testing.assert_allclose(out[1], out_p0[1], atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(out[0], -1*out_p0[0], atol=1E-5, rtol=1E-5)
    out_p1 = networks.fermi_net(params, data_p1, atoms, nspins, options)
    np.testing.assert_allclose(out[1], out_p1[1], atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(out[0], -1*out_p1[0], atol=1E-5, rtol=1E-5)
    # check translation
    out_t0 = networks.fermi_net(params, data_t0, atoms, nspins, options)
    np.testing.assert_allclose(out[1], out_t0[1], atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(out[0], out_t0[0], atol=1E-5, rtol=1E-5)


  def test_create_input_features(self):
    dtype = np.float32
    ndim = 3
    nelec = 6
    xs = np.random.normal(scale=3, size=(nelec, ndim)).astype(dtype)
    atoms = np.array([[0.2, 0.5, 0.3], [1.2, 0.3, 0.7]])
    input_features = networks.construct_input_features(xs, atoms)
    d_input_features = jax.jacfwd(networks.construct_input_features)(
        xs, atoms, ndim=3)
    r_ee = input_features[-1][:, :, 0]
    d_r_ee = d_input_features[-1][:, :, 0]
    # The gradient of |r_i - r_j| wrt r_k should only be non-zero for k = i or
    # k = j and the i = j term should be explicitly masked out.
    mask = np.fromfunction(
        lambda i, j, k: np.logical_and(np.logical_or(i == k, j == k), i != j),
        d_r_ee.shape[:-1])
    d_r_ee_non_zeros = d_r_ee[mask]
    d_r_ee_zeros = d_r_ee[~mask]
    with self.subTest('check forward pass'):
      chex.assert_tree_all_finite(input_features)
      # |r_i - r_j| should be zero.
      np.testing.assert_allclose(np.diag(r_ee), np.zeros(6), atol=1E-5)
    with self.subTest('check backwards pass'):
      # Most importantly, check the gradient of the electron-electron distances,
      # |x_i - x_j|, is masked out for i==j.
      chex.assert_tree_all_finite(d_input_features)
      # We mask out the |r_i-r_j| terms for i == j. Check these are zero.
      np.testing.assert_allclose(
          d_r_ee_zeros, np.zeros_like(d_r_ee_zeros), atol=1E-5, rtol=1E-5)
      self.assertTrue(np.all(np.abs(d_r_ee_non_zeros) > 0.0))

  def test_construct_symmetric_features(self):
    dtype = np.float32
    hidden_units_one = 8  # 128
    hidden_units_two = 4  # 32
    nspins = (6, 5)
    h_one = np.random.uniform(
        low=-5, high=5, size=(sum(nspins), hidden_units_one)).astype(dtype)
    h_two = np.random.uniform(
        low=-5,
        high=5,
        size=(sum(nspins), sum(nspins), hidden_units_two)).astype(dtype)
    h_two = h_two + np.transpose(h_two, axes=(1, 0, 2))
    features = networks.construct_symmetric_features(h_one, h_two, nspins)
    # Swap electrons
    swaps = np.arange(sum(nspins))
    np.random.shuffle(swaps[:nspins[0]])
    np.random.shuffle(swaps[nspins[0]:])
    inverse_swaps = [0] * len(swaps)
    for i, j in enumerate(swaps):
      inverse_swaps[j] = i
    inverse_swaps = np.asarray(inverse_swaps)
    features_swap = networks.construct_symmetric_features(
        h_one[swaps], h_two[swaps][:, swaps], nspins)
    np.testing.assert_allclose(
        features, features_swap[inverse_swaps], atol=1E-5, rtol=1E-5)

  @parameterized.parameters(_network_options())
  def test_fermi_net(self, vmap, **network_options):
    # Warning: this only tests we can build and run the network. It does not
    # test correctness of output nor test changing network width or depth.
    nspins = (6, 5)
    atoms = jnp.asarray([[0., 0., 0.2], [1.2, 1., -0.2], [2.5, -0.8, 0.6]])
    charges = jnp.asarray([2, 5, 7])
    key = jax.random.PRNGKey(42)
    feature_layer = networks.make_ferminet_features(
        charges,
        nspins,
        ndim=3,
    )
    kwargs = {}
    if network_options['envelope_label'] == envelopes.EnvelopeLabel.EXACT_CUSP:
      kwargs.update({'charges': charges, 'nspins': nspins})
    network_options['envelope'] = envelopes.get_envelope(
        network_options['envelope_label'], **kwargs)
    del network_options['envelope_label']
    init, fermi_net, _ = networks.make_fermi_net(atoms, nspins, charges,
                                                 feature_layer=feature_layer,
                                                 **network_options)

    key, subkey = jax.random.split(key)
    if vmap:
      batch = 10
      xs = jax.random.uniform(subkey, shape=(batch, sum(nspins), 3))
      fermi_net = jax.vmap(fermi_net, in_axes=(None, 0))
      expected_shape = (batch,)
    else:
      xs = jax.random.uniform(subkey, shape=(sum(nspins), 3))
      expected_shape = ()

    key, subkey = jax.random.split(key)
    envelope = network_options['envelope']
    if (envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL and
        network_options['bias_orbitals']):
      with self.assertRaises(ValueError):
        init(subkey)
    else:
      params = init(subkey)
      sign_psi, log_psi = fermi_net(params, xs)
      self.assertSequenceEqual(sign_psi.shape, expected_shape)
      self.assertSequenceEqual(log_psi.shape, expected_shape)

  @parameterized.parameters(
      *(itertools.product([(1, 0), (2, 0), (0, 1)], [True, False])))
  def test_spin_polarised_fermi_net(self, nspins, full_det):
    atoms = jnp.zeros(shape=(1, 3))
    charges = jnp.ones(shape=1)
    key = jax.random.PRNGKey(42)
    feature_layer = networks.make_ferminet_features(
        charges,
        nspins,
        ndim=3,
    )
    init, fermi_net, _ = networks.make_fermi_net(
        atoms, nspins, charges, feature_layer=feature_layer, full_det=full_det)
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    params = init(subkey1)
    xs = jax.random.uniform(subkey2, shape=(sum(nspins) * 3,))
    # Test fermi_net runs without raising exceptions for spin-polarised systems.
    sign_out, log_out = fermi_net(params, xs)
    self.assertEqual(sign_out.size, 1)
    self.assertEqual(log_out.size, 1)


if __name__ == '__main__':
  absltest.main()
