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
        'testcase_name': f'_envelope={envelope}_geminal={geminal}',
        'envelope_label': envelope,
        'geminal' : geminal,
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
  geminal = True
  avg_env_mat = [0.5, 0., 0., 0.]
  std_env_mat = [4., 5., 5., 5.]
  onehot_scale = 0.01
  cat_sum_embd = True
  nele_sel = 0
  nion_sel = 0
  rc_mode = 'poly'
  rc_cntr = 8
  rc_sprd = 0.2
  only_ele = True
  use_ae_feat = False
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
      # 'envelope': list(envelopes.EnvelopeLabel),
      'envelope_label': list(envelopes.EnvelopeLabel)[-2:],
      'bias_orbitals': [True, False],
      'full_det': [True, False],
      # 'use_last_layer': [True, False],
      # 'hidden_dims': [((32, 8), (32, 8))],
      'use_last_layer': [False],
      'hidden_dims' : [DPOptions()],
  }
  # Create the product of all options.
  for options in itertools.product(*all_options.values()):
    # Yield dict of the current combination of options.
    yield dict(zip(all_options.keys(), options))


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
    
    # give the fixed params
    # Randomize parameters of envelope
    if isinstance(params['envelope'], list):
      for i in range(len(params['envelope'])):
        if params['envelope'][i]:
          params['envelope'][i]['sigma'] = jnp.ones(
              params['envelope'][i]['sigma'].shape)
          params['envelope'][i]['pi'] = jnp.ones(
              params['envelope'][i]['pi'].shape)
    else:
      params['envelope']['sigma'] = jnp.ones(
              params['envelope']['sigma'].shape)
      params['envelope']['pi'] = jnp.ones(
              params['envelope']['pi'].shape)
    return params


class NetworksTest(parameterized.TestCase):
  def test_value_det(self):
    """Check that the Fermi Net value is what we want when initialized with the same random key.""" 
    from jax.config import config
    config.update("jax_enable_x64", True) 

    envelope_label = envelopes.EnvelopeLabel.EXACT_CUSP

    atoms = np.reshape(np.array([i for i in range(12)]), [4,3])
    charges = np.array([1,2,3,4])
    nspins = (4, 3)

    data1 = np.array([i*0.1 for i in range(sum(nspins)*3)])

    kwargs = {}
    if envelope_label == envelopes.EnvelopeLabel.EXACT_CUSP:
      kwargs.update({'charges': charges, 'nspins': nspins})
    hidden_dims = DPOptions()
    hidden_dims.geminal = False
    hidden_dims.avg_env_mat = [0.] * 4
    hidden_dims.std_env_mat = [1.] * 4
    options = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs))

    
    key = random.PRNGKey(42)
    params = networks.init_fermi_net_params(
        key,
        atoms=atoms,
        charges=charges,
        nspins=nspins,
        options=options,
    )

    params = set_fixed_parameters(params, hidden_dims)

    hidden_dims.only_ele = True
    hidden_dims.rc_mode = 'none'
    options1 = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs))
    exp_val1 = -227.532447
    out1 = networks.fermi_net(params, data1, atoms, charges, nspins, options1)
    np.testing.assert_allclose(out1[1], np.array([exp_val1]), atol=1E-5, rtol=1E-5)
 

    hidden_dims.only_ele = False
    hidden_dims.rc_mode = 'none'
    options2 = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs))
    exp_val2 = -227.212452
    out2 = networks.fermi_net(params, data1, atoms, charges, nspins, options2)
    np.testing.assert_allclose(out2[1], np.array([exp_val2]), atol=1E-5, rtol=1E-5)

    
    hidden_dims.only_ele = True
    hidden_dims.rc_mode = 'poly'
    options3 = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs))
    exp_val3 = -228.578777
    out3 = networks.fermi_net(params, data1, atoms, charges, nspins, options3)
    np.testing.assert_allclose(out3[1], np.array([exp_val3]), atol=1E-5, rtol=1E-5)


    hidden_dims.only_ele = False
    hidden_dims.rc_mode = 'poly'
    options4 = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs))
    exp_val4 = -227.38814
    out4 = networks.fermi_net(params, data1, atoms, charges, nspins, options4)
    np.testing.assert_allclose(out4[1], np.array([exp_val4]), atol=1E-5, rtol=1E-5)




  def test_value_det_sel(self):
    """Check that the Fermi Net value is what we want when initialized with the same random key.""" 
    from jax.config import config
    config.update("jax_enable_x64", True) 

    envelope_label = envelopes.EnvelopeLabel.EXACT_CUSP

    atoms = np.reshape(np.array([i for i in range(12)]), [4,3])
    charges = np.array([1,2,3,4])
    nspins = (4, 3)

    data1 = np.array([i*0.1 for i in range(sum(nspins)*3)])

    kwargs = {}
    if envelope_label == envelopes.EnvelopeLabel.EXACT_CUSP:
      kwargs.update({'charges': charges, 'nspins': nspins})
    hidden_dims = DPOptions()
    hidden_dims.geminal = False
    hidden_dims.avg_env_mat = [0.] * 4
    hidden_dims.std_env_mat = [1.] * 4
    hidden_dims.nele_sel = 2
    hidden_dims.nion_sel = 3
    options = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs))

    
    key = random.PRNGKey(42)
    params = networks.init_fermi_net_params(
        key,
        atoms=atoms,
        charges=charges,
        nspins=nspins,
        options=options,
    )

    params = set_fixed_parameters(params, hidden_dims)

    hidden_dims.only_ele = True
    hidden_dims.rc_mode = 'none'
    options1 = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs))
    exp_val1 = -225.52228
    out1 = networks.fermi_net(params, data1, atoms, charges, nspins, options1)
    np.testing.assert_allclose(out1[1], np.array([exp_val1]), atol=1E-5, rtol=1E-5)
 

    hidden_dims.only_ele = False
    hidden_dims.rc_mode = 'none'
    options2 = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs))
    exp_val2 = -229.973202
    out2 = networks.fermi_net(params, data1, atoms, charges, nspins, options2)
    np.testing.assert_allclose(out2[1], np.array([exp_val2]), atol=1E-5, rtol=1E-5)

    
    hidden_dims.only_ele = True
    hidden_dims.rc_mode = 'poly'
    options3 = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs))
    exp_val3 = -227.71677
    out3 = networks.fermi_net(params, data1, atoms, charges, nspins, options3)
    np.testing.assert_allclose(out3[1], np.array([exp_val3]), atol=1E-5, rtol=1E-5)


    hidden_dims.only_ele = False
    hidden_dims.rc_mode = 'poly'
    options4 = networks.FermiNetOptions(
        hidden_dims=hidden_dims,
        envelope=envelopes.get_envelope(envelope_label, **kwargs))
    exp_val4 = -225.80039
    out4 = networks.fermi_net(params, data1, atoms, charges, nspins, options4)
    np.testing.assert_allclose(out4[1], np.array([exp_val4]), atol=1E-5, rtol=1E-5)

    

if __name__ == '__main__':
  absltest.main()
