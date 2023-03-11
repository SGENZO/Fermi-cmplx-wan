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

"""Helper functions to create the loss and custom gradient of the loss."""

from typing import Tuple

import chex
from ferminet import constants
from ferminet import hamiltonian
from ferminet import networks
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol


@chex.dataclass
class AuxiliaryLossData:
    """Auxiliary data returned by total_energy.

  Attributes:
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
  """
    variance: jnp.DeviceArray
    local_energy: jnp.DeviceArray


class LossFn(Protocol):

    def __call__(
            self,
            params_trial: networks.ParamTree,
            params_wave: networks.ParamTree,
            key_trial: chex.PRNGKey,
            key_wave: chex.PRNGKey,
            data_trial: jnp.ndarray,
            data_wave: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
        """描述弱形式下优化问题的目标，详见文件weak form。
    这行不用看了 Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params_trial: 传给试探函数网络的参数。
      params_wave: 传给目标解的网络的参数。
      key_trial: psi网络的PRNG state.
      key_wave: phi网络的PRNG state.
      data_trial: Batched electron positions to pass to the network.
      data_wave:

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
# 这里的return是不是需要对trial和wave分别给一个返回值？


def make_loss(network_trial: networks.LogFermiNetLike,
              network_wave: networks.LogFermiNetLike,
              local_energy_trial: hamiltonian.MomentLocalEnergy,
              local_energy_wave: hamiltonian.MomentLocalEnergy,
              clip_local_energy_trial: float = 0.0,
              clip_local_energy_wave: float = 0.0) -> LossFn:
    """Creates the loss function, including custom gradients.

  Args:
    network_trial: callable which evaluates the log of the magnitude of the
      trial function(square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
      注意这里传入的data是单mcmc configuration，也就是没有batch，需要vmap一下
    network_wave: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
    local_energy_trial: callable which evaluates the local energy of the
      trial function.
    local_energy_wave: callable which evaluates the local energy of the
      wavefunction.
    clip_local_energy_trial: clip local energy for trial function.
    clip_local_energy_wave: clip local energy for wavefunction.

    For clip local energy. If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.

  Returns:
    Callable with signature (params_t, param_w, data_t, data_w) and returns
    (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
    # local_energy的输入是networks.ParamTree, key: chex.PRNGKey, data: jnp.ndarray).
    # data经过vmap后的维度是(nelectrons, ndim, ntimestep)，这里的vmap用于处理mcmc采样后的一堆batch的数据
    # 新的输入维度为(batch, nelectrons, ndim, ntimestep)，输出为(batch, ntimestep)的local energy
    batch_local_energy_trial = jax.vmap(local_energy_trial, in_axes=(None, 0, 0), out_axes=0)
    batch_local_energy_wave = jax.vmap(local_energy_wave, in_axes=(None, 0, 0), out_axes=0)
    # LofFermiNetLike为(self, params: ParamTree,
    #                  electrons: jnp.ndarray,
    #                  time: jnp.float_) -> jnp.ndarray:
    batch_network_trial = jax.vmap(network_trial, in_axes=(None, 0), out_axes=0)
    batch_network_wave = jax.vmap(network_wave, in_axes=(None, 0), out_axes=0)

    @jax.custom_jvp
    def total_energy(
            params_trial: networks.ParamTree,
            params_wave: networks.ParamTree,
            key_trial: chex.PRNGKey,
            key_wave: chex.PRNGKey,
            data_trial: jnp.ndarray,
            data_wave: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
        """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params_trial: parameters to pass to the network of trial function.
      params_wave: parameters to pass to the network of wave function.
      key_trial: PRNG state.
      key_wave: PRNG state.
      data_trial: Batched MCMC configurations of the trial function to pass to
        the local energy function.
      data_wave: Batched MCMC configurations of the wavefunction to pass to
        the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
        keys_trial = jax.random.split(key_trial, num=data_trial.shape[0])
        keys_wave = jax.random.split(key_wave, num=data_wave.shape[0])





        e_l_trial = batch_local_energy_trial(params_trial, keys_trial, data_trial)
        e_l_wave = batch_local_energy_wave(params_wave, keys_wave, data_wave)

        E_pt_1 = constants.pmean(jnp.mean(e_l_tr))

        loss = constants.pmean(jnp.mean(e_l))
        variance = constants.pmean(jnp.mean((e_l - loss) ** 2))
        return loss, AuxiliaryLossData(variance=variance, local_energy=e_l)

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
        """Custom Jacobian-vector product for unbiased local energy gradients."""
        params, key, data = primals
        loss, aux_data = total_energy(params, key, data)

        if clip_local_energy > 0.0:
            # Try centering the window around the median instead of the mean?
            tv = jnp.mean(jnp.abs(aux_data.local_energy - loss))
            tv = constants.pmean(tv)
            diff = jnp.clip(aux_data.local_energy,
                            loss - clip_local_energy * tv,
                            loss + clip_local_energy * tv) - loss
        else:
            diff = aux_data.local_energy - loss

        # Due to the simultaneous requirements of KFAC (calling convention must be
        # (params, rng, data)) and Laplacian calculation (only want to take
        # Laplacian wrt electron positions) we need to change up the calling
        # convention between total_energy and batch_network
        primals = primals[0], primals[2]
        tangents = tangents[0], tangents[2]
        psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
        kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
        primals_out = loss, aux_data
        device_batch_size = jnp.shape(aux_data.local_energy)[0]
        tangents_out = (jnp.dot(psi_tangent, diff) / device_batch_size, aux_data)
        return primals_out, tangents_out

    return total_energy
