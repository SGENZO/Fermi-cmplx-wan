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
  loss_imag: jnp.DeviceArray


class LossFn(Protocol):
  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched electron positions to pass to the network.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """

def make_loss(
        network: networks.LogFermiNetLike,
        local_energy: hamiltonian.LocalEnergy,
        clip_local_energy: float = 0.0,
        clip_mode: str = "radius",
) -> LossFn:
  """Creates the loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  batch_local_energy = jax.vmap(local_energy, in_axes=(None, 0, 0), out_axes=0)
  batch_network = jax.vmap(network, in_axes=(None, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    keys = jax.random.split(key, num=data.shape[0])
    e_l = batch_local_energy(params, keys, data)
    loss = constants.pmean(jnp.mean(e_l))
    variance = constants.pmean(jnp.mean((e_l - loss)**2))
    loss_real = loss.real
    loss_imag = loss.imag
    # return the real part of loss. 
    # The average of loc ener can be made real.
    loss_ret = loss_real
    return loss_ret, AuxiliaryLossData(
        variance=variance, 
        local_energy=e_l,
        loss_imag=loss_imag,
    )

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, key, data = primals
    loss, aux_data = total_energy(params, key, data)

    de = aux_data.local_energy - loss
    if clip_local_energy > 0.0:
      if clip_mode == "radius":
        # Try centering the window around the median instead of the mean?
        rr, aa = jnp.abs(de), jnp.angle(de)
        avg_rr = constants.pmean(jnp.mean(rr))
        tv = jnp.mean(jnp.abs(rr - avg_rr))
        tv = constants.pmean(tv)
        clip_rr = jnp.clip(rr,
                           avg_rr - clip_local_energy * tv,
                           avg_rr + clip_local_energy * tv)
        diff = clip_rr * jnp.exp(1j * aa)
      elif clip_mode == "radius_ds":
        # The "complex" implementation from deep solid
        radius, phase = jnp.abs(de), jnp.angle(de)
        radius_tv = constants.pmean(radius.std())
        radius_mean = jnp.median(radius)
        radius_mean = constants.pmean(radius_mean)
        clip_radius = jnp.clip(radius,
                               radius_mean - radius_tv * clip_local_energy,
                               radius_mean + radius_tv * clip_local_energy)
        diff = clip_radius * jnp.exp(1j * phase)
      elif clip_mode == "real":
        # The "real" implementation from deep solid
        tv_re = jnp.mean(jnp.abs(de.real))
        tv_re = constants.pmean(tv_re)
        tv_im = jnp.mean(jnp.abs(de.imag))
        tv_im = constants.pmean(tv_im)
        clip_diff_re = jnp.clip(de.real,
                                -clip_local_energy * tv_re,
                                clip_local_energy * tv_re)
        clip_diff_im = jnp.clip(de.imag,
                                -clip_local_energy * tv_im,
                                clip_local_energy * tv_im)
        diff = clip_diff_re + clip_diff_im * 1j
      else:
        raise ValueError('Unrecognized clip mode', clip_mode)
    else:
      diff = de
    
    # Due to the simultaneous requirements of KFAC (calling convention must be
    # (params, rng, data)) and Laplacian calculation (only want to take
    # Laplacian wrt electron positions) we need to change up the calling
    # convention between total_energy and batch_network
    primals = primals[0], primals[2]
    tangents = tangents[0], tangents[2]
    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    conj_psi_primal = jnp.conjugate(psi_primal)
    conj_psi_tangent = jnp.conjugate(psi_tangent)    
    kfac_jax.register_normal_predictive_distribution(conj_psi_primal[:, None])
    primals_out = loss, aux_data
    tangents_out = (jnp.mean((conj_psi_tangent * diff).real), aux_data)
    return primals_out, tangents_out

  return total_energy

    
