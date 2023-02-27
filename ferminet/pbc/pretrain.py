from typing import Callable, Optional, Sequence, Tuple, Union

from absl import logging
import chex
import ferminet
from ferminet import constants
from ferminet import envelopes
from ferminet import mcmc
from ferminet import networks

import jax
from jax import numpy as jnp
import kfac_jax
import numpy as np
import optax
import pyscf


def make_eval_orbitals(
        lattice : jnp.ndarray,
        nspins: Tuple[int, int],
        do_complex : Optional[bool] = False,
) -> jnp.ndarray:
    kpoints = ferminet.pbc.envelopes.make_kpoints(lattice, nspins)    # nk x 3

    def eval_orbitals(
            scf,
            pos: Union[np.ndarray, jnp.ndarray],
            nspins: Tuple[int, int],
            use_geminal : Optional[bool] = False,
    )->Tuple[np.ndarray, np.ndarray]:
        del scf
        if use_geminal:
            raise RuntimeError("geminal is not supported by heg pretraining")
        if not isinstance(pos, np.ndarray):  # works even with JAX array
            try:
                pos = pos.copy()
            except AttributeError as exc:
                raise ValueError('Input must be either NumPy or JAX array.') from exc
        leading_dims = pos.shape[:-1]
        pos = np.reshape(pos, [-1, 3])  # (batch*nelec, 3)
        phase_coords = pos @ kpoints.T  # (batch*nelec, nk)
        if do_complex:
            waves = np.exp(1j * phase_coords)   # (nb x nelec) x nk
        else:
            waves = np.cos(phase_coords)        # (nb x nelec) x nk
        # nb x nelec x nk
        waves = np.reshape(waves, leading_dims + (sum(nspins), -1))
        alpha_spin = waves[..., :nspins[0], :nspins[0]]
        beta_spin = waves[..., nspins[0]:, nspins[0]:nspins[0]+nspins[1]]
        return alpha_spin, beta_spin

    return eval_orbitals
