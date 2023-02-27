from typing import Sequence, Tuple, Optional

from absl import logging
from ferminet.utils import system
from ferminet.mcmc import apply_pbc
import numpy as np
import pyscf.pbc

class Scf:
    """Helper class for running Hartree-Fock (self-consistent field) with pyscf.
    
    Attributes:
    molecule: list of system.Atom objects giving the atoms in the
      molecule and their positions.
    nelectrons: Tuple with number of alpha electrons and beta
      electrons.
    basis: Basis set to use, best specified with the relevant string
      for a built-in basis set in pyscf. A user-defined basis set can be used
      (advanced). See https://sunqm.github.io/pyscf/gto.html#input-basis for
        more details.
    pyscf_mol: the PySCF 'Molecule'. If this is passed to the init,
      the molecule, nelectrons, and basis will not be used, and the
      calculations will be performed on the existing pyscf_mol
    restricted: If true, use the restricted Hartree-Fock method, otherwise use
      the unrestricted Hartree-Fock method.
    mean_field: the actual UHF object.
    """

    def __init__(self,
                 molecule: pyscf.pbc.gto.Cell,
                 twist : np.array = np.ones(3)*0.0,
                 restricted : bool = False,
                 ):
        # always assume un-restricted
        del restricted
        self._mol = molecule
        self.mean_field = None
        self.kpts = np.asarray([[0., 0., 0.]])
        self.lattice = self._mol.a
        
    def run(self, dm0: Optional[np.ndarray] = None):
        # self.mean_field = pyscf.pbc.scf.KUHF(self._mol, exxdiv='ewald', kpts=self.kpts).density_fit()
        self.mean_field = pyscf.pbc.scf.UHF(self._mol, exxdiv='ewald').density_fit()
        dm_up, dm_down = self.mean_field.get_init_guess()
        dm_down[:2, :2] = 0
        dm = (dm_up, dm_down)
        self.mean_field.kernel(dm)
        return self.mean_field
      
    def eval_mos(self, positions: np.ndarray,
                 deriv: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean_field is None:
            raise RuntimeError('Mean-field calculation has not been run.')
        coeffs = self.mean_field.mo_coeff
        if self._mol.cart:
            raise NotImplementedError(
                'Evaluation of molecular orbitals using cartesian GTOs.')
        positions = np.array(apply_pbc(positions, self.lattice), dtype=np.float64)
        gto_op = 'GTOval_sph_deriv1' if deriv else 'GTOval_sph'
        ao_values = self._mol.pbc_eval_gto(gto_op, positions)
        mo_values = tuple(np.matmul(ao_values, coeff) for coeff in coeffs)
        return mo_values
  
  
