import jax, chex, itertools
import numpy as np
import jax.numpy as jnp
import jax.random as jran
import logging
import functools
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union
from ferminet import envelopes
from ferminet import network_blocks
from ferminet import sto
from ferminet.utils import scf
from typing_extensions import Protocol
from jax.experimental.host_callback import id_print, id_tap

FermiLayers = Tuple[Tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]
# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param = Mapping[str, jnp.ndarray]

use_ion = False

def sorted_mol_coords(rions, charges):
    """
    This function sort the coordinates of ions with respect to their Z number.

    Returns:
    sorted_coords:      jnp.array
                        sorted coordinate, has shape (nionx3)
    n_ion:              list
                        number of ions of each Z number
    zz_list:            list
                        list of Z numbers. increasing order.
    """
    coords = np.array(rions)
    charges = np.array(charges)
    z_list = np.array(sorted(list(set(list(charges)))))
    out_z_list = np.array([])
    n_ion = []
    sorted_coords = np.array([])
    for zz in z_list:
        # skip dummy ions
        if zz == 0: continue
        out_z_list = np.append(out_z_list, zz)
        select = (charges == zz)
        n_ion.append(np.sum(select))
        for ii in range(len(select)):
            if select[ii]:
                sorted_coords = np.append(sorted_coords, coords[ii*3:ii*3+3])
    sorted_coords = jnp.array(sorted_coords).reshape([-1])
    n_ion = list(n_ion)
    z_list = list(z_list)
    return sorted_coords, n_ion, out_z_list

# called by pretrain
def construct_input_features(
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Constructs inputs to Fermi Net from raw electron and atomic positions.

  Args:
    pos: electron positions. Shape (nelectrons*ndim,).
    atoms: atom positions. Shape (natoms, ndim).
    ndim: dimension of system. Change only with caution.

  Returns:
    ae, ee, r_ae, r_ee tuple, where:
      ae: atom-electron vector. Shape (nelectron, natom, ndim).
      ee: atom-electron vector. Shape (nelectron, nelectron, ndim).
      r_ae: atom-electron distance. Shape (nelectron, natom, 1).
      r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
    The diagonal terms in r_ee are masked out such that the gradients of these
    terms are also zero.
  """
  assert atoms.shape[1] == ndim
  ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
  ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])

  r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
  # Avoid computing the norm of zero, as is has undefined grad
  n = ee.shape[0]
  r_ee = (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))

  return ae, ee, r_ae, r_ee[..., None]

def init_orbital_shaping(
    key: chex.PRNGKey,
    input_dim: int,
    nspin_orbitals: Sequence[int],
    bias_orbitals: bool,
) -> Sequence[Param]:
  """Initialises orbital shaping layer.

  Args:
    key: JAX RNG state.
    input_dim: dimension of input activations to the orbital shaping layer.
    nspin_orbitals: total number of orbitals in each spin-channel.
    bias_orbitals: whether to include a bias in the layer.

  Returns:
    Parameters of length len(nspin_orbitals) for the orbital shaping for each
    spin channel.
  """
  orbitals = []
  for nspin_orbital in nspin_orbitals:
    key, subkey = jax.random.split(key)
    orbitals.append(
        network_blocks.init_linear_layer(
            subkey,
            in_dim=input_dim,
            out_dim=nspin_orbital,
            include_bias=bias_orbitals))
  return orbitals


def init_fermi_net_params(
    key: chex.PRNGKey,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: Any,
    hf_solution: Optional[scf.Scf] = None,
    eps: float = 0.01,
) -> ParamTree:
  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    if options.bias_orbitals:
      raise ValueError('Cannot bias orbitals w/STO envelope.')
  if hf_solution is not None:
    if options.use_last_layer:
      raise ValueError('Cannot use last layer w/HF init')
    if options.envelope.apply_type not in ('sto', 'sto-poly'):
      raise ValueError('When using HF init, '
                       'envelope_type must be `sto` or `sto-poly`.')

  active_spin_channels = [spin for spin in nspins if spin > 0]
  nchannels = len(active_spin_channels)
  if nchannels == 0:
    raise ValueError('No electrons present!')

  z_xx, nions, zzs = sorted_mol_coords(atoms, charges)
  nae = sum(nions)
  spins = list(nspins)
  part_list = spins + nions    
  npart = sum(part_list)
  ntypes = len(part_list)
  nz = npart - sum(spins)
  
  env_mat_dim = 4
  type_embd = options.hidden_dims.type_embd
  cntr_type = options.hidden_dims.cntr_type
  n_conv_layers = options.hidden_dims.n_conv_layers
  nfeat = options.hidden_dims.nfeat
  nodim = options.hidden_dims.nodim
  embd_dims = options.hidden_dims.embd
  fitt_dims = options.hidden_dims.fitt
  feat_dims = options.hidden_dims.feat
  cat_sum_embd = options.hidden_dims.cat_sum_embd
  only_ele = options.hidden_dims.only_ele
  use_ae_feat = options.hidden_dims.use_ae_feat
  embd = []
  fitt = []
  feat = []
  rij = []

  params = {}
  if options.hidden_dims.use_ae_feat:
      (num_one_features, num_two_features), params['input'] = (
          options.feature_layer.init())
  natom, ndim = atoms.shape
  
  if not type_embd:
      embd_dims_in = [env_mat_dim] + embd_dims[:-1]
  else:
      if cntr_type:
          embd_dims_in = [env_mat_dim + nfeat * 2] + embd_dims[:-1]
      else:
          embd_dims_in = [env_mat_dim + nfeat] + embd_dims[:-1]
  embd_dims_out = embd_dims
  fitt_dims_in = [nfeat] + fitt_dims
  fitt_dims_out = fitt_dims + [nodim]
  if cat_sum_embd:
    if nz > 0:
      feat_dims_in = [3*embd_dims[-1] + nfeat] + feat_dims
    else:
      feat_dims_in = [2*embd_dims[-1] + nfeat] + feat_dims
    if only_ele and use_ae_feat:
      feat_dims_in[0] += nfeat*2
  else:
    feat_dims_in = [1*embd_dims[-1] + nfeat] + feat_dims
  feat_dims_out = feat_dims + [nfeat]

  def make_network_params(
          mykey,
          embd_dims_in,
          embd_dims_out,
  ):
    embd = []
    for ii in range(len(embd_dims_in)):
      mykey, subkey = jax.random.split(mykey)
      embd.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim = embd_dims_in[ii],
              out_dim = embd_dims_out[ii],
              include_bias=True
          ))
    return embd, mykey

  params['embd'] = []
  params['feat'] = []
  for ii in range(n_conv_layers):
    tmpe, key = make_network_params(key, feat_dims_in, feat_dims_out)
    tmpr, key = make_network_params(key, embd_dims_in, embd_dims_out )  
    params['feat'].append(tmpe)
    params['embd'].append(tmpr)
  params['fitt'], key = make_network_params(key, fitt_dims_in, fitt_dims_out)
  
  if not options.use_last_layer:
    # Just pass the activations from the final layer of the one-electron stream
    # directly to orbital shaping.
    # dims_orbital_in = nodim * 2
    dims_orbital_in = nodim
  else:
    raise RuntimeError('not suppported!')

  # How many spin-orbitals do we need to create per spin channel?
  nspin_orbitals = []
  for nspin in active_spin_channels:
    if options.full_det:
      # Dense determinant. Need N orbitals per electron per determinant.
      norbitals = sum(nspins) * options.determinants
    else:
      # Spin-factored block-diagonal determinant. Need nspin orbitals per
      # electron per determinant.
      norbitals = nspin * options.determinants
    nspin_orbitals.append(norbitals)

  # # Layer initialisation
  # key, subkey = jax.random.split(key, num=2)
  # params['single'], params['double'] = init_layers(
  #     key=subkey,
  #     dims_one_in=dims_one_in,
  #     dims_one_out=dims_one_out,
  #     dims_two_in=dims_two_in,
  #     dims_two_out=dims_two_out)    

  # create envelope params
  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    # Applied to output from final layer of 1e stream.
    output_dims = dims_orbital_in
  elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    # Applied to orbitals.
    output_dims = nspin_orbitals
  elif options.envelope.apply_type == envelopes.EnvelopeType.POST_DETERMINANT:
    # Applied to all determinants.
    output_dims = 1
  else:
    raise ValueError('Unknown envelope type')
  params['envelope'] = options.envelope.init(
      natom=natom, output_dims=output_dims, hf=hf_solution, ndim=ndim)

  # orbital shaping
  key, subkey = jax.random.split(key, num=2)
  params['orbital'] = init_orbital_shaping(
      key=subkey,
      input_dim=dims_orbital_in,
      nspin_orbitals=nspin_orbitals,
      bias_orbitals=options.bias_orbitals)

  if options.hidden_dims.geminal:
      key, subkey = jax.random.split(key)
      params['gemi'] = []
      numb_pad = nspins[0] - nspins[1]
      my_in_dim = nodim - numb_pad
      my_out_dim = (nodim - numb_pad) * options.determinants      
      params['gemi'].append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim = my_in_dim,
              out_dim = my_out_dim,
              include_bias=False
          ))
      # params['gemi'][0]['w'] = jnp.einsum('ij,kj->ik', params['gemi'][0]['w'], params['gemi'][0]['w'])
      # params['gemi'][0]['w'] += jnp.eye(my_in_dim)
  # if hf_solution is not None:
  #   params['single'], params['orbital'] = init_to_hf_solution(
  #       hf_solution=hf_solution,
  #       single_layers=params['single'],
  #       orbital_layer=params['orbital'],
  #       determinants=options.determinants,
  #       active_spin_channels=active_spin_channels,
  #       eps=eps)

  return params

residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
diag_eps = 1e-10

def _env_mat_rinv(
        rij,
        power : float = 1.0,
        rinv_shift : float = 1.0,
        switch_func = None,
):
    """
    env mat constructed as 

        1            xij            yij            zij
    ---------,  -------------, -------------, -------------, 
    rij^p + s   rij^(p+1) + s  rij^(p+1) + s  rij^(p+1) + s

    p is given by power
    s is given by rinv_shift
    """
    if power != 1.0:
        raise RuntimeError(f'the power {power} is not supported. only allows power==1.')
    # np0 x np1 x 3
    np0 = rij.shape[0]
    np1 = rij.shape[1]
    # np0 x np1
    nrij = jnp.linalg.norm(rij, axis=2)
    inv_nrij = switch_func(nrij)/(nrij + rinv_shift)
    # flaten 
    trij = jnp.reshape(rij, [-1,3])
    tnrij = jnp.tile(
        jnp.reshape(
            switch_func(nrij)/(nrij*nrij + rinv_shift),
            [-1, 1],
        ),
        [1,3],
    )
    # np0 x np1 x 3
    env_mat = jnp.reshape(jnp.multiply(trij, tnrij), [np0, np1, 3])
    # np0 x np1 x 4
    env_mat = jnp.concatenate(
        (jnp.reshape(inv_nrij, [np0, np1, 1]), env_mat),
        axis = 2
    )
    # np0 x np1 x 4, np0 x np1, np0 x np1
    return env_mat, inv_nrij, nrij


def switch_func_tanh(
        xx,
        rc_cntr = 3.0,
        rc_sprd = 0.2,
):
    uu = (xx - rc_cntr) / rc_sprd
    return 0.5 * (1. - jnp.tanh(uu))


def spline_func(xx, rc, rc_smth):
    uu = (xx - rc_smth) / (rc - rc_smth)
    return uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1


def switch_func_poly(
        xx,
        rc = 3.0,
        rc_smth = 0.2,
):
    ret = \
        1.0 * (xx < rc_smth) + \
        spline_func(xx, rc, rc_smth) * jnp.logical_and(xx >= rc_smth, xx < rc) + \
        0.0 * (xx >= rc)
    return ret

def choose_swf(rc_mode, rc_cntr, rc_sprd):
    do_smth = \
        (rc_mode is not None) and (rc_mode != 'none') and \
        (rc_cntr is not None) and (rc_sprd is not None)
    if do_smth:
        if rc_mode == 'tanh':
            swf = lambda xx: switch_func_tanh(xx, rc_cntr, rc_sprd)
        elif rc_mode == 'poly':
            swf = lambda xx: switch_func_poly(xx, rc_cntr, rc_sprd)
        else:
            raise RuntimeError('unknown rc mode', rc_mode)
    else :
        swf = jnp.ones_like
    return swf


def compute_ncopy(
        rc : float,
        lattice : np.array,
        rc_shell : float = 1e-5,
) -> np.array:
    vol = np.linalg.det(lattice)
    # tofacedist = np.cross([lattice[1], lattice[2], lattice[0]], 
    #                        [lattice[2], lattice[0], lattice[1]],)
    tofacedist = np.cross(lattice[[1,2,0],:], lattice[[2,0,1],:])
    tofacedist = vol * np.reciprocal(np.linalg.norm(tofacedist, axis=1))
    ncopy = (rc+rc_shell) * np.reciprocal(tofacedist)
    ncopy = np.array(np.ceil(ncopy), dtype=int)    
    return ncopy


def compute_copy_idx(
        rc : float,
        lattice : np.array,
):
    ncopy = compute_ncopy(rc, lattice)
    ordinals = np.asarray(list(
        itertools.product(
            range(-ncopy[0], ncopy[0]+1),
            range(-ncopy[1], ncopy[1]+1),
            range(-ncopy[2], ncopy[2]+1),
        )))
    ordinals = np.asarray(sorted(ordinals, key=np.linalg.norm))
    return ordinals


def compute_shift_vec(
        rc : float,
        lattice : np.array,
) -> np.array:
    ordinals = compute_copy_idx(rc, lattice)
    return np.matmul(ordinals, lattice)


def compute_background_coords(
        xx : jnp.array,
        lattice : jnp.array,
        shift_vec : jnp.array,
):
    ss = shift_vec
    xx = jnp.reshape(xx, [1,-1,3])
    ss = jnp.reshape(ss, [-1,1,3])
    coord = xx[None,:,:] + shift_vec[:,None,:]
    return coord


def compute_rij(
        xx : jnp.array,
        lattice : jnp.array = None,
        shift_vec : jnp.array = None,
):
    return compute_rij_2(xx, xx, lattice, shift_vec)


def compute_rij_2(
        cc : jnp.array,
        yy : jnp.array,
        lattice : jnp.array = None,
        shift_vec : jnp.array = None,
):
    if (lattice is not None) and (shift_vec is not None):
        bk_yy = compute_background_coords(yy, lattice, shift_vec).reshape([-1,3])
    else:
        bk_yy = yy
    # np0 x np1 x 3
    rij = bk_yy[None,:,:] - cc[:,None,:]    
    return rij


def compute_rij_nnei(
        xx : jnp.array,
        lattice : jnp.array = None,
        rec_lattice : jnp.array = None,
):
    """
    compute rij. PBC only with nearest neighbor
    """
    return compute_rij_2_nnei(xx, xx, lattice, rec_lattice)


def compute_rij_2_nnei(
        cc : jnp.array,
        yy : jnp.array,
        lattice : jnp.array = None,
        rec_lattice : jnp.array = None,
):
    """
    compute rij. PBC only with nearest neighbor
    """
    # np0 x np1 x 3
    rij = yy[None,:,:] - cc[:,None,:]    
    if lattice is not None:
        rij = apply_nearest_neighbor(rij, lattice, rec_lattice)
    return rij

def apply_nearest_neighbor(
        rij,
        lattice,
        rec_lattice,
):
    srij = jnp.matmul(rij, rec_lattice)
    srij = jnp.mod(srij+0.5, 1.0) - 0.5
    rij = jnp.matmul(srij, lattice)
    return rij


def auto_nearest_neighbor(
        lattice,
        rc,
) -> bool:
    vol = np.linalg.det(lattice)
    tofacedist = np.cross(lattice[[1,2,0],:], lattice[[2,0,1],:])
    tofacedist = vol * np.reciprocal(np.linalg.norm(tofacedist, axis=1))
    return (rc <= 0.5 * tofacedist[0]) and (rc <= 0.5 * tofacedist[1]) and (rc <= 0.5 * tofacedist[2])


def make_max_ele_nnei(
        lattice,
        shift_vec,
        rc,
        nspins,
):
    np1 = sum(nspins)
    ns = shift_vec.shape[0]
    slice_idx = [nspins[0]]
    def max_nnei(
            xx,
    ):
        np0 = xx.size // 3
        nrij = jnp.linalg.norm(compute_rij(xx.reshape([-1,3]), lattice, shift_vec), axis=2)
        tnrij = jnp.reshape(nrij, [np0, ns, np1])
        snrij = jnp.split(tnrij, slice_idx, axis=2)  
        # neighbors in different spin channels
        snrij = [ ii.reshape([np0, -1]) for ii in snrij ]
        # number of neighbors in different spin channels
        # 2 x np0
        nnei = jnp.asarray([ jnp.sum( (ii < rc), axis=1 ) for ii in snrij ])
        # max nnei
        max_nnei = jnp.max(nnei)
        return max_nnei
    def batch_max_nnei(xx):
        return jnp.max(jax.vmap(max_nnei, in_axes=(0))(xx))
    return batch_max_nnei


def make_max_ion_nnei(
        atoms,
        lattice,
        shift_vec,
        rc,
):
    np1 = atoms.size // 3
    ns = shift_vec.shape[0]
    def max_nnei(
            xx,
    ):
        np0 = xx.size // 3
        # np0 x np1
        nrij = jnp.linalg.norm(compute_rij_2(xx.reshape([-1,3]), atoms.reshape([-1,3]), lattice, shift_vec), axis=2).reshape([np0, ns*np1])
        # np0
        nnei = jnp.sum( (nrij < rc), axis=1 )
        # max nnei
        max_nnei = jnp.max(nnei)
        return max_nnei
    def batch_max_nnei(xx):
        return jnp.max(jax.vmap(max_nnei, in_axes=(0))(xx))
    return batch_max_nnei


def compute_env_mat(
        xx,
        power : float = 1.0,
        rinv_shift : float = 1.0,
        rc_mode : str = 'none', # 'tanh' or 'poly' or 'none'
        rc_cntr : float = None,
        rc_sprd : float = None,
        cut_dim : int = None,
        lattice : jnp.array = None,
        shift_vec : jnp.array = None,
        nearest_neighbor : bool = False,
):
    # np0 x 3
    xx = jnp.reshape(xx, [-1, 3])
    if nearest_neighbor:
        rij = compute_rij_nnei(xx, lattice, np.linalg.inv(lattice))
    else:
        rij = compute_rij(xx, lattice, shift_vec)
    # np0 x np1 x 3
    diag_shift = diag_eps * jnp.tile(jnp.expand_dims(jnp.eye(rij.shape[0], rij.shape[1]), axis=2), [1,1,3])
    rij += diag_shift    
    swf = choose_swf(rc_mode, rc_cntr, rc_sprd)
    # np0 x np1 x 4, np0 x np1, np0 x np1
    env_mat, inv_nrij, nrij = _env_mat_rinv(
        rij, 
        power=power, 
        rinv_shift=rinv_shift,
        switch_func=swf,
    )
    if cut_dim is not None:
        # cut_dim(npart/nele) x npart x 4
        env_mat = jax.lax.dynamic_slice(env_mat, (0,0,0), (cut_dim, env_mat.shape[1], env_mat.shape[2]))
        nrij = jax.lax.dynamic_slice(nrij, (0,0), (cut_dim, nrij.shape[1]))
    # np0 x np1 x 4, np0 x np1, np0 x np1
    return env_mat, inv_nrij, nrij


def _build_neigh_list(rr, nnei):
    # ascending sort
    idx = jnp.argsort(rr)
    nlist = jnp.split(idx, [nnei])[0]
    return nlist

build_neigh_list = jax.vmap(_build_neigh_list, in_axes=[0,None])

def _apply_neigh_list(mat, nlist):
    return mat[nlist, ...]

apply_neigh_list = jax.vmap(_apply_neigh_list, in_axes=[0,0])


def compute_layout(nspins, nz, nshift, only_ele):
    slc0 = [nspins[0]]
    typ0 = list(nspins)
    slc1 = [nspins[0]]
    typ1 = list(nspins)
    if nz > 0:
        slc1 += [nspins[0]+nspins[1]]
        typ1 += [nz]
    if not only_ele:
        slc0 += [nspins[0]+nspins[1]]
        typ0 += [nz]
    return {
        'slc0' : slc0,
        'slc1' : slc1,
        'typ0' : typ0,
        'typ1' : typ1,
        'np0': sum(nspins) if only_ele else sum(nspins) + nz,
        'np1': sum(nspins) + nz,
        'dim0' : len(typ0),
        'dim1' : len(typ1),
        'nshift' : nshift,
    }

def slice_data(
        data,
        nlout,
        with_feat_dim = True,
):    
    np0 = nlout['np0']
    np1 = nlout['np1']
    nshift = nlout['nshift']
    if with_feat_dim:
        dd = data.reshape([np0, nshift, np1, -1])
    else:
        dd = data.reshape([np0, nshift, np1])
    split_dd = [jnp.split(ii, nlout['slc1'], axis=2)
                for ii in jnp.split(dd, nlout['slc0'], axis=0)]
    for ii in range(nlout['dim0']):
        for jj in range(nlout['dim1']):
            if with_feat_dim:
                shape = [nlout['typ0'][ii], nshift*nlout['typ1'][jj], -1]
            else:
                shape = [nlout['typ0'][ii], nshift*nlout['typ1'][jj]]
            split_dd[ii][jj] = jnp.reshape(split_dd[ii][jj], shape)
    return split_dd


def data_apply_nlist(
        nlist,
        feat,
):
    """
    rr          [npart_, n_shift x npart]
    feat:       [npart_, n_shift x npart, nfeat]
    """    
    if nlist is None:
        return feat
    nspins = nlist['nlout']['typ0']
    if len(nspins) == 2 or len(nspins) == 3:
        ndata = nlist['ndata']
        nlout = nlist['nlout']
        ff_split = slice_data(feat, nlout)
        for ii in range(nlout['dim0']):
            for jj in range(nlout['dim1']):                
                ff_split[ii][jj] = apply_neigh_list(
                    ff_split[ii][jj], ndata[ii][jj],)
    else:
        raise RuntimeError('not supported: axis 0 n particles:', nspins)
    res = [jnp.concatenate(ii, axis=1) for ii in ff_split]
    # npart_ x (2 * nnei_ele + nnei_ion) x nfeat
    return jnp.concatenate(res, axis=0)


def feat_apply_nlist(
        nlist,
        feat,
        npart_,
        nshift,
):
    [npart, nfeat] = feat.shape
    # npart_ x npart x nf    
    tt_in = jnp.tile(feat.reshape([1, npart, nfeat]), [npart_, nshift, 1])
    if nlist is not None:
        tt_in = data_apply_nlist(nlist, tt_in)
    return tt_in


def cutting_func(rr, embd_in, rc_mode='none', rc_cntr=None, rc_sprd=None):
    '''
    rr:      |r_ij|              npart/npart_ x npart 
    embd_in: cat([rr,Feature])   npart/npart_ x npart x (nf+4)
    '''
    npart_ = embd_in.shape[0]
    npart  = embd_in.shape[1]
    in_dim = embd_in.shape[2]
    rr = jax.lax.dynamic_slice(rr, (0, 0), (npart_, npart))
    swf = choose_swf(rc_mode, rc_cntr, rc_sprd)

    cut_factor = jnp.tile(
        jnp.reshape(swf(rr), [npart_, npart, 1],
        ),
        [1, 1, in_dim],
    )
    embd_in = jnp.multiply(cut_factor, embd_in)
    return embd_in, rr


def dp_orbitals(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: Any,
    ae_features : Optional[jnp.ndarray] = None,
):
    z_xx, nions, zzs = sorted_mol_coords(atoms, charges)
    spins = list(nspins)
    nele = sum(spins)    
    part_list = spins + nions
    npart = sum(part_list)
    nz = npart - sum(spins)
    ntypes = len(part_list)
    if use_ion:
        nele = npart
    else:
        nele = sum(spins)
    part_list = tuple(part_list)
    # one hot representation
    part_type = []
    for idx,ii in enumerate(part_list):
        part_type += [idx] * ii

    atomic = True
    only_ele = options.hidden_dims.only_ele
    power = options.hidden_dims.power
    rinv_shift = options.hidden_dims.rinv_shift
    nfeat = options.hidden_dims.nfeat
    type_embd = options.hidden_dims.type_embd
    cntr_type = options.hidden_dims.cntr_type
    onehot_scale = options.hidden_dims.onehot_scale
    cat_sum_embd = options.hidden_dims.cat_sum_embd
    nele_sel = options.hidden_dims.nele_sel
    nion_sel = options.hidden_dims.nion_sel
    rc_mode = options.hidden_dims.rc_mode
    rc_cntr = options.hidden_dims.rc_cntr
    rc_sprd = options.hidden_dims.rc_sprd
    do_ele_sel = (nele_sel > 0) and (nion_sel > 0)
    lattice = options.lattice
    use_ae_feat = options.hidden_dims.use_ae_feat
    nearest_neighbor = options.hidden_dims.nearest_neighbor
    
    if nearest_neighbor == 'enable' and lattice is not None:
        nearest_neighbor = True
        if not auto_nearest_neighbor(lattice, rc_cntr):
            raise RuntimeError('rc should be larger than half-cell size')
    elif nearest_neighbor == 'auto' and lattice is not None:
        nearest_neighbor = auto_nearest_neighbor(lattice, rc_cntr)
    else :
        nearest_neighbor = False

    if lattice is not None and not nearest_neighbor:
        # search all possible pbc images
        shift_vec = jnp.asarray(compute_shift_vec(rc_cntr, lattice))
        n_shift = shift_vec.shape[0]
    else:
        # search only nearest_neighbor or open boundary condition
        shift_vec = None
        n_shift = 1
    if only_ele: 
        npart_ = nele
    else:
        npart_ = npart
    if not do_ele_sel:
        simple_part_list = spins + [sum(nions)]
    else:
        simple_part_list = \
            [ min(ii*n_shift, nele_sel) for ii in spins ] + \
            [ min(ii*n_shift, nion_sel) for ii in [sum(nions)] ]

    # do not include ions when nz == 0
    if nz == 0:
        assert 0 == simple_part_list.pop(-1)
    oh_all = jax.nn.one_hot(part_type, nfeat) * onehot_scale
    # (npart-npart_)nion x nf
    oh_atom = jax.lax.dynamic_slice(oh_all, (nele,0), (npart-npart_, nfeat))
    # average and inverse std of the env mat
    avg_env_mat = jnp.array(options.hidden_dims.avg_env_mat)
    std_env_mat = jnp.array(options.hidden_dims.std_env_mat)
    avg_env_mat = jnp.tile(jnp.reshape(avg_env_mat, [1, 1, 4]), [npart_,npart*n_shift,1])
    std_env_mat = jnp.tile(jnp.reshape(std_env_mat, [1, 1, 4]), [npart_,npart*n_shift,1])
        
    def env_cat_feat(env_mat, feat, nlist, one_side=True):
        """
        env_mat         np0 x np1 x nenv
        feat            np0 x nf
        """
        np0 = env_mat.shape[0]
        assert(np0 == npart_)
        np1 = env_mat.shape[1]
        nf = feat.shape[-1]
        if not one_side:
            feat0 = jnp.split(feat, [np0], axis = 0)[0]
            # npart_ x npart x nf
            tt_0 = jnp.tile(feat0.reshape([np0, 1, nf]), [1, np1, 1])
        # npart_ x npart x nf
        tt_1 = feat_apply_nlist(nlist, feat, np0, n_shift)
        # npart_ x npart x (nenv+nf+nf)
        if not one_side:
            ret = jnp.concatenate([env_mat, tt_0, tt_1], axis=2)
        else:
            ret = jnp.concatenate([env_mat, tt_1], axis=2)
        return ret


    def feat_layer(ll_idx, rr, nlist, feat, env_mat):
        # npart x nf
        if feat.shape[0]==npart:
            feat = jnp.reshape(feat, [npart, nfeat])
        else:
            feat = jnp.reshape(feat, [npart_, nfeat])
            # npart x nf
            feat = jnp.concatenate([feat, oh_atom], axis=0)
        # npart_ x npart x (4+nf)
        embd_in = env_cat_feat(env_mat, feat, nlist, one_side=(not cntr_type))
        # rr: npart_ x npart
        embd_in, rr = cutting_func(rr, embd_in, rc_mode=rc_mode, rc_cntr=rc_cntr, rc_sprd=rc_sprd)
        for ii in range(len(params['embd'][ll_idx])):
            # id_print(embd_in[:,:,:])
            # id_print(params['embd'][ll_idx][ii]['w'])
            # print('w b embd shapes', '==============',
            #       embd_in.shape, 
            #       params['embd'][ll_idx][ii]['w'].shape, 
            #       params['embd'][ll_idx][ii]['b'].shape)
            embd_next = jnp.tanh(
                network_blocks.vmap_linear_layer(
                    embd_in,
                    params['embd'][ll_idx][ii]['w'], 
                    params['embd'][ll_idx][ii]['b'],
                ))
            embd_in = residual(embd_in, embd_next)
        # id_print(embd_in)
        # npart_ x npart x nembd
        embd = embd_in
        nembd = embd_in.shape[-1]
        part_partitions = network_blocks.array_partitions(simple_part_list)
        # npart_ x nfeat
        cat_feat = [jax.lax.dynamic_slice(feat, (0,0), (npart_, nfeat))]
        if only_ele and use_ae_feat:
            e1_feat = jnp.split(feat, [nspins[0]], axis=0)
            e1_feat = [jnp.tile(jnp.mean(ii, axis=0, keepdims=True), 
                                [npart_, 1]) for ii in e1_feat]
            cat_feat += e1_feat
        if cat_sum_embd:
            if lattice is None or do_ele_sel:
                split_embd = jnp.split(embd, part_partitions, axis=1)
                for ii in split_embd:
                    # npart_ x nembd
                    cat_feat.append(jnp.mean(ii, axis=1))
            else:
                split_embd = jnp.split(
                    embd.reshape([npart_, n_shift, npart, nembd]),
                    part_partitions, axis = 2)
                for ii in split_embd:
                    # npart_ x nembd
                    cat_feat.append(jnp.mean(ii.reshape([npart_, -1, nembd]), axis=1))
        else:
            cat_feat.append(jnp.mean(embd, axis=1))
        # npart_ x (3 x nembd + nf)
        feat_in = jnp.concatenate(cat_feat, axis = 1)
        # id_print(feat_in)
        for ii in range(len(params['feat'][ll_idx])):
            feat_next = jnp.tanh(
                network_blocks.linear_layer(
                    feat_in, **params['feat'][ll_idx][ii]))
            # feat_in = residual(feat_in, feat_next)
            feat_in = feat_next
        feat_out = residual(feat, feat_in)
        # id_print(feat)
        # id_print(feat_in)
        # id_print(feat_out)
        # npart x nf
        return feat_out


    def conv_layers(feat, env_mat, rr, nlist):
        """
        feat : npart x nfeat
        env_mat : npart_ x npart x 4

        return
        feat : npart x nfeat        
        """
        n_conv_layers = len(params['feat'])
        for ii in range(n_conv_layers):
            feat = feat_layer(ii, rr, nlist, feat, env_mat)
        return feat

    xx = jnp.append(pos, z_xx)

    # vectorized
    # npart x npart x 4, npart x npart
    env_mat, _, rr = compute_env_mat(
        xx, 
        power=power, rinv_shift=rinv_shift,
        rc_mode='none', rc_cntr=None, rc_sprd=None,
        cut_dim=npart_,
        lattice=lattice,
        shift_vec=shift_vec,
        nearest_neighbor=nearest_neighbor,
    )
    # npart_ x npart x 4
    env_mat = (env_mat - avg_env_mat) * std_env_mat

    # npart x nfeat
    feat = jax.nn.one_hot(part_type, nfeat) * onehot_scale
    if use_ae_feat:
        if feat.shape[1] < ae_features.shape[1] + ntypes:
            raise RuntimeError(f'the size of the ae_features + ntypes is larger than feature. increase the feat size')
        np0 = feat.shape[0]
        pad = feat.shape[1] - ae_features.shape[1]
        feat = feat + jnp.concatenate([jnp.zeros([np0, pad]), ae_features], axis=1)

    # build neighbor list
    if do_ele_sel:
        nlout = compute_layout(nspins, nz, n_shift, only_ele)
        split_rr = slice_data(rr, nlout, with_feat_dim=False)
        nnei = [min(ii*n_shift, nele_sel) for ii in nspins]
        if nz > 0:
            nnei += [min(nz*n_shift, nion_sel)]
        nlist = {
            'nlout' : nlout,
            'ndata' : [[build_neigh_list(jj,nn) for jj,nn in zip(ii,nnei)]
                       for ii in split_rr],
        }
    else :
        nlist = None

    env_mat = data_apply_nlist(nlist, env_mat)
    rr = data_apply_nlist(nlist, rr)
    rr = jnp.squeeze(rr)

    # npart x nfeat
    feat = conv_layers(feat, env_mat, rr, nlist)
    # nele x nfeat
    feat = jax.lax.dynamic_slice(feat, (0, 0), (nele, nfeat))
    # nele x odim
    ff_in = feat
    for ii in range(len(params['fitt'])):
        ff_next = jnp.tanh(
            network_blocks.linear_layer(ff_in, **params['fitt'][ii])
        )
        ff_in = residual(ff_in, ff_next)
    # id_print(ff_in)
    # nele x odim
    ff = ff_in.reshape([nele, -1])
    odim = ff.shape[-1]
    if not atomic:
        ff = jnp.reshape(ff, [nele*odim])
        # odim
        ff = jnp.sum(ff, axis=0)
    return ff

    
def fermi_net_orbitals_orig(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: Any,
):
    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms)
    if options.hidden_dims.use_ae_feat:
        ae_features, ee_features = options.feature_layer.apply(
            ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input'])
    else:
        ae_features = None
    dpo = dp_orbitals(params, pos, atoms, charges, nspins, options, 
                      ae_features=ae_features)
    h_to_orbitals = dpo
    #
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
        envelope_factor = options.envelope.apply(
            ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope'])
        h_to_orbitals = envelope_factor * h_to_orbitals
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0)
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = network_blocks.array_partitions(active_spin_channels)
    # Create orbitals.
    orbitals = [
        network_blocks.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params['orbital'])
    ]
    # Apply envelopes if required.
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
        ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
        r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
        r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
        for i in range(len(active_spin_channels)):
            orbitals[i] = orbitals[i] * options.envelope.apply(
                ae=ae_channels[i],
                r_ae=r_ae_channels[i],
                r_ee=r_ee_channels[i],
                **params['envelope'][i],
            )    
    # Reshape into matrices.
    shapes = [(spin, -1, sum(nspins) if options.full_det else spin)
            for spin in active_spin_channels]
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    if options.full_det:
        orbitals = [jnp.concatenate(orbitals, axis=1)]
    
    return orbitals, (ae, r_ae, r_ee)


def fermi_net_orbitals_gemi(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: Any,
):
    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms)
    dpo = dp_orbitals(params, pos, atoms, charges, nspins, options)
    h_to_orbitals = dpo
    #
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
        raise RuntimeError('geminal does not supported envlop type', options.envelope.apply_type)
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0)
    # [nele0 x nodims, nele1 x nodims]
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = network_blocks.array_partitions(active_spin_channels)
    nodims = h_to_orbitals[0].shape[1]
    numb_pad = nspins[0] - nspins[1]

    if numb_pad > 0:
        tmp_o = []
        for idx,ii in enumerate(h_to_orbitals):
            tmp_o.append(
                jax.lax.dynamic_slice(ii, (0, 0), (nspins[idx], nodims - numb_pad)))
        pad_mat = jax.lax.dynamic_slice(h_to_orbitals[0], (0, nodims - numb_pad), (nspins[0], numb_pad))
        # nele0 x npad
        pad_mat = jnp.reshape(pad_mat, [nspins[0], numb_pad, 1])
        h_to_orbitals = tmp_o    
    # nele0 x (nodims x ndet)
    tmp_orb = network_blocks.linear_layer(h_to_orbitals[0], params['gemi'][0]['w'])
    # nele0 x nodims x ndet    
    tmp_orb = tmp_orb.reshape([nspins[0], nodims - numb_pad, -1,])
    ndet = tmp_orb.shape[-1]
    # nele0 x nele1 x ndet
    tmp_orb = jnp.einsum('ijk,lj->ilk', tmp_orb, h_to_orbitals[1]) / (float(nodims - numb_pad))
    if numb_pad > 0:
        # nele0 x npad x ndet
        pad_mat = jnp.tile(pad_mat, [1,1,ndet])
        # nele0 x nele0 x ndet
        tmp_orb = jnp.concatenate([tmp_orb, pad_mat], axis=1)
    tmp_orb = jnp.transpose(tmp_orb, [2, 0, 1])
    # print('===========================', jnp.linalg.det(tmp_orb))
    # id_print(h_to_orbitals[0])
    # id_print(h_to_orbitals[1])
    # id_print(params['gemi'][0]['w'])
    # id_print(tmp_orb)
    # id_print(jnp.linalg.det(tmp_orb))
    # id_print(tmp_orb[0])
    # id_print(h_to_orbitals[0])
    # id_print(h_to_orbitals[1])
    # id_print(pos[:6])
    # id_print(pos[6:])
    orbitals = [tmp_orb]    
    
    return orbitals, (ae, r_ae, r_ee)


def fermi_net_orbitals(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: Any,
):
    if options.hidden_dims.geminal :
        return fermi_net_orbitals_gemi(
            params, pos, atoms, charges, nspins, options)
    else:
        return fermi_net_orbitals_orig(
            params, pos, atoms, charges, nspins, options)
        
