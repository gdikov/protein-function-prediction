import lasagne
import numpy as np
import theano
import theano.tensor.nlinalg
import theano.tensor as T
import colorlog as log
import logging

from protfun.visualizer.molview import MoleculeView

log.basicConfig(level=logging.DEBUG)
floatX = theano.config.floatX
intX = np.int32


class MoleculeMapLayer(lasagne.layers.MergeLayer):
    """
    This is a Lasagne layer to calculate 3D grid maps (electron density estimated from VdW radii)
    of molecules. (using Theano, i.e. on the GPU).

    Usage::
        >>> from lasagne.layers import InputLayer
        >>> minibatch_size = 8
        >>> dummy_coords_input = InputLayer(shape=(minibatch_size, None, None))
        >>> dummy_vdwradii_input = InputLayer(shape=(minibatch_size, None))
        >>> dummy_natoms_input = InputLayer(shape=(minibatch_size,))
        >>> molmap_layer = MoleculeMapLayer(
        >>>    incomings=[dummy_coords_input, dummy_vdwradii_input, dummy_natoms_input],
        >>>    minibatch_size=minibatch_size, rotate=True)

    """

    def __init__(self, incomings, minibatch_size=None, grid_side=127.0, resolution=1.0, rotate=True,
                 **kwargs):
        """
        :param incomings: list of lasagne InputLayers for coords, vdwradii and n_atoms for the
            molecules in the minibatch.
        :param minibatch_size: size of the mini-batches that will be passed to this layer
        :param grid_side: length of the grid_side (in angstroms, not number of points)
        :param resolution: length of the side of a single voxel in the grid, in angstroms.
        :param rotate: boolean flag, whether to rotate the molecule before creating the grid or not.
        :param kwargs: lasagne **kwargs
        """
        super(MoleculeMapLayer, self).__init__(incomings, **kwargs)
        if minibatch_size is None:
            minibatch_size = 1
            log.info("Minibatch size not provided - assuming {}.".format(minibatch_size))

        self.minibatch_size = minibatch_size
        self.rotate = rotate

        # Set the grid side length and resolution in Angstroms.
        self.endx = grid_side / 2
        # +1 because N Angstroms "-" contain N+1 grid points "x": x-x-x-x-x-x-x
        self.side_points_count = int(grid_side / resolution) + 1
        # minimal distance from the borders in Angstrom; for random translations
        self.min_dist_from_border = 5

        # an ndarray of grid coordinates: cartesian coordinates of each voxel in the grid
        # the grid coordinates remain constant across all molecules
        grid_coords = lasagne.utils.floatX(
            np.mgrid[-self.endx:self.endx:self.side_points_count * 1j,
            -self.endx:self.endx:self.side_points_count * 1j,
            -self.endx:self.endx:self.side_points_count * 1j])
        # flatten the grid coordinates for easier computations later on
        grid_coords = np.reshape(grid_coords, newshape=(grid_coords.shape[0], -1))
        # share grid coordinates on the GPU (as Theano variable)
        self.grid_coords = self.add_param(grid_coords, grid_coords.shape, 'grid_coords',
                                          trainable=False)

    def get_output_shape_for(self, input_shape):
        """
        :param input_shape: not needed
        :return: the shape of the two computed grid (electron density)
        """
        return self.minibatch_size, 1, self.side_points_count, self.side_points_count, self.side_points_count

    def get_output_for(self, mols_info, **kwargs):
        """
        :param mols_info: a list of the TheanoVariables: coords, vdwradii, natoms.
            They contain information about the coordinates, atom charges, vdwradii and number of
            atoms for the molecules in the  minibatch. Dimensions:
                            coords: (minibatch_dim x atom_dim x 3)
                            vdwradii: (minibatch_dim x atom_dim)
                            natoms: (minibatch_dim)
        :param kwargs: ...
        :return: A 3D grid for each molecule in the minibatch, with the computed electron density.
                 Dimensions: (minibatch_dim x 1 x grid_side_size x grid_side_size x grid_side_size)
        """
        mols_coords, mols_vdwradii, mols_natoms = mols_info
        if self.rotate:
            mols_coords = self.rotate_and_translate(mols_coords)

        # initialize the computed electron density with 0s, it will be computed part by part
        # in place.
        zeros = np.zeros((self.minibatch_size, 1, self.side_points_count ** 3), dtype=floatX)
        grids_density = self.add_param(zeros, zeros.shape, 'grids_density', trainable=False)

        # determine the free GPU memory
        free_gpu_memory = self.get_free_gpu_memory()
        points_count = self.side_points_count

        # NOTE: keep in mind the declarative implementation (regular for loop) is slightly faster
        # than using theano.scan(), but it takes exponentially more time to compile as the
        # minibatch_size increases and does not allow for different array sizes.
        def compute_grid_per_mol(i, mol_natoms, mol_coords, mol_vdwradii, grid_density,
                                 grid_coords):
            """
            The function is used in theano scan to compute the electron density grid for a single
            molecule in the mini-batch.

            :param i: the index of the molecule in the mini-batch
            :param mol_natoms: the number of atoms in this molecule
            :param mol_coords: the coordinates of the atoms in this molecule
            :param mol_vdwradii: the vdwradii of the atoms in this molecule
            :param grid_density: the el. density grid for the whole mini-batch (not only this
            molecule). This function will update the part for this molecule in place.
            :param grid_coords: the coordinates of all voxels in a computed grid (always constant)
            :return: the grid_density array for the whole minibatch, with the part for the current
                molecule already computed.
            """
            # make the arrays broadcastable
            mol_vdwradii = mol_vdwradii[T.arange(mol_natoms), None]
            mol_coords = mol_coords[T.arange(mol_natoms), :, None]

            # add 100 % overhead to make sure there's some free memory left on the GPU
            approx_extra_space_factor = 2
            # (n_atoms x 3 coords x 4 bytes) memory per (grid point, molecule)
            needed_bytes_per_grid_point = (mol_natoms * 3 * 4) * approx_extra_space_factor
            # determine how many grid points can be computed at once
            grid_points_per_step = free_gpu_memory // needed_bytes_per_grid_point
            # determine how many iterations will be needed to compute the whole grid for the current
            # molecule
            niter = points_count ** 3 // grid_points_per_step + 1

            def compute_grid_part(j, grid_density, mol_coords, mol_vdwradii, grid_coords):
                """
                The function is used to iteratively compute parts of the grid for a single molecule.
                Thus the computation of an electron density grid for a single molecule is split into
                multiple parts.
                The size of those parts is dynamically optimized to fit into GPU memory.

                :param j: index of the current part of the grid being computed
                :param grid_density: the el. density grid for the whole mini-batch.
                :param mol_coords: the coordinates of the atoms in the current molecule
                :param mol_vdwradii: the vdwradii of the atoms in the current molecule
                :param grid_coords: the coordinates of the voxels in the grids (always constant)
                :return: the grid_density (whole minibatch) with a part of the current molecule's
                    grid already computed.
                """
                # start and end indices of the current part being computed
                grid_idx_start = j * grid_points_per_step
                grid_idx_end = (j + 1) * grid_points_per_step

                # pairwise distances between atom coordinates and the current part of grid points
                distances_i = T.sqrt(
                    T.sum((grid_coords[None, :, grid_idx_start:grid_idx_end] - mol_coords) ** 2,
                          axis=1))

                # electron density computation for the current part
                density_i = T.sum(T.exp((-distances_i ** 2) / mol_vdwradii ** 2), axis=0,
                                  keepdims=True)

                # set the computed values in the overall array for the whole minibatch
                # i is the molecule index
                # grid_idx_start:grid_idx_end defines the part of the grid for the current
                # molecule
                grid_density = T.set_subtensor(grid_density[i, :, grid_idx_start:grid_idx_end],
                                               density_i)
                return grid_density

            # this theano.scan iterates over the grid parts for the current molecule in the
            # mini-batch
            partial_result, _ = theano.scan(fn=compute_grid_part,
                                            sequences=T.arange(niter),
                                            outputs_info=grid_density,
                                            non_sequences=[mol_coords,
                                                           mol_vdwradii,
                                                           grid_coords],
                                            n_steps=niter,
                                            allow_gc=True)

            grid_density = partial_result[-1]
            return grid_density

        # this theano.scan iterates over each molecule in the mini-batch
        result, _ = theano.scan(fn=compute_grid_per_mol,
                                sequences=[T.arange(self.minibatch_size),
                                           mols_natoms,
                                           mols_coords,
                                           mols_vdwradii],
                                outputs_info=grids_density,
                                non_sequences=self.grid_coords,
                                n_steps=self.minibatch_size,
                                allow_gc=True)

        # result[-1] has the final computation of the electron density for all molecules
        # in the minibatch
        grids_density = result[-1]
        grids_density = T.reshape(grids_density, newshape=(
            self.minibatch_size, 1, self.side_points_count,
            self.side_points_count, self.side_points_count))
        return grids_density

    @staticmethod
    def get_free_gpu_memory():
        """
        :return: the GPU memory that is currently free on the machine.
        """
        import theano.sandbox.cuda.basic_ops as cuda
        # free gpu memory in bytes
        free_gpu_memory = cuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
        return free_gpu_memory

    def rotate_and_translate(self, coords, golkov=False, angle_std=0.392):
        """
        Rotates and translates the coordinates of a molecule.
        Two options exist for the rotation matrix: either to create it through a QR decomposition
        (cf. Golkov MSc thesis), or to use Given's rotation matrices.

        :param coords: the coordinates of the molecule that we want to rotate and translate
        :param golkov: boolean: True - use QR decomposition to obtain a rotation matrix
                                False - use Givens rotations to define the rotation matrix
        :param angle_std: only for the Given's rotations case: sets the std. deviation of the
                    rotation angle (which is sampled from a gaussian with mean 0).
        :return: the rotated and translated coordinates
        """
        # generate a random rotation matrix Q
        random_streams = T.shared_randomstreams.RandomStreams()

        if golkov:
            randn_matrix = random_streams.normal((3, 3), dtype=floatX)
            # QR decomposition, Q is orthogonal, see Golkov MSc thesis, Lemma 1
            Q, R = T.nlinalg.qr(randn_matrix)
            # Mezzadri 2007 "How to generate random matrices from the classical compact groups"
            Q = T.dot(Q, T.nlinalg.AllocDiag()(
                T.sgn(R.diagonal())))  # stackoverflow.com/questions/30692742
            Q = Q * T.nlinalg.Det()(Q)  # stackoverflow.com/questions/30132036
            R = Q
        else:
            angle = random_streams.normal((3,), avg=0., std=angle_std, ndim=1,
                                          dtype=floatX)
            R_X = T.as_tensor([1, 0, 0,
                               0, T.cos(angle[0]), -T.sin(angle[0]),
                               0, T.sin(angle[0]), T.cos(angle[0])]).reshape(
                (3, 3))
            R_Y = T.as_tensor([T.cos(angle[1]), 0, -T.sin(angle[1]),
                               0, 1, 0,
                               T.sin(angle[1]), 0, T.cos(angle[1])]).reshape(
                (3, 3))
            R_Z = T.as_tensor([T.cos(angle[2]), -T.sin(angle[2]), 0,
                               T.sin(angle[2]), T.cos(angle[2]), 0,
                               0, 0, 1]).reshape((3, 3))
            R = T.dot(T.dot(R_Z, R_Y), R_X)

        # apply rotation matrix to all molecules
        perturbated_coords = T.dot(coords, R)

        # determine a random translation vector
        coords_min = T.min(perturbated_coords, axis=1, keepdims=True)
        coords_max = T.max(perturbated_coords, axis=1, keepdims=True)
        transl_min = (-self.endx + self.min_dist_from_border) - coords_min
        transl_max = (self.endx - self.min_dist_from_border) - coords_max
        rand01 = random_streams.uniform((self.minibatch_size, 1, 3),
                                        dtype=floatX)  # unifom random in open interval ]0;1[
        rand01 = T.Rebroadcast((1, True), )(rand01)
        rand_translation = rand01 * (transl_max - transl_min) + transl_min
        perturbated_coords += rand_translation
        return perturbated_coords


if __name__ == "__main__":
    """
    A quick test of the MoleculeMapLayer.
    The computed electron density grid is visualized using the molecule view.
    """
    import os

    # replace this with something meaningful on your system
    path_to_moldata = "/home/valor/workspace/DLCV_ProtFun/data/full/processed_single_64/1A0H"

    # init the molmap layer
    coords_var = T.tensor3('coords')
    vdwradii_var = T.matrix('vdwradii')
    n_atoms_var = T.ivector('n_atoms')
    coords_input = lasagne.layers.InputLayer(shape=(1, None, None),
                                             input_var=coords_var)
    vdwradii_input = lasagne.layers.InputLayer(shape=(1, None),
                                               input_var=vdwradii_var)
    natoms_input = lasagne.layers.InputLayer(shape=(1,),
                                             input_var=n_atoms_var)
    molmap_layer = MoleculeMapLayer(incomings=[coords_input, vdwradii_input, natoms_input],
                                    minibatch_size=1, grid_side=126.0, resolution=2.0, rotate=True)

    # test the molmap layer
    coords = np.memmap(os.path.join(path_to_moldata, 'coords.memmap'), mode='r',
                       dtype=floatX).reshape((1, -1, 3))
    vdwradii = np.memmap(os.path.join(path_to_moldata, 'vdwradii.memmap'), mode='r',
                         dtype=floatX).reshape(1, -1)
    n_atoms = np.array(vdwradii.shape[1], dtype=np.int32).reshape(1, )

    mol_info = [theano.shared(coords), theano.shared(vdwradii), theano.shared(n_atoms)]

    grids = molmap_layer.get_output_for(mols_info=mol_info).eval()
    viewer = MoleculeView(data_dir=path_to_moldata, data={"density": grids[0, 0]},
                          info={"name": "test"})
    viewer.density3d()
