import theano
import theano.tensor as T
import lasagne
import numpy as np
import theano.tensor.nlinalg
import colorlog as log
import logging

log.basicConfig(level=logging.DEBUG)

floatX = theano.config.floatX
intX = np.int32


class MoleculeMapLayer(lasagne.layers.MergeLayer):
    """
    This is a Lasagne layer to calculate 3D grid maps (electrostatic potential, and
    electron density estimated from VdW radii) of molecules (using Theano, i.e. on the GPU if the user wishes so).
    """

    def __init__(self, incomings, minibatch_size=None,
                 grid_side=126.0, resolution=2.0,
                 rotate=True, use_esp=True, **kwargs):
        """
        :param incomings: list of lasagne InputLayers for coords, charges, vdwradii and n_atoms for the molecules
                          int the minibatch.
        :param minibatch_size: size of the mini-batches that will be passed to this layer
        :param grid_side: length of the grid_side in angstroms
        :param resolution: length of the side of a single voxel in the grid, in angstroms.
        :param rotate: boolean flag, whether to rotate the molecule before creating the grid or not.
        :param kwargs: ...
        """
        super(MoleculeMapLayer, self).__init__(incomings, **kwargs)
        if minibatch_size is None:
            minibatch_size = 1
            log.info("Minibatch size not provided - assuming {}.".format(minibatch_size))

        self.minibatch_size = minibatch_size
        self.rotate = rotate
        self.use_esp = use_esp

        # Set the grid side length and resolution in Angstroms.
        self.endx = grid_side / 2
        # +1 because N Angstroms "-" contain N+1 grid points "x": x-x-x-x-x-x-x
        self.side_points_count = int(grid_side / resolution) + 1
        self.min_dist_from_border = 5  # in Angstrom; for random translations
        # an ndarray of grid coordinates: cartesian coordinates of each voxel in the grid
        # the grid coordinates remain constant across all molecules
        grid_coords = lasagne.utils.floatX(
            np.mgrid[-self.endx:self.endx:self.side_points_count * 1j,
            -self.endx:self.endx:self.side_points_count * 1j,
            -self.endx:self.endx:self.side_points_count * 1j])
        grid_coords = np.reshape(grid_coords, newshape=(grid_coords.shape[0], -1))
        # share grid coordinates on the GPU
        self.grid_coords = self.add_param(grid_coords, grid_coords.shape, 'grid_coords', trainable=False)

    def get_output_shape_for(self, input_shape):
        """
        :param input_shape: not needed
        :return: the shape of the two computed grids (electron density, esp), stacked along axis 1
        """
        channel_count = 2 if self.use_esp else 1
        return self.minibatch_size, channel_count, self.side_points_count, \
               self.side_points_count, self.side_points_count

    def get_output_for(self, mols_info, **kwargs):
        """
        :param mols_info: a list of the TheanoVariables: coords, charges, vdwradii, natoms. They contain information
                         about the coordinates, atom charges, vdwradii and number of atoms for the molecules in the
                         minibatch. Dimensions:
                            coords: (minibatch_dim x atom_dim x 3)
                            charges: (minibatch_dim x atom_dim)
                            vdwradii: (minibatch_dim x atom_dim)
                            natoms: (minibatch_dim)
        :param kwargs: ...
        :return: Two grids, stacked along the axis 1: electrostatic potential and electron density.
                 Dimensions: (minibatch_dim x 2 x grid_side_size x grid_side_size x grid_side_size)
        """
        mols_coords, mols_charges, mols_vdwradii, mols_natoms = mols_info
        zeros = np.zeros(
            (self.minibatch_size, 1, self.side_points_count ** 3), dtype=floatX)
        grids_density = self.add_param(zeros, zeros.shape, 'grids_density', trainable=False)
        grids_esp = self.add_param(zeros, zeros.shape, 'grids_esp', trainable=False)

        free_gpu_memory = self.get_free_gpu_memory()
        if self.rotate:
            mols_coords = self.rotate_and_translate(mols_coords)
        points_count = self.side_points_count

        # TODO: keep in mind the declarative implementation (regular for loop) is faster
        # but it takes exponentially more time to compile as the batch_size increases
        # for i in range(0, self.minibatch_size):
        def compute_grid_per_mol(i, mol_natoms, mol_coords, mol_charges, mol_vdwradii, grid_esp, grid_density,
                                 grid_coords):
            """
            The function is used in theano scan to compute the grids for a single molecule from the mini-batch.

            :param i: the index of the molecule in the mini-batch
            :param mol_natoms: the number of atoms in this molecule
            :param mol_coords: the coordinates of the atoms in this molecule
            :param mol_charges: the charges of the atoms in this molecule
            :param mol_vdwradii: the vdwradii of the atoms in this molecule
            :param grid_esp: the ESP grid for the whole mini-batch. Here the result for all molecules is accumulated.
            :param grid_density: the el. density grid for the whole mini-batch. Also gets accumulated.
            :param grid_coords: the coordinates of the voxels in the grids (always constant)
            :return:
            """
            mol_charges = mol_charges[T.arange(mol_natoms), None]
            mol_vdwradii = mol_vdwradii[T.arange(mol_natoms), None]
            mol_coords = mol_coords[T.arange(mol_natoms), :, None]

            # add 100 % overhead to make sure there's some free memory left
            approx_extra_space_factor = 2
            # (n_atoms x 3 coords x 4 bytes) memory per (grid point, molecule)
            needed_bytes_per_grid_point = (mol_natoms * 3 * 4) * approx_extra_space_factor
            grid_points_per_step = free_gpu_memory // needed_bytes_per_grid_point
            niter = points_count ** 3 // grid_points_per_step + 1

            def compute_grid_part(j, grid_esp, grid_density, mol_coords, mol_charges, mol_vdwradii, grid_coords):
                """
                The function is used to iteratively compute parts of the grid for a single molecule. The size of those
                parts is dynamically optimized to fit into GPU memory.

                :param j: index of the current part of the grid being computed
                :param grid_esp: the ESP grid for the whole mini-batch. Here the result for all molecules is accumulated
                :param grid_density: the el. density grid for the whole mini-batch. Also gets accumulated.
                :param mol_coords: the coordinates of the atoms in the current molecule
                :param mol_charges: the charges of the atoms in the current molecule
                :param mol_vdwradii: the vdwradii of the atoms in the current molecule
                :param grid_coords: the coordinates of the voxels in the grids (always constant)
                :return:
                """
                grid_idx_start = j * grid_points_per_step
                grid_idx_end = (j + 1) * grid_points_per_step

                distances_i = T.sqrt(
                    T.sum((grid_coords[None, :, grid_idx_start:grid_idx_end] - mol_coords) ** 2, axis=1))

                # grid point distances should not be smaller then vwd radius when computing ESP
                capped_distances_i = T.maximum(distances_i, mol_vdwradii)

                if self.use_esp:
                    esp_i = T.sum(mol_charges / capped_distances_i, axis=0, keepdims=True)
                density_i = T.sum(T.exp((-distances_i ** 2) / mol_vdwradii ** 2), axis=0, keepdims=True)

                grid_density = T.set_subtensor(grid_density[i, :, grid_idx_start:grid_idx_end], density_i)
                if self.use_esp:
                    grid_esp = T.set_subtensor(grid_esp[i, :, grid_idx_start:grid_idx_end], esp_i)
                return grid_esp, grid_density

            partial_result, _ = theano.scan(fn=compute_grid_part,
                                            sequences=T.arange(niter),
                                            outputs_info=[grid_esp, grid_density],
                                            non_sequences=[mol_coords,
                                                           mol_charges,
                                                           mol_vdwradii,
                                                           grid_coords],
                                            n_steps=niter,
                                            allow_gc=True)

            grid_esp, grid_density = partial_result[0][-1], partial_result[1][-1]
            return grid_esp, grid_density

        result, _ = theano.scan(fn=compute_grid_per_mol,
                                sequences=[T.arange(self.minibatch_size),
                                           mols_natoms,
                                           mols_coords,
                                           mols_charges,
                                           mols_vdwradii],
                                outputs_info=[grids_esp, grids_density],
                                non_sequences=self.grid_coords,
                                n_steps=self.minibatch_size,
                                allow_gc=True)

        if self.use_esp:
            grids_esp = result[0][-1]
            grids_esp = T.reshape(grids_esp, newshape=(
                self.minibatch_size, 1, self.side_points_count, self.side_points_count, self.side_points_count))

        grids_density = result[1][-1]
        grids_density = T.reshape(grids_density, newshape=(
            self.minibatch_size, 1, self.side_points_count, self.side_points_count, self.side_points_count))

        if self.use_esp:
            grids = T.concatenate([grids_esp, grids_density], axis=1)
        else:
            grids = grids_density
        return grids

    @staticmethod
    def get_free_gpu_memory():
        """
        :return: the GPU memory that is currently free on the machine.
        """
        import theano.sandbox.cuda.basic_ops as cuda
        # free gpu memory in bytes
        free_gpu_memory = cuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
        return free_gpu_memory

    def rotate_and_translate(self, coords, golkov=False, angle_std=0.392):  # pi/8 ~= 0.392
        """
        Rotates and translates the coordinates of a molecule.

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
            Q = T.dot(Q, T.nlinalg.AllocDiag()(T.sgn(R.diagonal())))  # stackoverflow.com/questions/30692742
            Q = Q * T.nlinalg.Det()(Q)  # stackoverflow.com/questions/30132036
            R = Q
        else:
            angle = random_streams.normal((3,), avg=0., std=angle_std, ndim=1, dtype=floatX)
            R_X = T.as_tensor([1, 0, 0,
                               0, T.cos(angle[0]), -T.sin(angle[0]),
                               0, T.sin(angle[0]), T.cos(angle[0])]).reshape((3, 3))
            R_Y = T.as_tensor([T.cos(angle[1]), 0, -T.sin(angle[1]),
                               0, 1, 0,
                               T.sin(angle[1]), 0, T.cos(angle[1])]).reshape((3, 3))
            R_Z = T.as_tensor([T.cos(angle[2]), -T.sin(angle[2]), 0,
                               T.sin(angle[2]), T.cos(angle[2]), 0,
                               0, 0, 1]).reshape((3, 3))
            R = T.dot(T.dot(R_Z, R_Y), R_X)

        # apply rotation matrix to all molecules
        perturbated_coords = T.dot(coords, R)

        coords_min = T.min(perturbated_coords, axis=1, keepdims=True)
        coords_max = T.max(perturbated_coords, axis=1, keepdims=True)
        # order of summands important, otherwise error (maybe due to broadcastable properties)
        transl_min = (-self.endx + self.min_dist_from_border) - coords_min
        transl_max = (self.endx - self.min_dist_from_border) - coords_max
        rand01 = random_streams.uniform((self.minibatch_size, 1, 3),
                                        dtype=floatX)  # unifom random in open interval ]0;1[
        rand01 = T.Rebroadcast((1, True), )(rand01)
        rand_translation = rand01 * (transl_max - transl_min) + transl_min
        perturbated_coords += rand_translation
        return perturbated_coords
