import theano
import theano.tensor as T
import lasagne
import numpy as np

import theano.tensor.nlinalg

from os import path

floatX = theano.config.floatX
intX = np.int32  # FIXME is this the best choice? (changing would require removing and recreating memmap files)


class MoleculeMapLayer(lasagne.layers.Layer):
    """
    This is a Lasagne layer to calculate 3D maps (electrostatic potential, and
    electron density estimated from VdW radii) of molecules (using Theano,
    i.e. on the GPU if the user wishes so).
    """

    def __init__(self, incoming, minibatch_size=None, grid_side=189.0, resolution=3.0, **kwargs):
        # input to layer are indices of molecule
        super(MoleculeMapLayer, self).__init__(incoming, **kwargs)
        if minibatch_size is None:
            minibatch_size = 1
            print("INFO: Minibatch size not provided - assuming {}.".format(minibatch_size))

        self.minibatch_size = minibatch_size

        # load saved state from memmaps
        path_to_moldata = path.join(path.dirname(path.realpath(__file__)), "../../data/moldata")
        max_atoms = np.memmap(path.join(path_to_moldata, 'max_atoms.memmap'), mode='r', dtype=intX)[0]
        coords = np.memmap(path.join(path_to_moldata, 'coords.memmap'), mode='r', dtype=floatX).reshape(
            (-1, max_atoms, 3))
        charges = np.memmap(path.join(path_to_moldata, 'charges.memmap'), mode='r', dtype=floatX).reshape(
            (-1, max_atoms))
        vdwradii = np.memmap(path.join(path_to_moldata, 'vdwradii.memmap'), mode='r', dtype=floatX).reshape(
            (-1, max_atoms))
        n_atoms = np.memmap(path.join(path_to_moldata, 'n_atoms.memmap'), mode='r', dtype=intX)
        atom_mask = np.memmap(path.join(path_to_moldata, 'atom_mask.memmap'), mode='r', dtype=floatX).reshape(
            (-1, max_atoms))
        print("INFO: Loaded %d molecules in molmap, max atoms: %d" % (coords.shape[0], max_atoms))
        self.max_atoms = max_atoms

        # Set the grid side length and resolution in Angstroms.
        endx = grid_side / 2

        # +1 because N Angstroms "-" contain N+1 grid points "x": x-x-x-x-x-x-x
        self.side_points_count = int(grid_side / resolution) + 1
        self.min_dist_from_border = 5  # in Angstrom; for random translations

        # an ndarray of grid coordinates: cartesian coordinates of each voxel
        # this will be consistent across all molecules if the grid size doesn't change
        grid_coords = lasagne.utils.floatX(
            np.mgrid[-endx:endx:self.side_points_count * 1j, -endx:endx:self.side_points_count * 1j,
            -endx:endx:self.side_points_count * 1j])
        grid_coords = np.reshape(grid_coords, newshape=(grid_coords.shape[0], -1))

        # share variables (on GPU)
        self.grid_coords = self.add_param(grid_coords, grid_coords.shape, 'grid_coords', trainable=False)
        endx_on_GPU = True
        if endx_on_GPU:
            endx = np.asarray([[[endx]]],
                              dtype=floatX)  # list brackets required, otherwise error later (maybe due to array shape)
            self.endx = self.add_param(endx, endx.shape, 'endx', trainable=False)
            self.min_dist_from_border = np.asarray([[[self.min_dist_from_border]]], dtype=floatX)
            self.min_dist_from_border = self.add_param(self.min_dist_from_border, self.min_dist_from_border.shape,
                                                       'min_dist_from_border', trainable=False)
            self.endx = T.Rebroadcast((1, True), (2, True), )(self.endx)
            self.min_dist_from_border = T.Rebroadcast((1, True), (2, True), )(self.min_dist_from_border)
        else:
            self.endx = endx  # TODO ok to have it on CPU?

        # molecule data (tensors)
        self.coords = self.add_param(coords, coords.shape, 'coords', trainable=False)
        self.charges = self.add_param(charges, charges.shape, 'charges', trainable=False)
        self.vdwradii = self.add_param(vdwradii, vdwradii.shape, 'vdwradii', trainable=False)
        self.n_atoms = self.add_param(n_atoms, n_atoms.shape, 'n_atoms', trainable=False)
        self.atom_mask = self.add_param(atom_mask, atom_mask.shape, 'atom_mask', trainable=False)

    def get_output_shape_for(self, input_shape):
        return self.minibatch_size, 2, self.side_points_count, self.side_points_count, self.side_points_count

    def get_output_for(self, molecule_ids, **kwargs):
        zeros = np.zeros(
            (self.minibatch_size, 1, self.side_points_count ** 3), dtype=floatX)
        grid_density = self.add_param(zeros, zeros.shape, 'grid_density', trainable=False)
        grid_esp = self.add_param(zeros, zeros.shape, 'grid_esp', trainable=False)

        free_gpu_memory = self.get_free_gpu_memory()
        pertubated_coords = self.perturbate(self.coords[molecule_ids])
        points_count = self.side_points_count

        def preprocess_molecule(idx, grid_esp, grid_density, n_atoms, coords, charges, vdwradii,
                                grid_coords):
            atoms_count = n_atoms[idx]
            current_coords = coords[idx, T.arange(atoms_count), :]
            cha = charges[idx, T.arange(atoms_count), None]
            vdw = vdwradii[idx, T.arange(atoms_count), None]

            # to determine the grid points, we need the bytes needed for the distance
            # computation between grid points and atoms
            # (n_atoms x 3 coords x 4 bytes) per (grid point, molecule)
            # add 100 % overhead to make sure there's some free memory left
            needed_per_grid_point = (atoms_count * 3 * 4) * 2
            step_size = free_gpu_memory // needed_per_grid_point
            niter = points_count ** 3 // step_size + 1

            def partial_computation(i, grid_esp, grid_density, current_coords, cha, vdw, grid_coords):
                grid_idx_start = i * step_size
                grid_idx_end = (i + 1) * step_size
                distances_i = T.sqrt(T.sum(
                    (grid_coords[None, :, grid_idx_start:grid_idx_end] - current_coords[:, :, None]) ** 2,
                    axis=1))
                # grid point distances should not be smaller then vwd radius
                # when computing ESP
                capped_distances_i = T.maximum(distances_i, vdw)

                esp_i = T.sum(cha / capped_distances_i, axis=0, keepdims=True)
                density_i = T.sum(T.exp((-distances_i ** 2) / vdw ** 2), axis=0, keepdims=True)

                # esp_i = T.sum((cha / capped_distances_i) * ama, axis=1, keepdims=True)
                # density_i = T.sum((T.exp((-distances_i ** 2) / vdw ** 2) * ama), axis=1, keepdims=True)

                grid_density = T.set_subtensor(grid_density[idx, :, grid_idx_start:grid_idx_end], density_i)
                grid_esp = T.set_subtensor(grid_esp[idx, :, grid_idx_start:grid_idx_end], esp_i)
                return grid_esp, grid_density

            result, _ = theano.scan(fn=partial_computation,
                                    sequences=T.arange(niter),
                                    outputs_info=[grid_esp, grid_density],
                                    non_sequences=[current_coords, cha, vdw, grid_coords],
                                    n_steps=niter,
                                    allow_gc=True)

            grid_esp, grid_density = result[0][-1], result[1][-1]
            return grid_esp, grid_density

        result, _ = theano.scan(fn=preprocess_molecule,
                                sequences=T.arange(self.minibatch_size),
                                outputs_info=[grid_esp, grid_density],
                                non_sequences=[self.n_atoms[molecule_ids], pertubated_coords,
                                               self.charges[molecule_ids], self.vdwradii[molecule_ids],
                                               self.grid_coords],
                                n_steps=self.minibatch_size,
                                allow_gc=True)

        grid_esp = result[0]
        grid_density = result[1]

        grid_esp = T.reshape(grid_esp, newshape=(
            self.minibatch_size, 1, self.side_points_count, self.side_points_count, self.side_points_count))
        grid_density = T.reshape(grid_density, newshape=(
            self.minibatch_size, 1, self.side_points_count, self.side_points_count, self.side_points_count))

        grids = T.concatenate([grid_esp, grid_density], axis=1)
        return grids

    @staticmethod
    def get_free_gpu_memory():
        import theano.sandbox.cuda.basic_ops as cuda
        # free gpu memory in bytes
        free_gpu_memory = cuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
        return free_gpu_memory

    def perturbate(self, coords):
        # generate a random rotation matrix Q
        random_streams = theano.sandbox.rng_mrg.MRG_RandomStreams()
        randn_matrix = random_streams.normal((3, 3), dtype=floatX)
        # QR decomposition, Q is orthogonal, see Golkov MSc thesis, Lemma 1
        Q, R = T.nlinalg.qr(randn_matrix)
        # Mezzadri 2007 "How to generate random matrices from the classical compact groups"
        Q = T.dot(Q, T.nlinalg.AllocDiag()(T.sgn(R.diagonal())))  # stackoverflow.com/questions/30692742
        Q = Q * T.nlinalg.Det()(Q)  # stackoverflow.com/questions/30132036

        # apply rotation matrix to all molecules
        perturbated_coords = T.dot(coords, Q)

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
