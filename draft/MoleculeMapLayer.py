import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import gzip
import theano.tensor.nlinalg as Tnlin

floatX = theano.config.floatX
intX = np.int32  # FIXME is this the best choice? (changing would require removing and recreating memmap files)


class MoleculeMapLayer(lasagne.layers.Layer):
    '''
    This is a Lasagne layer to calculate 3D maps (electrostatic potential, and
    electron density estimated from VdW radii) of molecules (using Theano,
    i.e. on the GPU if the user wishes so).
    At initialization, the layer is told whether it should use the file
    with active or inactive compounds. When called, the layer input is an array
    of molecule indices (both for actives and inactives - the layer selects the
    respective half depending on whether it was initialized for actives or
    inactives), and the output are the 3D maps.
    Currently works faster (runtime per sample) if `minibatch_size=1` because
    otherwise `theano.tensor.switch` is slow.
    '''

    def __init__(self, incoming, active_or_inactive, minibatch_size=None, **kwargs):
        # input to layer are indices of molecule

        super(MoleculeMapLayer, self).__init__(incoming, **kwargs)  # see creating custom layer!

        if minibatch_size is None:
            minibatch_size = 1
            print "minibatch_size not provided - assuming it is `{:d}`.  " + "If this is wrong, please provide the correct one, otherwise dropout will not work.".format(
                minibatch_size)

        # Molecule file name
        if active_or_inactive == 0:
            filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/1798_inactives_cleaned.sdf")
        elif active_or_inactive == 1:
            filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/1798_actives_cleaned.sdf")

        # Create a sensible output prefix from the input file name
        split_path = os.path.splitext(filename)
        while split_path[1] == ".gz" or split_path[1] == ".sdf":
            split_path = os.path.splitext(split_path[0])
        prefix = split_path[0]

        try:
            # attempt to load saved state from memmaps
            max_atoms = np.memmap(prefix + '_max_atoms.memmap', mode='r', dtype=intX)[0]
            coords = np.memmap(prefix + '_coords.memmap', mode='r', dtype=floatX).reshape((-1, max_atoms, 3))
            charges = np.memmap(prefix + '_charges.memmap', mode='r', dtype=floatX).reshape((-1, max_atoms))
            vdwradii = np.memmap(prefix + '_vdwradii.memmap', mode='r', dtype=floatX).reshape((-1, max_atoms))
            n_atoms = np.memmap(prefix + '_n_atoms.memmap', mode='r', dtype=intX)
            atom_mask = np.memmap(prefix + '_atom_mask.memmap', mode='r', dtype=floatX).reshape((-1, max_atoms))
            self.molecules_count = n_atoms.size
        except IOError:
            # memmap files not found, create them
            print "Creating memmap files..."

            import rdkit.Chem as Chem
            import rdkit.Chem.rdPartialCharges as rdPC
            import rdkit.Chem.rdMolTransforms as rdMT

            # Make sure the .sdf molecule file exists
            if not os.path.isfile(filename):
                print "File \"" + filename + "\" does not exist"

            # Open up the file containing the molecules
            if os.path.splitext(filename)[1] == ".gz":
                infile = gzip.open(filename, "r")
            else:
                infile = open(filename, "r")

            # the SDF parser object, reads in molecules
            # there is also a random-access version of this, but it must be given
            # a filename instead of a file stream (called SDMolSupplier, or FastSDMolSupplier)
            # defined using: import rdkit.Chem as Chem
            sdread = Chem.ForwardSDMolSupplier(infile, removeHs=False)

            # Periodic table object, needed for getting VDW radii
            pt = Chem.GetPeriodicTable()

            mol_number = 0
            n_atoms = []
            molecules = [x for x in sdread]
            self.molecules_count = len(molecules)
            max_atoms = max([mol.GetNumAtoms() for mol in molecules])

            coords = np.zeros(shape=(self.molecules_count, max_atoms, 3), dtype=floatX)
            charges = np.zeros(shape=(self.molecules_count, max_atoms), dtype=floatX)
            vdwradii = np.zeros(shape=(self.molecules_count, max_atoms), dtype=floatX)
            atom_mask = np.zeros(shape=(self.molecules_count, max_atoms), dtype=floatX)

            for mol_index, mol in enumerate(molecules):
                # compute the atomic partial charges
                rdPC.ComputeGasteigerCharges(mol)

                # get the conformation of the molecule and number of atoms (3D coordinates)
                conformer = mol.GetConformer()

                # calculate the center of the molecule
                # Centroid is the center of coordinates (center of mass of unit-weight atoms)
                # Center of mass would require atomic weights for each atom: pt.GetAtomicWeight()
                center = rdMT.ComputeCentroid(conformer, ignoreHs=False)

                atoms = mol.GetAtoms()

                mol_atoms = mol.GetNumAtoms()
                n_atoms.append(mol.GetNumAtoms())
                atom_mask[mol_index, 0:mol_atoms] = 1

                def get_coords(i):
                    coord = conformer.GetAtomPosition(i)
                    return np.asarray([coord.x, coord.y, coord.z])

                # set the coordinates, charges and VDW radii
                coords[mol_index, 0:mol_atoms] = np.asarray([get_coords(i) for i in range(0, mol_atoms)]) - np.asarray(
                    [center.x, center.y, center.z])
                charges[mol_index, 0:mol_atoms] = np.asarray(
                    [float(atom.GetProp("_GasteigerCharge")) for atom in atoms])
                vdwradii[mol_index, 0:mol_atoms] = np.asarray([pt.GetRvdw(atom.GetAtomicNum()) for atom in atoms])

            n_atoms = np.asarray(n_atoms, dtype=intX)

            self.save_to_memmap(prefix + '_max_atoms.memmap', np.asarray([max_atoms], dtype=intX))
            self.save_to_memmap(prefix + '_coords.memmap', coords)
            self.save_to_memmap(prefix + '_charges.memmap', charges)
            self.save_to_memmap(prefix + '_vdwradii.memmap', vdwradii)
            self.save_to_memmap(prefix + '_n_atoms.memmap', n_atoms)
            self.save_to_memmap(prefix + '_atom_mask.memmap', atom_mask)

        print("Total number of molecules: %s" % self.molecules_count)

        # Set the grid side length and resolution in Angstroms.
        grid_side_length = float(34.5)  # FIXME choose this number such that grid coordinates are nice
        resolution = float(0.5)
        endx = grid_side_length / 2

        # +1 because N Angstroms "-" contain N+1 grid points "x": x-x-x-x-x-x-x
        self.stepx = int(grid_side_length / resolution) + 1

        # an ndarray of grid coordinates: cartesian coordinates of each voxel
        # this will be consistent across all molecules if the grid size doesn't change
        grid_coords = lasagne.utils.floatX(
            np.mgrid[-endx:endx:self.stepx * 1j, -endx:endx:self.stepx * 1j, -endx:endx:self.stepx * 1j])

        self.min_dist_from_border = 5  # in Angstrom; for random translations; TODO ok to have it on CPU?

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

        # layer options
        self.minibatch_size = minibatch_size
        self.active_or_inactive = active_or_inactive  # to know which row of the inputs (molecule numbers) to use

        # molecule data (tensors)
        self.coords = self.add_param(coords, coords.shape, 'coords', trainable=False)
        self.charges = self.add_param(charges, charges.shape, 'charges', trainable=False)
        self.vdwradii = self.add_param(vdwradii, vdwradii.shape, 'vdwradii', trainable=False)
        self.n_atoms = self.add_param(n_atoms, n_atoms.shape, 'n_atoms', trainable=False)
        self.atom_mask = self.add_param(atom_mask, atom_mask.shape, 'atom_mask', trainable=False)

    @staticmethod
    def save_to_memmap(filename, data):
        tmp = np.memmap(filename, shape=data.shape, mode='w+', dtype=floatX)
        tmp[:] = data[:]
        tmp.flush()
        del tmp

    def get_output_shape_for(self, input_shape):
        return self.minibatch_size, 2, self.stepx, self.stepx, self.stepx

    def get_output_for(self, molecule_numbers01, **kwargs):
        # select relevant molecule number, depending on active_or_inactive
        molecule_number = molecule_numbers01[self.active_or_inactive]
        # see also stackoverflow.com/questions/2640147

        random_streams = theano.sandbox.rng_mrg.MRG_RandomStreams()  # dynamic generation of values on GPU

        # random rotation matrix Q
        # randn_matrix = lasagne.random.get_rng().normal(0, 1, size=(3,3)).astype(floatX) # fixed value
        randn_matrix = random_streams.normal((3, 3), dtype=floatX)  # dynamic generation of values on GPU
        Q, R = Tnlin.qr(randn_matrix)  # see Golkov MSc thesis, Lemma 1
        # Mezzadri 2007 "How to generate random matrices from the classical compact groups"
        Q = T.dot(Q, Tnlin.AllocDiag()(T.sgn(R.diagonal())))  # stackoverflow.com/questions/30692742
        Q = Q * Tnlin.Det()(Q)  # stackoverflow.com/questions/30132036

        # apply rotation matrix to coordinates of current molecule
        coords_current = T.dot(self.coords[molecule_number], Q)

        # random translation
        random_translations = True
        print "random_translations:", random_translations
        if random_translations:
            coords_min = T.min(coords_current, axis=1, keepdims=True)
            coords_max = T.max(coords_current, axis=1, keepdims=True)
            # order of summands important, otherwise error (maybe due to broadcastable properties)
            transl_min = (-self.endx + self.min_dist_from_border) - coords_min
            transl_max = (self.endx - self.min_dist_from_border) - coords_max
            rand01 = random_streams.uniform((self.minibatch_size, 1, 3),
                                            dtype=floatX)  # unifom random in open interval ]0;1[
            rand01 = T.Rebroadcast((1, True), )(rand01)
            rand_translation = rand01 * (transl_max - transl_min) + transl_min
            # coords_current = coords_current + rand01 # FIXME
            coords_current = coords_current + rand_translation
            # coords_current = T.add(coords_current, rand_translation)
            # reshape from (1,135,3) to (135, 3)
            # coords_current = coords_current[0,:,:,:]
            # print "coords_current", getshape(coords_current)

        # coords_current = T.Rebroadcast((0,True),(1,True),(2,True),(3,True),(4,True),(5,True),)(coords_current)

        # select subarray for current molecule; extend to 5D using `None`
        cha = self.charges[molecule_number, :, None, None, None]
        vdw = self.vdwradii[molecule_number, :, None, None, None]
        ama = self.atom_mask[molecule_number, :, None, None, None]

        if self.minibatch_size == 1:
            # minibatch size 1: ignore trailing empty molecule slots
            # Apparently theano cannot select along several dimensions at the same time (maybe due to ndim of index vars?)
            # (although `self.vdwradii[molecule_number,T.arange(natoms),None,None,None]` does work within T.maximum),
            # so here is the second step:
            natoms = self.n_atoms[molecule_number[
                0]]  # [0] for correct number of dimensions of natoms (such that T.arange(natoms) works); only for minibatch_size==1
            cha = cha[:, T.arange(natoms), :, :, :]
            vdw = vdw[:, T.arange(natoms), :, :, :]
            ama = ama[:, T.arange(natoms), :, :, :]
            # coords_current = coords_current[0,:,:] # reshape from (1,135,3) to (135, 3)
            coords_current = coords_current[:, T.arange(natoms), :]

        # TODO make self.grid_coords and others as required from the start instead of reshaping here (delete memmap files first)

        # pairwise distances from all atoms to all grid points
        distances = T.sqrt(
            T.sum((self.grid_coords[None, None, :, :, :, :] - coords_current[:, :, :, None, None, None]) ** 2, axis=2))

        # "distance" from atom to grid point should never be smaller than the vdw radius of the atom (otherwise infinite proximity possible)
        distances_esp_cap = T.maximum(distances, vdw)

        # TODO: if MBsize>1: [grids with T.switch], else: [grids without T.switch but with the above "relics" - works faster]

        # grids_0: electrostatic potential in each of the 70x70x70 grid points
        # (sum over all N atoms, i.e. axis=0, so that shape turns from (N,1,70,70,70) to (1,1,70,70,70))
        # keepdims so that we have (1,70,70,70) instead of (70, 70, 70)
        # grids_1: vdw value in each of the 70x70x70 grid points (sum over all N atoms, i.e. axis=0, so that shape turns from (N,1,70,70,70) to (1,1,70,70,70))
        if self.minibatch_size == 1:
            # For `minibatch_size==1`, the selection via `T.arange(natoms)` above is a faster alternative to `T.swich`.
            grids_0 = T.sum(cha / distances_esp_cap, axis=1, keepdims=True)
            grids_1 = T.sum(T.exp((-distances ** 2) / vdw ** 2), axis=1, keepdims=True)
        else:
            # T.switch: stackoverflow.com/questions/26621341
            grids_0 = T.sum(T.switch(ama, cha / distances_esp_cap, 0), axis=1, keepdims=True)
            grids_1 = T.sum(T.switch(ama, T.exp((-distances ** 2) / vdw ** 2), 0), axis=1, keepdims=True)

        grids = T.concatenate([grids_0, grids_1], axis=1)

        # TODO is T.set_subtensor faster than T.concatenate?

        # print "grids: ", getshape(self.grids)
        return grids
