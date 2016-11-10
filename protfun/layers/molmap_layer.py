import theano
import theano.tensor as T
import lasagne
import numpy as np
import os
import theano.tensor.nlinalg

from os import path, listdir

floatX = theano.config.floatX
intX = np.int32  # FIXME is this the best choice? (changing would require removing and recreating memmap files)


class MoleculeMapLayer(lasagne.layers.Layer):
    """
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
    """

    def __init__(self, incoming, minibatch_size=None, grid_side=62.0, resolution=2.0, **kwargs):
        # input to layer are indices of molecule
        super(MoleculeMapLayer, self).__init__(incoming, **kwargs)
        if minibatch_size is None:
            minibatch_size = 1
            print("minibatch_size not provided - assuming {}.".format(minibatch_size))

        self.minibatch_size = minibatch_size

        # PDB data directory
        prefix = path.join(path.dirname(path.realpath(__file__)), "../../data")
        dir_path = path.join(prefix, 'test')

        try:
            # attempt to load saved state from memmaps
            max_atoms = np.memmap(path.join(prefix, 'max_atoms.memmap'), mode='r', dtype=intX)[0]
            coords = np.memmap(path.join(prefix, 'coords.memmap'), mode='r', dtype=floatX).reshape((-1, max_atoms, 3))
            charges = np.memmap(path.join(prefix, 'charges.memmap'), mode='r', dtype=floatX).reshape((-1, max_atoms))
            vdwradii = np.memmap(path.join(prefix, 'vdwradii.memmap'), mode='r', dtype=floatX).reshape((-1, max_atoms))
            n_atoms = np.memmap(path.join(prefix, 'n_atoms.memmap'), mode='r', dtype=intX)
            atom_mask = np.memmap(path.join(prefix, 'atom_mask.memmap'), mode='r', dtype=floatX).reshape(
                (-1, max_atoms))
            self.molecules_count = n_atoms.size
        except IOError:
            # memmap files not found, create them
            print "Creating memmap files..."
            import rdkit.Chem as Chem
            import rdkit.Chem.rdPartialCharges as rdPC
            import rdkit.Chem.rdMolTransforms as rdMT

            fetcher = PDBFetcher(dir_path=dir_path, count=1)
            n_atoms = []

            molecules = fetcher.get_molecules()

            # Periodic table object, needed for getting VDW radii
            pt = Chem.GetPeriodicTable()

            self.molecules_count = len(molecules)
            max_atoms = max([mol.GetNumAtoms() for mol in molecules])

            coords = np.zeros(shape=(self.molecules_count, max_atoms, 3), dtype=floatX)
            charges = np.zeros(shape=(self.molecules_count, max_atoms), dtype=floatX)
            vdwradii = np.ones(shape=(self.molecules_count, max_atoms), dtype=floatX)
            atom_mask = np.zeros(shape=(self.molecules_count, max_atoms), dtype=floatX)

            for mol_index, mol in enumerate(molecules):
                # compute the atomic partial charges
                rdPC.ComputeGasteigerCharges(mol, throwOnParamFailure=True)

                # get the conformation of the molecule and number of atoms (3D coordinates)
                conformer = mol.GetConformer()

                # calculate the center of the molecule
                # Centroid is the center of coordinates (center of mass of unit-weight atoms)
                # Center of mass would require atomic weights for each atom: pt.GetAtomicWeight()
                center = rdMT.ComputeCentroid(conformer, ignoreHs=False)

                atoms_count = mol.GetNumAtoms()
                atoms = mol.GetAtoms()

                n_atoms.append(atoms_count)
                atom_mask[mol_index, 0:atoms_count] = 1

                def get_coords(i):
                    coord = conformer.GetAtomPosition(i)
                    return np.asarray([coord.x, coord.y, coord.z])

                # set the coordinates, charges and VDW radii
                coords[mol_index, 0:atoms_count] = np.asarray(
                    [get_coords(i) for i in range(0, atoms_count)]) - np.asarray(
                    [center.x, center.y, center.z])
                charges[mol_index, 0:atoms_count] = np.asarray(
                    [float(atom.GetProp("_GasteigerCharge")) for atom in atoms])
                vdwradii[mol_index, 0:atoms_count] = np.asarray([pt.GetRvdw(atom.GetAtomicNum()) for atom in atoms])

            n_atoms = np.asarray(n_atoms, dtype=intX)

            self.save_to_memmap(path.join(prefix, 'max_atoms.memmap'), np.asarray([max_atoms], dtype=intX), dtype=intX)
            self.save_to_memmap(path.join(prefix, 'coords.memmap'), coords, dtype=floatX)
            self.save_to_memmap(path.join(prefix, 'charges.memmap'), charges, dtype=floatX)
            self.save_to_memmap(path.join(prefix, 'vdwradii.memmap'), vdwradii, dtype=floatX)
            self.save_to_memmap(path.join(prefix, 'n_atoms.memmap'), n_atoms, dtype=intX)
            self.save_to_memmap(path.join(prefix, 'atom_mask.memmap'), atom_mask, dtype=floatX)

        print("Total number of molecules: %s" % self.molecules_count)

        # Set the grid side length and resolution in Angstroms.
        endx = grid_side / 2

        # +1 because N Angstroms "-" contain N+1 grid points "x": x-x-x-x-x-x-x
        self.grid_points_count = int(grid_side / resolution) + 1

        # an ndarray of grid coordinates: cartesian coordinates of each voxel
        # this will be consistent across all molecules if the grid size doesn't change
        grid_coords = lasagne.utils.floatX(
            np.mgrid[-endx:endx:self.grid_points_count * 1j, -endx:endx:self.grid_points_count * 1j,
            -endx:endx:self.grid_points_count * 1j])
        self.min_dist_from_border = 5  # in Angstrom; for random translations

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
        self.batch_size = minibatch_size

        # molecule data (tensors)
        self.coords = self.add_param(coords, coords.shape, 'coords', trainable=False)
        self.charges = self.add_param(charges, charges.shape, 'charges', trainable=False)
        self.vdwradii = self.add_param(vdwradii, vdwradii.shape, 'vdwradii', trainable=False)
        self.n_atoms = self.add_param(n_atoms, n_atoms.shape, 'n_atoms', trainable=False)
        self.atom_mask = self.add_param(atom_mask, atom_mask.shape, 'atom_mask', trainable=False)

    @staticmethod
    def save_to_memmap(filename, data, dtype):
        tmp = np.memmap(filename, shape=data.shape, mode='w+', dtype=dtype)
        print("Saving memmap. Shape of {} is {}".format(filename, data.shape))
        tmp[:] = data[:]
        tmp.flush()
        del tmp

    def get_output_shape_for(self, input_shape):
        return self.batch_size, 2, self.grid_points_count, self.grid_points_count, self.grid_points_count

    def get_output_for(self, molecule_ids, **kwargs):
        current_coords = self.pertubate(self.coords[molecule_ids])

        # select subarray for current molecule; extend to 5D using `None`
        cha = self.charges[molecule_ids, :, None, None, None]
        vdw = self.vdwradii[molecule_ids, :, None, None, None]
        ama = self.atom_mask[molecule_ids, :, None, None, None]

        # pairwise distances from all atoms to all grid points
        distances = T.sqrt(
            T.sum((self.grid_coords[None, None, :, :, :, :] - current_coords[:, :, :, None, None, None]) ** 2, axis=2))

        # "distance" from atom to grid point should never be smaller than the vdw radius of the atom
        # (otherwise infinite proximity possible)
        distances_esp_cap = T.maximum(distances, vdw)

        # grids_0: electrostatic potential in each of the 70x70x70 grid points
        # grids_1: vdw value in each of the 70x70x70 grid points
        if self.minibatch_size == 1:
            grids_0 = T.sum(cha / distances_esp_cap, axis=1, keepdims=True)
            grids_1 = T.sum(T.exp((-distances ** 2) / vdw ** 2), axis=1, keepdims=True)
        else:
            grids_0 = T.sum((cha / distances_esp_cap) * ama, axis=1,
                            keepdims=True)
            grids_1 = T.sum((T.exp((-distances ** 2) / vdw ** 2) * ama), axis=1, keepdims=True)

        grids = T.concatenate([grids_0, grids_1], axis=1)

        # print "grids: ", grids.shape.eval()
        return grids

    def pertubate(self, coords):
        print "Doing random rotations ... "
        # generate a random rotation matrix Q
        random_streams = theano.sandbox.rng_mrg.MRG_RandomStreams()
        randn_matrix = random_streams.normal((3, 3), dtype=floatX)
        # QR decomposition, Q is orthogonal, see Golkov MSc thesis, Lemma 1
        Q, R = T.nlinalg.qr(randn_matrix)
        # Mezzadri 2007 "How to generate random matrices from the classical compact groups"
        Q = T.dot(Q, T.nlinalg.AllocDiag()(T.sgn(R.diagonal())))  # stackoverflow.com/questions/30692742
        Q = Q * T.nlinalg.Det()(Q)  # stackoverflow.com/questions/30132036

        # apply rotation matrix to all molecules
        pertubated_coords = T.dot(coords, Q)

        print "Doing random translations ... "
        coords_min = T.min(pertubated_coords, axis=1, keepdims=True)
        coords_max = T.max(pertubated_coords, axis=1, keepdims=True)
        # order of summands important, otherwise error (maybe due to broadcastable properties)
        transl_min = (-self.endx + self.min_dist_from_border) - coords_min
        transl_max = (self.endx - self.min_dist_from_border) - coords_max
        rand01 = random_streams.uniform((self.batch_size, 1, 3), dtype=floatX)  # unifom random in open interval ]0;1[
        rand01 = T.Rebroadcast((1, True), )(rand01)
        rand_translation = rand01 * (transl_max - transl_min) + transl_min
        pertubated_coords += rand_translation
        return pertubated_coords


class PDBFetcher(object):
    """
    PDBFetcher can download PDB files from the PDB and
    also convert the files to rdkit molecules.
    """

    def __init__(self, dir_path, count=None):
        self.dir_path = dir_path
        self.count = count

    def download_pdb(self):
        from Bio.PDB import PDBList
        pl = PDBList(pdb=self.dir_path)
        pl.flat_tree = 1
        pl.download_entire_pdb()

    def download_protein(self, pdb_code):
        from Bio.PDB import PDBList
        pl = PDBList(pdb=self.dir_path)
        pl.flat_tree = 1
        pl.retrieve_pdb_file(pdb_code=pdb_code)

    def get_molecules(self):
        import rdkit.Chem as Chem
        import rdkit.Chem.rdmolops as rdMO
        files = [path.join(self.dir_path, f) for f in listdir(self.dir_path)
                 if f.endswith(".ent") or f.endswith(".pdb")]
        if self.count is None:
            self.count = len(files)
        files = files[:self.count]
        res = [Chem.MolFromPDBFile(molFileName=f) for f in files]
        return [rdMO.AddHs(mol, addCoords=True) for mol in res if mol is not None]


# run this if you wish to download the PDB database
if __name__ == "__main__":
    pdb_dir = path.join(path.dirname(path.realpath(__file__)), "../data")
    if not path.exists(pdb_dir):
        os.makedirs(pdb_dir)
    fetcher = PDBFetcher(dir_path=pdb_dir)
    fetcher.download_pdb()
