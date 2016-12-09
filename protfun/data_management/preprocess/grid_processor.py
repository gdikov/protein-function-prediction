from protfun.layers import MoleculeMapLayer
import lasagne
import theano
import numpy as np
import os
import colorlog as log
import logging
import cPickle

log.basicConfig(level=logging.DEBUG)


class GridProcessor(object):
    def __init__(self, data_dirs, force_process=False):
        self.dirs = data_dirs
        self.grid_dir = os.path.join(self.dirs['data'], 'grids')
        if not os.path.exists(self.grid_dir):
            os.makedirs(self.grid_dir)
        self.force_process = force_process

        dummy = lasagne.layers.InputLayer(shape=(None,))
        self.processor = MoleculeMapLayer(incomings=[dummy, dummy], minibatch_size=1, rotate=False)

        path_to_moldata = os.path.join(self.dirs['pdb'], 'mol_info.pickle')
        with open(path_to_moldata, 'r') as f:
            mol_data = cPickle.load(f)
        self.max_atoms = mol_data['max_atoms']


    def process_all(self, mol_ids):
        for mol_id in mol_ids:
            self.process(mol_id)

    def process(self, mol_id):
        if not os.path.isfile(os.path.join(self.grid_dir, "grid" + str(mol_id) + ".memmap")) or self.force_process:
            grid = self._process(mol_id)
            self._persist(grid, mol_id)

    def _process(self, mol_id):
        path_to_protein_data = os.path.join(self.dirs['moldata'], mol_id)
        coords = np.memmap(path_to_protein_data + '_coords.memmap', mode='r', dtype=np.float32).reshape(
            (-1, self.max_atoms, 3))
        charges = np.memmap(path_to_protein_data + '_charges.memmap', mode='r', dtype=np.float32).reshape(
            (-1, self.max_atoms))
        vdwradii = np.memmap(path_to_protein_data + '_vdwradii.memmap', mode='r', dtype=np.float32).reshape(
            (-1, self.max_atoms))
        n_atoms = np.memmap(path_to_protein_data + '_natoms.memmap', mode='r', dtype=np.int32)
        mol_info = [theano.shared(np.array(coords, dtype=np.float32)),
                    theano.shared(np.array(charges, dtype=np.float32)),
                    theano.shared(np.array(vdwradii, dtype=np.float32)),
                    theano.shared(np.array(n_atoms, dtype=np.int32))]
        grid = self.processor.get_output_for(mols_info=mol_info).eval()
        return grid

    def _persist(self, grid, mol_index):
        self._save_to_memmap(os.path.join(self.grid_dir, "grid" + str(mol_index) + ".memmap"), grid, np.float32)

    @staticmethod
    def _save_to_memmap(filename, data, dtype):
        tmp = np.memmap(filename, shape=data.shape, mode='w+', dtype=dtype)
        log.info("Saving grid {0}. Shape is {1}".format(filename, data.shape))
        tmp[:] = data[:]
        tmp.flush()
        del tmp
