from protfun.layers import MoleculeMapLayer
import lasagne
import theano
import numpy as np
import os
import colorlog as log
import logging

log.basicConfig(level=logging.DEBUG)


class GridProcessor(object):
    def __init__(self, folder_name='grids', force_process=False):
        self.grid_dir = os.path.join(os.path.dirname(__file__), '../../data_old', folder_name)
        if not os.path.exists(self.grid_dir):
            os.makedirs(self.grid_dir)
        self.force_process = force_process

        dummy = lasagne.layers.InputLayer(shape=(None,))
        self.processor = MoleculeMapLayer(incomings=[dummy, dummy], minibatch_size=1, rotate=False)

        path_to_moldata = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data_old/moldata")
        max_atoms = np.memmap(os.path.join(path_to_moldata, 'max_atoms.memmap'), mode='r', dtype=np.int32)[0]
        self.coords = np.memmap(os.path.join(path_to_moldata, 'coords.memmap'), mode='r', dtype=np.float32).reshape(
            (-1, max_atoms, 3))
        self.charges = np.memmap(os.path.join(path_to_moldata, 'charges.memmap'), mode='r', dtype=np.float32).reshape(
            (-1, max_atoms))
        self.vdwradii = np.memmap(os.path.join(path_to_moldata, 'vdwradii.memmap'), mode='r', dtype=np.float32).reshape(
            (-1, max_atoms))
        self.n_atoms = np.memmap(os.path.join(path_to_moldata, 'n_atoms.memmap'), mode='r', dtype=np.int32)

    def process(self, mol_index):
        if not os.path.isfile(os.path.join(self.grid_dir, "grid" + str(mol_index) + ".memmap")) or self.force_process:
            grid = self._process(mol_index)
            self._persist(grid, mol_index)

    def _process(self, mol_index):
        mol_info = [theano.shared(np.array(self.coords[[mol_index]], dtype=np.float32)),
                    theano.shared(np.array(self.charges[[mol_index]], dtype=np.float32)),
                    theano.shared(np.array(self.vdwradii[[mol_index]], dtype=np.float32)),
                    theano.shared(np.array(self.n_atoms[[mol_index]], dtype=np.int32))]
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
