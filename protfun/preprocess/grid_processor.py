from protfun.layers import MoleculeMapLayer
import lasagne
import theano
import numpy as np
import os
import colorlog as log
import logging

log.basicConfig(level=logging.DEBUG)


class GridProcessor(object):
    def __init__(self, folder_name='train_grids'):
        self.grid_dir = os.path.join(os.path.dirname(__file__), '../../data', folder_name)
        if not os.path.exists(self.grid_dir):
            os.makedirs(self.grid_dir)
        dummy = lasagne.layers.InputLayer(shape=(None,))
        self.processor = MoleculeMapLayer(incoming=dummy, minibatch_size=1)

    def process(self, mol_index):
        grid = self._process(mol_index)
        self._persist(grid, mol_index)

    def _process(self, mol_index):
        mol_index = theano.shared(np.array([mol_index], dtype=np.int32))
        grid = self.processor.get_output_for(molecule_ids=mol_index).eval()
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
