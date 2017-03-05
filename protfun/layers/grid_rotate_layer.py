import numpy as np
import theano
import lasagne
import logging
import colorlog as log

from theano import tensor as T

log.basicConfig(level=logging.DEBUG)
floatX = theano.config.floatX


class GridRotationLayer(lasagne.layers.Layer):
    """
    GridRotationLayer is a dynamic 3D augmentation layer that can be used in the beginning of
    any neural network. It performs random rotations and (small) translations in 3D space on the
    fly.

    Usage::
        >>> from lasagne.layers import InputLayer
        >>> minibatch_size = 8
        >>> n_channels = 2
        >>> side = 32
        >>> input_layer = InputLayer(shape=(minibatch_size, n_channels, side, side, side),
        >>>                          input_var=input_grid)
        >>> # apply the rotation layer
        >>> rotation_layer = GridRotationLayer(incoming=input_layer, grid_side=side,
        >>>                                    n_channels=n_channels, interpolation='linear')
    """
    min_dist_from_border = 5

    def __init__(self, incoming, grid_side, n_channels, interpolation='linear',
                 avg_rotation_angle=np.pi, **kwargs):
        """
        :param incoming: the incoming lasagne layer (usually an InputLayer)
            expected shape is (minibatch_size, n_channels, grid_side, grid_side, grid_side)
        :param grid_side: number of points on each side of the 3D input
        :param n_channels: number of channels in the 3D input
        :param interpolation: 'linear' or 'nearest'
        :param avg_rotation_angle: default is np.pi
        :param kwargs: lasagne **kwargs
        """
        super(GridRotationLayer, self).__init__(incoming, **kwargs)
        self.grid_side = grid_side
        self.n_channels = n_channels
        self.interpolation = interpolation
        self.angle = avg_rotation_angle

    def get_output_shape_for(self, input_shape):
        return None, self.n_channels, self.grid_side, self.grid_side, self.grid_side

    def get_output_for(self, grids, **kwargs):
        height = width = depth = self.grid_side

        # np.indices() returns 3 train_grids exactly as big as the original one.
        # The first grid contains the X coordinate of each point at the location of the point.
        # The second grid contains the Y coordinate of each point at the location of the point.
        # The third grid contains the Z coordinate of each point at the location of the point.
        indices_grids = T.as_tensor_variable(np.indices((width, height, depth), dtype=floatX),
                                             name="grid_indices")

        # Translate:
        # the translation vector will be broad-casted:
        # t_x will be added to all values in the first indices grid
        # t_y will be added to all values in the second indices grid
        # t_z will be added to all values in the third indices grid
        # resulting in a translation in the direction of translation_vector
        indices_grids = T.add(indices_grids, self._translation_vector())

        # Rotate:
        # the origin is just the center point in the grid
        origin = T.as_tensor_variable(np.array((width // 2, height // 2, depth // 2),
                                               dtype=floatX).reshape((3, 1, 1, 1)), name='origin')
        # We first center all indices, just as in the translation above
        indices_grids = T.sub(indices_grids, origin)

        # T.tensordot is a generalized version of a dot product.
        # The axes parameter is of length 2, and it gives the axis for each of the two tensors
        # passed, over which the summation will occur. Of course, those two axis need to be of the
        # same dimension.
        # Here we have a (3 x 3) matrix <dot product> (3, width, height, depth) grid, and the
        # summation happens over the first axis (index 0). The result is of size
        # (3 x width x height x depth) and contains again 3 train_grids of this time
        # **rotated** indices for each dimension X, Y, Z respectively.
        indices_grids = T.tensordot(self._rotation_matrix(), indices_grids,
                                    axes=(0, 0))

        # Decenter
        indices_grids = T.add(indices_grids, origin)

        # Since indices_grids was transformed, we now might have indices at certain locations
        # that are out of the range of the original grid. We this need to clip them to valid values.
        # For the first grid: between 0 and width - 1
        # For the second grid: between 0 and height - 1
        # For the third grid: between 0 and depth - 1
        # Note that now te index train_grids might contain real numbers (not only integers).
        x_indices = T.clip(indices_grids[0], 0, width - 1 - .001)
        y_indices = T.clip(indices_grids[1], 0, height - 1 - .001)
        z_indices = T.clip(indices_grids[2], 0, depth - 1 - .001)

        if self.interpolation == "nearest":
            # Here we just need to round the indices for each spatial dimension to the closest
            # integer, and than index the original input grid with the 3 indices train_grids
            # (numpy style indexing with arrays) to obtain the final result. Note that here,
            # as usual, the multi-dim array that you index with has the
            # same spatial dimensionality as the multi-dim array being index.

            # We intentionally flatten everything before indexing, so that Theano can use
            # ArraySubtensor1 instead of ArraySubtensor, because only the former can be run
            # on the GPU.
            # https://groups.google.com/forum/#!topic/theano-users/XkPJP6on50Y
            flat_grids, flat_indices = grids.reshape(
                (grids.shape[0], grids.shape[1], -1)), \
                                       width * height * T.iround(
                                           x_indices).flatten() + \
                                       height * T.iround(y_indices).flatten() + \
                                       T.iround(z_indices).flatten()
            output = flat_grids[:, :, flat_indices]
            output = output.reshape(grids.shape)
        else:
            flat_grids = grids.reshape((grids.shape[0], grids.shape[1], -1))
            # For linear interpolation, we use the transformed indices x_indices, y_indices and
            # z_indices to linearly calculate the desired values at each of the original indices
            # in each dimension.

            # Again, everything is flattened so that Theano can put it on the GPU, just as in
            # the other part of this if block.
            # https://groups.google.com/forum/#!topic/theano-users/XkPJP6on50Y
            top = T.cast(y_indices, 'int32').flatten()
            left = T.cast(x_indices, 'int32').flatten()
            forward = T.cast(z_indices, 'int32').flatten()

            x_indices = x_indices.flatten()
            y_indices = y_indices.flatten()
            z_indices = z_indices.flatten()

            # this computes the amount of shift into each direction from the original position
            fraction_y = T.cast(y_indices - top, theano.config.floatX).flatten()
            fraction_x = T.cast(x_indices - left,
                                theano.config.floatX).flatten()
            fraction_z = T.cast(z_indices - forward,
                                theano.config.floatX).flatten()

            # then the new value is the linear combination based on the shifts in all
            # of the 8 possible directions in 3D
            output = flat_grids[:, :, self.grid_side ** 2 * top + self.grid_side * left + forward] \
                     * (1 - fraction_y) * (1 - fraction_x) * (1 - fraction_z) + \
                     flat_grids[:, :,
                     self.grid_side ** 2 * top + self.grid_side * left + (forward + 1)] \
                     * (1 - fraction_y) * (1 - fraction_x) * fraction_z + \
                     flat_grids[:, :,
                     self.grid_side ** 2 * top + self.grid_side * (left + 1) + forward] \
                     * (1 - fraction_y) * fraction_x * (1 - fraction_z) + \
                     flat_grids[:, :,
                     self.grid_side ** 2 * top + self.grid_side * (left + 1) + (forward + 1)] \
                     * (1 - fraction_y) * fraction_x * fraction_z + \
                     flat_grids[:, :,
                     self.grid_side ** 2 * (top + 1) + self.grid_side * left + forward] \
                     * fraction_y * (1 - fraction_x) * (1 - fraction_z) + \
                     flat_grids[:, :,
                     self.grid_side ** 2 * (top + 1) + self.grid_side * left + (forward + 1)] \
                     * fraction_y * (1 - fraction_x) * fraction_z + \
                     flat_grids[:, :,
                     self.grid_side ** 2 * (top + 1) + self.grid_side * (left + 1) + forward] \
                     * fraction_y * fraction_x * (1 - fraction_z) + \
                     flat_grids[:, :,
                     self.grid_side ** 2 * (top + 1) + self.grid_side * (left + 1) + (forward + 1)] \
                     * fraction_y * fraction_x * fraction_z
            output = output.reshape(grids.shape)

        return output

    def _rotation_matrix(self):
        # generate a random rotation matrix Q
        random_streams = T.shared_randomstreams.RandomStreams()

        # random givens rotations in all 3 spatial dimensions
        angle = random_streams.uniform((3,), low=-self.angle, high=self.angle,
                                       ndim=1, dtype=floatX)
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

        return R

    @staticmethod
    def _translation_vector():
        min = T.constant(-2.5, 'min_translation', dtype=floatX)
        max = T.constant(2.5, 'max_translation', dtype=floatX)
        random_streams = T.shared_randomstreams.RandomStreams()

        # unifom random in open interval ]0;1[
        rand01 = random_streams.uniform((3, 1, 1, 1), dtype=floatX)
        rand_translation = T.add(T.mul(rand01 * T.sub(max, min)), min)
        return rand_translation


if __name__ == "__main__":
    """
    A quick test of the grid rotation layer.
    The rotations are visualized using the MoleculeView.
    """
    from protfun.visualizer.molview import MoleculeView

    # set those directories to something meaningful in your environment
    data_dir = "/home/valor/workspace/DLCV_ProtFun/data/full/processed_single_64/1A0H"
    grid_file = "/home/valor/workspace/DLCV_ProtFun/data/full/processed_single_64/1A0H/grid.memmap"

    # visualize the original grid
    test_grid = np.memmap(grid_file, mode='r', dtype=floatX).reshape((1, 2, 64, 64, 64))
    log.debug(test_grid.shape)
    viewer = MoleculeView(data_dir=data_dir, data={"potential": test_grid[0, 1],
                                                   "density": test_grid[0, 1]},
                          info={"name": "test"})
    viewer.density3d()
    grid_side = test_grid.shape[3]

    # initialize the rotation layer
    input_grid = T.TensorType(floatX, (False,) * 5)()
    input_layer = lasagne.layers.InputLayer(shape=(1, 2, grid_side, grid_side, grid_side),
                                            input_var=input_grid)
    rotate_layer = GridRotationLayer(incoming=input_layer, grid_side=grid_side,
                                     n_channels=2, interpolation='nearest')

    # create a small function to test the rotation layer
    func = theano.function(inputs=[input_grid],
                           outputs=lasagne.layers.get_output(rotate_layer))

    # show 10 different rotations of the test grid
    import time
    for i in range(0, 10):
        start = time.time()
        rotated_grid = func(test_grid)
        log.info("took time: {}".format(time.time() - start))
        viewer = MoleculeView(data_dir=data_dir,
                              data={"potential": rotated_grid[0, 1],
                                    "density": rotated_grid[0, 1]},
                              info={"name": "test"})
        viewer.density3d()
