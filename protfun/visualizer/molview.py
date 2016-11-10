import numpy as np
import os


class MoleculeView(object):
    """
     Parameters:
        - data : a dictionary with keys "density" and "potential" containing 3-dimensional numpy arrays
         with the molecule's electron density and electron potential distribution.
        - info : a dictionary with keys "id", "name" (and more).
    """

    def __init__(self,
                 data=None,
                 info=None):

        if info is not None:
            self.molecule_name = info["name"]

        if data is not None:
            self.electron_density = data["density"]
            self.electron_potential = data["potential"]

    """
    Create a 3D interactive plot and export images of molecule's electron density.

    Input:
        - plot_params : a dictionary with keys "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"
            containing the boundaries of the plot in units of length.
        - export_figure : boolean to tell whether to export images from the generated figure.
    """

    def density3d(self, plot_params=None, export_figure=True):
        from mayavi import mlab

        density = self.electron_density

        if plot_params is None:
            plot_params = {"mimax_ratio": 0.3}

        grid = mlab.pipeline.scalar_field(density)
        min = density.min()
        max = density.max()

        mlab.pipeline.volume(grid, vmin=min, vmax=min + plot_params["mimax_ratio"] * (max - min))

        mlab.axes()

        if export_figure:
            if not os.path.exists("../data/figures"):
                os.makedirs("../data/figures")
            mlab.savefig(filename='../data/figures/{0}_elden3d.png'.format(self.molecule_name))

        mlab.show()

    """
    Create a 2D interactive plot and export images of molecule's electron density.

    Input:
        - export_figure : boolean to tell whether to export images from the generated figure.
    """

    def density2d(self, plot_params=None, export_figure=True):

        if plot_params is None:
            # use default:
            plot_params = {"im_shape": (3, 3),
                           "orientation": "z"}

        density = self.electron_density

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(*plot_params["im_shape"])

        n_slices = np.prod(plot_params["im_shape"])
        im = None
        min = density.min()
        max = density.max()
        if plot_params["orientation"] == "z":
            for ax, slice, index in zip(axs.flat, density[::(density.shape[2] / n_slices), :, :], xrange(n_slices)):
                im = ax.imshow(slice,
                               interpolation="nearest",
                               vmin=min, vmax=max)
                ax.set_title("z {0}".format(index + 1), fontsize=10)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_aspect('equal')

        elif plot_params["orientation"] == "y":
            for ax, slice, index in zip(axs.flat, density[:, ::(density.shape[1] / n_slices), :], xrange(n_slices)):
                im = ax.imshow(slice,
                               interpolation="nearest",
                               vmin=min, vmax=max)
                ax.set_title("y {0}".format(index + 1), fontsize=10)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_aspect('equal')

        elif plot_params["orientation"] == "x":
            for ax, slice, index in zip(axs.flat, density[:, :, ::(density.shape[0] / n_slices)], xrange(n_slices)):
                im = ax.imshow(slice,
                               interpolation="nearest",
                               vmin=min, vmax=max)
                ax.set_title("x {0}".format(index + 1), fontsize=10)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_aspect('equal')
        else:
            raise ValueError('Orientation can be only "x", "y" or "z".')

        fig.subplots_adjust(wspace=0.1, hspace=0.35, left=0.01, right=0.92)
        fig.suptitle('Electron Density of Molecule {0}'.format(self.molecule_name))

        cax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('Electron Density', rotation=270)

        if export_figure:
            if not os.path.exists("../data/figures"):
                os.makedirs("../data/figures")
            plt.savefig(filename='../data/figures/{0}_elden2d.png'.format(self.molecule_name))

        plt.show()

    """
    Create a 3D interactive plot and export images of molecule's electrostatic potential.

    Input:
        - plot_params : a dictionary with keys "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"
            containing the boundaries of the plot in units of length.
        - export_figure : boolean to tell whether to export images from the generated figure.
    """

    def potential3d(self, export_figure=True):

        from mayavi import mlab

        potential = self.electron_potential

        grid = mlab.pipeline.scalar_field(potential)
        # mlab.pipeline.image_plane_widget(grid,
        #                                  plane_orientation='x_axes',
        #                                  slice_index=10)
        #
        # mlab.pipeline.image_plane_widget(grid,
        #                                  plane_orientation='y_axes',
        #                                  slice_index=10)
        # mlab.outline()

        min = potential.min()
        negative_steps = np.percentile(potential[potential < 0], [2.0, 3.0, 7.5])
        positive_steps = np.percentile(potential[potential > 0], [92.5, 97.0, 98.0])
        max = potential.max()

        vol = mlab.pipeline.volume(grid, vmin=min, vmax=max)
        # mlab.contour3d(potential)

        from tvtk.util.ctf import ColorTransferFunction
        ctf = ColorTransferFunction()
        ctf.add_rgb_point(min, 1, 0.3, 0.3)  # numbers are r,g,b in [0;1]
        ctf.add_rgb_point(negative_steps[1], 1, 0.3, 0.3)
        ctf.add_rgb_point(negative_steps[2], 1, 1, 1)
        ctf.add_rgb_point(positive_steps[0], 1, 1, 1)
        ctf.add_rgb_point(positive_steps[1], 0.3, 0.3, 1)
        ctf.add_rgb_point(max, 0.3, 0.3, 1)
        ctf.range = np.asarray([min, max])
        vol._volume_property.set_color(ctf)
        vol._ctf = ctf
        vol.update_ctf = True

        # Changing the otf:
        from tvtk.util.ctf import PiecewiseFunction
        otf = PiecewiseFunction()
        otf.add_point(min, 1.0)
        otf.add_point(negative_steps[1], 1.0)
        otf.add_point(negative_steps[2], 0.0)
        otf.add_point(positive_steps[0], 0.0)
        otf.add_point(positive_steps[1], 1.0)
        otf.add_point(max, 1.0)
        vol._otf = otf
        vol._volume_property.set_scalar_opacity(otf)

        mlab.axes()

        if export_figure:
            if not os.path.exists("../data/figures"):
                os.makedirs("../data/figures")
            mlab.savefig(filename='../data/figures/{0}_elstpot3d.png'.format(self.molecule_name))

        mlab.show()

    def demo(self):

        # create some dummy molecule data (Gaussian ftw)
        mean = np.zeros(3)
        cov = np.eye(3)

        from scipy.stats import multivariate_normal

        xs, ys, zs = np.mgrid[-3:3:10j, -3:3:10j, -3:3:10j]
        pos = np.empty(xs.shape + (3,))
        pos[:, :, :, 0] = xs
        pos[:, :, :, 1] = ys
        pos[:, :, :, 2] = zs
        sample_density = multivariate_normal.pdf(pos, mean=mean, cov=cov)

        self.electron_density = sample_density
        self.electron_potential = None
        self.molecule_name = "Demo Molecule: Gaussian"

        import imp
        try:
            imp.find_module('mayavi')
            use_mayavi = True
        except ImportError:
            use_mayavi = False

        if use_mayavi:
            self.density3d()
        else:
            self.density2d()
