# import matplotlib
# matplotlib.use("Agg")	# needed for the ssh connection
# import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
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

            self.molecule_name = info["name"]

            self.electron_density = data["density"]
            self.electron_potential = data["potential"]

        """
        Create a 3D interactive plot and export images of molecule's electron density and potential.

        Input:
            - plot_params : a dictionary with keys "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"
                containing the boundaries of the plot in units of length.
            - export_figure : boolean to tell whether to export images from the generated figure.
        """
        def density3d(self, plot_params=None, export_figure=True):

            density = self.electron_density

            if plot_params is None:
                # use default:
                nx = complex(0, density.shape[0])
                ny = complex(0, density.shape[1])
                nz = complex(0, density.shape[2])
                plot_params = {"xmin":-1, "xmax":1,
                               "ymin":-1, "ymax":1,
                               "zmin":-1, "zmax":1,
                               "nx":nx, "ny":ny, "nz":nz,
                               "mimax_ratio":0.5}

            from mayavi import mlab

            figure = mlab.figure('Electron Density of Molecule {0}'.format(self.molecule_name))

            xmin, xmax, ymin, ymax, zmin, zmax = (plot_params[key] for key in ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])
            nx, ny, nz = (plot_params[key] for key in ["nx", "ny", "nz"])
            xs, ys, zs = np.mgrid[xmin:xmax:nx, ymin:ymax:ny, zmin:zmax:nz]

            grid = mlab.pipeline.scalar_field(xs, ys, zs, density)
            min = density.min()
            max = density.max()

            mlab.pipeline.volume(grid, vmin=min, vmax=min + plot_params["mimax_ratio"] * (max - min))

            mlab.axes()

            if export_figure:
                if not os.path.exists("../data/figures"):
                    os.makedirs("../data/figures")
                mlab.savefig(filename='../data/figures/{0}_elden.png'.format(self.molecule_name))

            mlab.show()

        """
            Create a 2D interactive plot and export images of molecule's electron density and potential.

            Input:
                - export_figure : boolean to tell whether to export images from the generated figure.
            """
        def density2d(self, export_figure=True):

            # TODO: break the data array in n slices
            # TODO: for each slice plot the density using imshow
            # TODO: maybe also use interpolation.
            # TODO: Plot a colorbar too
            pass

if __name__ == "__main__":

    # create some dummy molecule data (Gaussian ftw)
    mean = np.zeros(3)
    cov = np.eye(3)

    from scipy.stats import multivariate_normal

    xs, ys, zs = np.mgrid[-3:3:50j, -3:3:50j, -3:3:50j]
    pos = np.empty(xs.shape + (3,))
    pos[:, :, :, 0] = xs; pos[:, :, :, 1] = ys; pos[:, :, :, 2] = zs
    sample_density = multivariate_normal.pdf(pos, mean=mean, cov=cov)

    mol_data = {"density": sample_density, "potential": None}
    mol_info = {"name": "Unknown", "id": 0}

    mv = MoleculeView(data=mol_data, info=mol_info)
    mv.density3d()


