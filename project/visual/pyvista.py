import numpy as np
import pyvista as pv


def _all_same_shape(arrays):
    return len(set(a.shape for a in arrays)) == 1


def as_pyvista_grid(affine=None, **kwargs):
    if affine is None:
        affine = np.eye(4, dtype=float)

    assert affine.shape == (4, 4)
    assert len(kwargs) > 0
    assert _all_same_shape(kwargs.values())

    shape = next(v.shape for v in kwargs.values())
    spacing = np.linalg.norm(affine[:3,:3], axis=1)
    direction = affine[:3,:3] @ np.diag(1 / spacing)

    grid = pv.ImageData(
        dimensions=shape,
        spacing=spacing,
        origin=affine[:3,3],
        #direction=direction,
    )
    for k, v in kwargs.items():
        grid.point_data[k] = v.flatten(order='F')

    return grid


def as_pyvista_mesh(mesh, scalar=None):
    mesh = pv.wrap(mesh)
    mesh.set_active_scalars(scalar)
    return mesh


def plot_mesh(mesh, scalar=None, plotter=None, **plot_kws):
    plotter = plotter or pv.Plotter()
    mesh = as_pyvista_mesh(mesh, scalar)
    plotter.add_mesh(mesh, scalars=scalar, **plot_kws)
    plotter.enable_depth_peeling(10)
    return plotter


def plot_volume(array, affine=None, scalar='value', plotter=None, **plot_kws):
    plotter = plotter or pv.Plotter()
    grid = as_pyvista_grid(affine, **{scalar: array})
    plotter.add_volume(grid, scalars=scalar, **plot_kws)
    plotter.enable_depth_peeling(10)
    return plotter


def plot_contour(array, affine=None, level=0, scalar='value', plotter=None, **plot_kws):
    plotter = plotter or pv.Plotter()
    grid = as_pyvista_grid(affine, **{scalar: array})
    plotter.add_mesh(grid.contour(level), scalars=scalar, **plot_kws)
    plotter.enable_depth_peeling(10)
    return plotter

