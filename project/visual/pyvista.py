import numpy as np
import pyvista as pv


def wrap_image(image, key='image', on_points=False):
    assert image.ndim in {3, 4}, image.shape
    X, Y, Z = image.shape[:3]
    m = pv.ImageData(
        dimensions=(X, Y, Z) if on_points else (X+1, Y+1, Z+1),
        spacing=(1, 1, 1),
        origin=(0, 0, 0)
    )
    mesh_data = m.point_data if on_points else m.cell_data
    if image.ndim == 4:
        mesh_data[key] = image.reshape(-1, image.shape[-1], order='F')
    elif image.ndim == 3:
        mesh_data[key] = image.reshape(-1, order='F')
    return m


def render_image(image, mask=None, **kwargs):
    m = wrap_image(image, key='image')
    if mask is not None:
        m.cell_data['mask'] = mask.reshape(-1, order='F')
        m = m.threshold(scalars='mask', value=1.0 - 1e-12)
    return render_mesh(m, scalars='image', **kwargs)


def show_image(image, mask=None, **kwargs):
    m = wrap_image(image, key='image')
    if mask is not None:
        m.cell_data['mask'] = mask.reshape(-1, order='F')
        m = m.threshold(scalars='mask', value=0.5)
    return show_mesh(m, scalars='image', **kwargs)


def render_mesh(mesh, size=512, view='iso', **kwargs):
    m = pv.wrap(mesh)
    p = pv.Plotter(window_size=(size, size), off_screen=True)
    plot_mesh(m, plotter=p, **kwargs)
    if view == 'iso':
        p.camera_position = 'iso'
        p.camera.azimuth = 180
    elif view == 'xy':
        p.view_xy()
    elif view == 'xz':
        p.view_xz()
    elif view == 'yz':
        p.view_yz()
    p.camera.parallel_projection = True
    try:
        return p.screenshot(return_img=True)
    finally:
        p.close()


def show_mesh(mesh, size=512, **kwargs):
    m = pv.wrap(mesh)
    p = pv.Plotter(window_size=(size, size), off_screen=False)
    plot_mesh(m, plotter=p, **kwargs)
    p.camera_position = 'iso'
    p.camera.azimuth = 180
    try:
        return p.show(jupyter_backend='static')
    finally:
        p.close()


def plot_mesh(
    mesh,
    plotter=None,
    main_kws=None,
    slice_xyz=None,
    slice_kws=None, 
    glyph_factor=None,
    glyph_kws=None,
    **kwargs
):
    m = pv.wrap(mesh)
    p = plotter or pv.Plotter()

    kws = kwargs | (main_kws or {})
    p.add_mesh(m, show_scalar_bar=False, **kws)

    if slice_xyz is True or (slice_xyz is None and slice_kws):
        slice_xyz = (0.5, 0.5, 0.5)

    if slice_xyz:
        s = slice_mesh(mesh, *slice_xyz)
        kws = kwargs | (slice_kws or {})
        plot_mesh(s, plotter=p, **kws)

    if glyph_factor is None and glyph_kws:
        glyph_factor = 1.0

    if glyph_factor and glyph_factor > 0:
        kws = kwargs | (glyph_kws or {})
        vector = kws.pop('scalars', 'u')
        g = m.glyph(scale=vector, orient=vector, factor=glyph_factor)
        plot_mesh(g, plotter=p, **kws)

    return p


def slice_mesh(mesh, x: float, y: float, z: float):
    m = pv.wrap(mesh)
    x_vals, y_vals, z_vals = m.points.T
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    z_min, z_max = z_vals.min(), z_vals.max()
    s = m.slice_orthogonal(
        x=x_min + x * (x_max - x_min),
        y=y_min + y * (y_max - y_min),
        z=z_min + z * (z_max - z_min)
    )
    return s


# DEPRECATED


def _all_same_shape(arrays):
    return len(set(a.shape for a in arrays)) == 1


def as_pyvista_mesh(mesh, scalar=None):
    mesh = pv.wrap(mesh)
    mesh.set_active_scalars(scalar)
    return mesh


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

