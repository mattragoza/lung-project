from __future__ import annotations
import numpy as np
import torch

from .core import fileio, utils, transforms, metrics


def optimize_elasticity_field(
    mesh_path,
    output_path,
    nu_value=0.4,
    unit_m=1e-3, # meters per world unit
    rho_known=True,
    rho_bias=1000.,
    scalar_degree=1,
    vector_degree=1,
    solver_kws=None,
    theta_init=3.0,
    global_kws=None,
    local_kws=None,
    device='cuda'
):
    from .preprocessing import simulation

    mesh = fileio.load_meshio(mesh_path)
    utils.log(mesh)

    verts = mesh.points # world coords
    cells = mesh.cells_dict['tetra']

    if rho_known:
        if scalar_degree == 0:
            rho_values = mesh.cell_data_dict['rho']['tetra']
        elif scalar_degree == 1:
            rho_values = mesh.point_data['rho']
    else:
        img_cells  = mesh.cell_data_dict['image']['tetra']
        img_nodes  = mesh.point_data['image']
        img_values = transforms.smooth_mesh_values(verts, cells, img_nodes, img_cells, scalar_degree)
        rho_values = img_values * rho_bias

    if vector_degree == 0:
        u_obs_values = mesh.cell_data_dict['u'] # world units

    elif vector_degree == 1:
        u_obs_values = mesh.point_data['u']

    utils.log('Optimizing elasticity using observed displacement')
    out_values = simulation.optimize_elasticity(
        verts=verts,
        cells=cells,
        rho_values=rho_values,
        u_obs_values=u_obs_values,
        nu_value=nu_value,
        unit_m=unit_m,
        scalar_degree=scalar_degree,
        vector_degree=vector_degree,
        solver_kws=solver_kws,
        theta_init=theta_init,
        global_kws=global_kws,
        local_kws=local_kws,
        device=device
    )
    utils.log('Assigning optimization fields to mesh')
    for k, v in out_values.items():
        utils.log((k, v.shape, v.dtype, v.mean()))
        if v.shape[0] == len(verts):
            mesh.point_data[k] = v.astype(np.float32)
        elif v.shape[0] == len(cells):
            mesh.cell_data[k] = [v.astype(np.float32)]
        else:
            raise ValueError(f'Invalid mesh field shape: {v.shape}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_meshio(output_path, mesh)

    utils.log('Evaluating optimization metrics')
    if vector_degree == 0:
        u_t = mesh.cell_data['u'][0]
        u_p = mesh.cell_data['u_opt'][0]
        res = mesh.cell_data['res_opt'][0]

    elif vector_degree == 1:
        u_t = mesh.point_data['u']
        u_p = mesh.point_data['u_opt']
        res = mesh.point_data['res_opt']

    if scalar_degree == 0:
        E_t = mesh.cell_data['E'][0]
        E_p = mesh.cell_data['E_opt'][0]

    elif scalar_degree == 1:
        E_t = mesh.point_data['E']
        E_p = mesh.point_data['E_opt']

    ret_metrics = (
        utils.namespace(metrics.evaluate_metrics(res, which='res'), 'res')  |
        utils.namespace(metrics.evaluate_metrics(u_p, u_t, which='u'), 'u') |
        utils.namespace(metrics.evaluate_metrics(E_p, E_t, which='E'), 'E')
    )
    utils.log(ret_metrics)

    return ret_metrics

