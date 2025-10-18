import torch
import project


def optimize(f, param, maxiter, tol, eps):
    optim = torch.optim.LBFGS(
        params=[param],
        lr=1.0,
        max_iter=10,
        history_size=10,
        line_search_fn='strong_wolfe'
    )
    loss_prev = None
    grad_norm0 = None
    
    def closure():
        optim.zero_grad()
        loss = f(param)
        loss.backward()
        return loss

    for it in range(max_iter):
        loss = optim.step(closure)

        grad_norm = param.grad.norm()
        if grad_norm0 is None:
            grad_norm0 = max(grad_norm, eps)

        rel_grad_norm = grad_norm.item() / grad_norm0

        if loss_prev is None:
            rel_loss_delta = np.nan
        else:
            rel_loss_delta = abs(loss_prev - loss.item()) / abs(loss_prev)

        print(f'iteration {it} | loss = {loss.item()}:.4f')

        if rel_loss_delta < tol or rel_grad_norm < tol:
            print('converged')
            break

        loss_prev = loss.item()


ds = project.datasets.copdgene.COPDGeneDataset(
    data_root='data/COPDGene'
)

examples = ds.examples(
    subjects=['16514P'],
    visit='Phase-1',
    variant='ISO',
    state_pairs=[('EXP', 'INSP')]
)

E0 = 3e3
nu = 0.5

max_iter = 1000
lr = 1e-3

mm_to_m = 1e-3

for ex in project.datasets.base.TorchDataset(examples):
    image  = ex['image']
    mask   = ex['mask']
    disp   = ex['disp']
    mesh   = ex['mesh']
    affine = ex['affine']

    points = project.core.transforms.world_to_voxel_coords(mesh.points, affine)
    points = torch.as_tensor(points, dtype=image.dtype, device=image.device)

    image_vals = project.core.interpolation.interpolate_image(image, points)
    u_obs_vals = project.core.interpolation.interpolate_image(disp, points) * mm_to_m
    rho_vals = project.core.transforms.compute_density_from_ct(image_vals)

    solver = project.solvers.warp.WarpFEMSolver(mesh, relative_loss=True)
    module = project.solvers.base.PDESolverModule(solver, rho_vals, u_obs_vals)

    theta_global = torch.zeros(1, dtype=image.dtype, device=image.device, requires_grad=True)
    theta_local = torch.zeros_like(image_vals, dtype=image.dtype, device=image.device, requires_grad=True)

    def compute_E(theta_global, theta_local):
        log_E = theta_global + theta_local - theta_local.mean()
        return torch.pow(10, log_E)

    def f_global(theta_global):
        E = compute_E(theta_global, theta_local)
        mu, lam = project.core.transforms.compute_lame_parameters(E, nu=0.4)
        return module.forward(mu, lam)

    def f_local(theta_local):
        E = compute_E(theta_global, theta_local)
        mu, lam = project.core.transforms.compute_lame_parameters(E, nu=0.4)
        return module.forward(mu, lam)

    optimize(f_global, theta_global)
    optimize(f_local, theta_local)

