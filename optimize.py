import project

ds = project.datasets.copdgene.COPDGeneDataset(
	data_root='data/COPDGene'
)

examples = ds.examples(
	subjects=['16514P'],
	visits=['Phase-1'],
	variant='ISO',
	state_pairs=[('EXP', 'INSP')],
	recon='STD',
	mask_name='lung_regions',
	mesh_tag='volume'
)

E0 = 3e3
nu = 0.5

max_iter = 1000
lr = 1e-3

for ex in project.datasets.base.TorchDataset(examples):
	image  = ex['image']
	mask   = ex['mask']
	disp   = ex['disp']
	mesh   = ex['mesh']
	affine = ex['affine']

	points = project.core.transforms.world_to_voxel_coords(mesh.points, affine)
	points = torch.as_tensor(points, dtype=image.dtype, device=image.device)

	image_vals = project.core.interpolation.interpolate_image(image, points)
	u_obs_vals = project.core.interpolation.interpolate_image(disp, points)
	rho_vals = project.core.transforms.compute_density_from_ct(image_vals)

	solver = project.solvers.warp.WarpSolver(mesh, relative=True)
	solver.set_fixed(rho_vals, u_obs_vals)
	solver.assemble_f()

	module = project.solvers.base.PDESolverModule(solver)

	theta = torch.ones_like(image_vals, dtype=torch.float32, requires_grad=True)
	optim = torch.optim.Adam([theta], lr=lr)

	for i in range(max_iter):
		optim.zero_grad()
		E_vals = E0 * torch.exp(theta)
		mu_vals, lam_vals = project.core.transforms.compute_lame_parameters(E_vals, nu)
		loss = module.forward(mu_vals, lam_vals)
		grad_norm = theta.grad.norm()
		loss.backward()
		print(f'iteration {i} | loss={loss.item():.8e} | grad_norm={grad_norm.item():.8e}')
		optim.step()
