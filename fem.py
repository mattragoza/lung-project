import fenics as fe
import matplotlib.pyplot as plt


if __name__ == '__main__':
	mesh_size = 32
	mesh = fe.UnitSquareMesh(mesh_size, mesh_size)
	func_space = fe.FunctionSpace(mesh, 'Lagrange', 1)

	# initial and boundary condition
	ic_expression = fe.Expression(
		cpp_code='exp(-(pow(x[0] - 0.5,2) + pow(x[1] - 0.5, 2))*4)',
		degree=1
	)
	u_prev = fe.interpolate(ic_expression, func_space)

	def boundary(x, on_boundary):
		return on_boundary

	bc = fe.DirichletBC(func_space, fe.Constant(0.0), boundary)

	u_trial = fe.TrialFunction(func_space)
	v_test  = fe.TestFunction(func_space)

	dt = 0.01
	alpha = 1.0
	weak_residual = (
		u_trial * v_test * fe.dx +
		-u_prev * v_test * fe.dx +
		dt * alpha * fe.dot(fe.grad(u_trial), fe.grad(v_test)) * fe.dx
	)
	weak_lhs = fe.lhs(weak_residual)
	weak_rhs = fe.rhs(weak_residual)

	u_solution = fe.Function(func_space)

	n_t = 10
	n_axes = n_t + 1
	fig, ax = plt.subplots(1, n_axes, figsize=(4*n_axes, 4))
	for i in range(n_t + 1):
		plt.sca(ax[i])
		fe.plot(u_prev, label=f't = {i*dt:.4f}', vmin=0, vmax=1, cmap='magma')
		if i < n_t:
			fe.solve(weak_lhs == weak_rhs, u_solution, bc)
			u_prev.assign(u_solution)

	plt.savefig('fem.png')
