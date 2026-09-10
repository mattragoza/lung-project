# physics/api.py


def get_solver(config):
    from . import solvers
    class_name = config.get('_class', 'WarpFEMSolver')
    solver_cls = solvers.PDESolver.get_subclass(class_name)
    solver_kws = {k: v for k, v in config.items() if k != '_class'}
    return solver_cls(**solver_kws)


def get_adapter(config):
    from . import adapter
    pde_solver = get_solver(config.get('pde_solver', {}))
    adapter_kws = config.get('physics_adapter', {})
    return adapter.PhysicsAdapter(pde_solver, **adapter_kws)


def get_bc_spec(config):
    from . import bc_spec
    boundary_kws = config.get('boundary_condition', {})
    return bc_spec.BoundaryConditionSpec(**boundary_kws)

