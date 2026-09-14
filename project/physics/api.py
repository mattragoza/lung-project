# physics/api.py


def get_subclass(type: str = 'warp'):
    from .solvers import PDESolver
    return PDESolver.get_subclass(type)


def get_solver(type: str = 'warp', **kwargs):
    solver_cls = get_subclass(type)
    return solver_cls(**kwargs)


def get_adapter(solver, **kwargs):
    from . import adapter
    return adapter.PhysicsAdapter(solver, **kwargs)


def get_bc_spec(**kwargs):
    from . import bc_spec
    return bc_spec.BoundaryConditionSpec(**kwargs)

