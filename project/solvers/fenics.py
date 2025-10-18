import sys
import numpy as np
import torch
try:
    import fenics as fe
    import fenics_adjoint as fa
except Exception as e:
    print(f'Failed to import fenics: {e}', file=sys.stderr)


class FenicsFEMSolver:

    def __init__(self, geometry):
        self.geometry = geometry

        # function spaces
        self.S = fe.FunctionSpace(geometry, 'CG', 1)
        self.V = fe.VectorFunctionSpace(geometry, 'CG', 1)

        # trial and test functions
        self.u_trial = fa.TrialFunction(self.V)
        self.v_test  = fa.TestFunction(self.V)

        # physical fields and constants
        self.u_obs_func = fa.Function(self.V)
        self.u_sim_func = fa.Function(self.V)

        self.mu_func  = fa.Function(self.S)
        self.lam_func = fa.Function(self.S)
        self.rho_func = fa.Function(self.S)

        self.g = fe.as_vector([0, 0, -9.81])
    
    def solve(self):

        a = pde_bilinear_form(
            u=self.u_trial,
            v=self.v_test,
            mu=self.mu_func,
            lam=self.lam_func
        )
        L = linear_form(
            v=self.v_test,
            rho=self.rho_func,
            g=self.g
        )
        dbc = fa.DirichletBC(self.V, self.u_obs_func, 'on_boundary')

        fa.solve(a == L, self.u_sim_func, [dbc])


def pde_bilinear_form(
    u: fa.Function,
    v: fa.Function,
    mu: fa.Function,
    lam: fa.Function
):
    I = fe.Identity(3)
    eps_u = fe.sym(fe.grad(u))
    eps_v = fe.sym(fe.grad(v))
    div_u = fe.div(u)
    sigma = 2*mu*eps_u + lam*div_u*I
    return fe.inner(sigma, eps_v) * fe.dx


def pde_linear_form(
    v: fa.Function,
    rho: fa.Function,
    g: fe.Vector
):
    return rho * fe.inner(g, v) * fe.dx

