import sys, os, argparse
import numpy as np
import xarray as xr
import fenics as fe


class AbsorbingBoundary(fe.UserExpression):

    def eval(self, values, x):
        if x[0] < 0.05 or x[0] > 0.95 or x[1] > 0.95:
            values[0] = 0 #-1000
        else:
            values[0] = 0 #-10


class WaveSimulation(object):

    def __init__(
        self,
        n_nodes=81,
        omega=50, # Hz
        rho=1000, # kg / m^3
        mu=4000, # Pa
        L=0.2, # meters
        dt=1e-4, # seconds
        total_time=0.04 # seconds
    ):
        self.initialize_mesh(n_nodes)
        self.initialize_icbc(omega)
        self.initialize_material(rho, mu, L)
        self.check_cfl_condition(dt)
        self.initialize_weak_form(dt)
        self.solve(dt, total_time)

    def initialize_mesh(self, n_nodes):
        n_elements = n_nodes - 1
        self.mesh_spacing = 1 / n_elements
        self.mesh = fe.UnitSquareMesh(n_elements, n_elements)
        self.V = fe.FunctionSpace(mesh, 'Lagrange', 1)

    def initialize_icbc(self, omega):

        # initial condition
        self.ic_expression = fe.Constant(0)
        self.u0 = fe.interpolate(ic_expression, self.V)
        self.u1 = fe.interpolate(ic_expression, self.V)

        def boundary(x, on_boundary):
            return on_boundary and fe.near(x[1], 0, 1e-14)

        # Dirichlet boundary condition
        self.bc_expression = fe.Constant(
            'sin(6.2831 * omega * t)', degree=2, t=0, omega=omega
        )
        self.bc = fe.DirichletBC(self.V, self.bc_expression, boundary)

        # source term and Neumann boundary condition
        self.f = fe.Expression('0', degree=0)
        self.g = fe.Expression('0', degree=1)

    def initialize_material(self, mu, rho, L):
        self.c = np.sqrt(mu, rho) / L

    def check_cfl_condition(self, dt):
        cfl_number = self.c * dt / self.mesh_spacing
        assert cfl_number <= 1, 'CFL condition violated'

    def initialize_weak_form(self, dt):
        from fenics import grad, dot, dx, ds

        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        k = fe.interpolate(AbsorbingBoundary(), self.V)

        a = (
            (1 - k * dt) * (u * v) * dx + 
            (self.c * dt)**2 * dot(grad(u), grad(v)) * dx
        )
        L = (
            ((2 - k * dt) * self.u1 - self.u0 + self.f * dt**2) * v * dx +
            (self.g * v) * ds
        )
        return a, L

    def solve(self, a, L, dt, total_time):
        u = fe.Function(self.V)
        t = 0
        while t < total_time:
            t += dt
            self.bc_expression.t = t
            fe.solve(a == L, u, [self.bc])
            self.u0.assign(self.u1)
            self.u1.assign(u)

            if i % 10 == 0:
                print(f'Iteration {i+1} / {n_steps}')



def parse_args(argv):
    parser = argparse.ArgumentParser()
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    simulation = WaveSimulation()
    simulation.solve()


if __name__ == '__main__':
    #main(sys.argv[1:])
    pass


