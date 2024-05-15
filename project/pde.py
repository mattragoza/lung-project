import fenics as fe
import fenics_adjoint as fa
import torch_fenics

from fenics import grad, inner, dx, ds


class PDESolver(torch_fenics.FEniCSModule):
    
    def __init__(self, n_nodes):
        super().__init__()
        
        # create mesh and function space
        n_elements = n_nodes - 1
        self.mesh = fe.UnitIntervalMesh(n_elements)
        self.V = fe.FunctionSpace(self.mesh, 'P', 1)
        
        # create trial and test functions
        self.u = fe.TrialFunction(self.V)
        self.v = fe.TestFunction(self.V)
        
        # construct bilinear form
        self.a = inner(grad(self.u), grad(self.v)) * dx

    def solve(self, f, ub):
        
        # construct linear form
        L = f * self.v * dx
        
        # construct boundary condition
        bc = fa.DirichletBC(self.V, ub, 'on_boundary')
        
        # solve the Poisson equation
        u = fa.Function(self.V)
        fa.solve(self.a == L, u, bc)
        
        return u
    
    def input_templates(self):
        return fa.Function(self.V), fa.Constant(0)
