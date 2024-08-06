import fenics as fe
import fenics_adjoint as fa
import torch_fenics


class StaticElasticity(torch_fenics.FEniCSModule):
    
    def __init__(self, mesh):
        super().__init__()

        # initialize function spaces
        self.scalar_space = fe.FunctionSpace(mesh, 'P', 1)
        self.vector_space = fe.VectorFunctionSpace(mesh, 'P', 1)

    def input_templates(self):
        return (
            fa.Function(self.vector_space), # u_true
            fa.Function(self.scalar_space), # mu
            fa.Function(self.scalar_space), # rho
        )

    def solve(self, u_true, mu, rho):

        # define physical parameters
        g  = 9.8e-3 # gravitational acc (mm/s^2)
        nu = 0.4    # Poisson's ratio (unitless)

        # Lame's first parameter (Pa)
        lam = 2*mu*nu/(1 - 2*nu)

        # set displacement boundary condition
        u_bc = fa.DirichletBC(self.vector_space, u_true, 'on_boundary')

        # body force and traction
        b = fe.as_vector([-rho*g, 0, 0])
        t = fa.Constant([0, 0, 0])

        # define stress and strain
        def epsilon(u):
            return (fe.grad(u) + fe.grad(u).T) / 2

        def sigma(u):
            I = fe.Identity(u.geometric_dimension)
            return lam*fe.div(u)*I + 2*mu*epsilon(u)

        # weak formulation
        u = fe.TrialFunction(self.vector_space)
        v = fe.TestFunction(self.vector_space)

        a = fe.inner(sigma(u), epsilon(v)) * fe.dx
        L = fe.dot(b, v)*fe.dx + fe.dot(t, v)*fe.dx
        
        # solve for displacement
        u_pred = fa.Function(self.vector_space)
        fa.solve(a == L, u_pred, u_bc)

        return u_pred
