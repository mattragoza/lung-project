import numpy as np
import fenics as fe
import fenics_adjoint as fa
import torch
import torch_fenics
import dolfin


class FiniteElementModel(torch_fenics.FEniCSModule):
    
    def __init__(self, mesh, resolution, cell_labels):
        super().__init__()
        self.mesh = mesh
        self.S = fe.FunctionSpace(mesh, 'P', 1)
        self.V = fe.VectorFunctionSpace(mesh, 'P', 1)

        self.points = torch.as_tensor(self.S.tabulate_dof_coordinates())
        self.radius = compute_point_radius(self.points, resolution)

        self.cell_labels = cell_labels
        self.has_interface = False

        if cell_labels is not None:
            # detect interface between subdomains
            dim = mesh.geometry().dim()
            self.on_interface = fe.MeshFunction('size_t', mesh, dim - 1)
            for facet in dolfin.facets(mesh):
                cell_indices = [c for c in facet.entities(dim)]
                subdomain_ids = set(cell_labels[c] for c in cell_indices)
                if len(subdomain_ids) > 1:
                    self.on_interface[facet] = 1
                    self.has_interface = True
        
    def __repr__(self):
        return f'{type(self).__name__}({self.mesh})'
        
    def input_templates(self):
        scalar_f = fa.Function(self.S)
        vector_f = fa.Function(self.V)
        return vector_f, scalar_f, scalar_f
    
    def solve(self, u_bc, E, rho):

        # define physical parameters
        g  = 9.8e-3 # gravitational acc (mm/s^2)
        nu = 0.4    # Poisson's ratio (unitless)

        # convert to Lame parameters (Pa)
        mu  = E / (2*(1 + nu))
        lam = E * nu / ((1 + nu)*(1 - 2*nu))

        # set displacement boundary condition
        u_bc_surface = fa.DirichletBC(self.V, u_bc, 'on_boundary')
        bcs = [u_bc_surface]

        if self.has_interface:
            u_bc_interface = fa.DirichletBC(self.V, u_bc, self.on_interface, 1)
            bcs.append(u_bc_interface)

        # body force and traction
        b = fe.as_vector([0, rho*g, 0])
        t = fa.Constant([0, 0, 0])

        # define stress and strain
        def epsilon(u):
            return (fe.grad(u) + fe.grad(u).T) / 2

        def sigma(u):
            I = fe.Identity(u.geometric_dimension())
            return lam*fe.div(u)*I + 2*mu*epsilon(u)

        # weak formulation
        u = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)

        a = fe.inner(sigma(u), epsilon(v)) * fe.dx
        L = fe.dot(b, v)*fe.dx + fe.dot(t, v)*fe.dx

        u_pred = fa.Function(self.V)
        fa.solve(a == L, u_pred, bcs)

        return u_pred


def compute_point_radius(points, resolution):
    min_radius = np.linalg.norm(resolution) / 2
    distance = torch.norm(points[:,None,:] - points[None,:,:], dim=-1)
    distance[distance == 0] = 1e3
    distance[distance < min_radius] = min_radius
    return distance.min(dim=-1, keepdims=True).values
