# physics/warp/forms.py

import warp as wp
import warp.fem


# ----- FEM integrand factories -----


def build_residual_form(stress_func):
    stress_name = stress_func.func.__name__

    def residual_form(
        x: wp.fem.Sample,
        u: wp.fem.Field,
        v: wp.fem.Field,
        mu: wp.fem.Field,
        lam: wp.fem.Field,
        rho: wp.fem.Field,
        g: wp.vec3
    ):
        I = wp.identity(3, float)
        F = I + wp.fem.grad(u, x)

        P = stress_func(F, mu(x), lam(x))

        internal = wp.ddot(P, wp.fem.grad(v, x))
        external = rho(x) * wp.dot(g, v(x))

        return external - internal

    residual_form.__qualname__ = f'{stress_name}_residual_form'

    return wp.fem.integrand(residual_form)


def build_jacobian_form(stress_func):
    stress_name = stress_func.func.__name__

    # scalar function whose gradient wrt F gives the tangent action
    # d/dF [P(F):H] = (dP/dF)^T : H = (dP/dF) : H (due to symmetry)

    def stress_dot(F: wp.mat33, H: wp.mat33, mu: float, lam: float):
        return wp.ddot(stress_func(F, mu, lam), H)

    stress_dot.__qualname__ = f'{stress_name}_dot'

    stress_dot = wp.func(stress_dot)
    stress_dot_grad = wp.grad(stress_dot)

    def jacobian_form(
        x: wp.fem.Sample,
        u: wp.fem.Field,
        v: wp.fem.Field,
        du: wp.fem.Field,
        mu: wp.fem.Field,
        lam: wp.fem.Field
    ):
        I = wp.identity(3, float)
        F = I + wp.fem.grad(u, x)
        H = wp.fem.grad(du, x)

        dP = stress_dot_grad(F, H, mu(x), lam(x))[0]

        return wp.ddot(dP, wp.fem.grad(v, x))

    jacobian_form.__qualname__ = f'{stress_name}_jacobian_form'

    return wp.fem.integrand(jacobian_form)


# ----- linear elasticity model -----


@wp.func
def linear_elastic_stress(F: wp.mat33, mu: float, lam: float):
    I = wp.identity(n=3, dtype=float)

    # infinitesimal strain (epsilon)
    eps = 0.5 * (F + wp.transpose(F)) - I

    # isotropic Cauchy stress (sigma)
    return 2.0 * mu * eps + lam * wp.trace(eps) * I


# ----- St. Venant-Kirchhoff model -----


@wp.func
def st_venant_kirchhoff_stress(F: wp.mat33, mu: float, lam: float):
    I = wp.identity(n=3, dtype=float)

    # Green-Lagrange strain
    E = 0.5 * (wp.transpose(F) * F - I)

    # second Piola-Kirchoff stress
    S = 2.0 * mu * E + lam * wp.trace(E) * I

    # S = mu (C - I) + lam/2 tr(C - I) I

    # first Piola-Kirchoff stress (P)
    return F * S


# ----- Neo-Hookean model(s) -----


@wp.func
def ciarlet_neohookean_stress(F: wp.mat33, mu: float, lam: float):

    # strain energy density:
    # W = mu/2 (I₁ - 3 - 2 ln J) + lam/4 (J² - 1 - 2 ln J)
    # P = dW/dF

    # kinematic differentials:
    # d/dF [I₁]   = d/dF [F:F]   = 2F
    # d/dF [J]    = d/dF [det F] = JF⁻ᵀ
    # d/dF [ln J] = d/dF [J] / J = F⁻ᵀ

    # first Piola-Kirchoff stress:
    # P = mu (F - F⁻ᵀ) + lam/2 (J² - 1) F⁻ᵀ
    # P = FS

    # second Piola-Kirchoff stress:
    # S = mu (I - C⁻¹) + lam/2 (J² - 1) C⁻¹

    J = wp.determinant(F)
    F_inv_T = wp.transpose(wp.inverse(F))

    return mu * (F - F_inv_T) + 0.5 * lam * (J*J - 1.) * F_inv_T


@wp.func
def decoupled_neohookean_stress(F: wp.mat33, mu: float, lam: float):

    # strain energy density:
    # W = W_iso + W_vol
    # W_iso = C₁ (I₁_bar - 3)
    # W_vol = D₁ (J - 1)²

    # isochoric first invariant:
    # I₁_bar = J⁻²ᐟ³ I₁

    # material parameters:
    # C₁ = G/2 = mu/2
    # D₁ = K/2 = [lam + (2/3) mu] / 2

    # first Piola-Kirchhoff stress:
    # P = P_iso + P_vol
    # P_iso = G J⁻²ᐟ³ (F - (1/3) I₁ F⁻ᵀ)
    # P_vol = K J (J - 1) F⁻ᵀ

    J = wp.determinant(F)
    F_inv_T = wp.transpose(wp.inverse(F))

    I1 = wp.ddot(F, F)
    J_neg_23 = wp.pow(J, -2/3)

    G, K = mu, lam + (2/3) * mu

    H = F - (I1 / 3) * F_inv_T
    P_iso = G * J_neg_23 * H
    P_vol = K * J * (J - 1.) * F_inv_T

    return P_iso + P_vol


# ----- Yeoh hyperelastic model -----


@wp.func
def yeoh_hyperelastic_stress(
    F: wp.mat33,
    mu: float,
    lam: float,
    a2: float,
    a3: float
):
    # strain energy density:
    # W = W_iso + W_vol
    # W_iso = C₁ (I₁_bar - 3) + C₂ (I₁_bar - 3)² + C₃ (I₁_bar - 3)³
    # W_vol = D₁ (J - 1)²

    J = wp.determinant(F)
    F_inv_T = wp.transpose(wp.inverse(F))

    I1 = wp.ddot(F, F)
    J_neg_23 = wp.pow(J, -2/3)
    I1_bar = J_neg_23 * I1

    G, K = mu, lam + (2/3) * mu

    C1 = G / 2.0
    C2 = a2 * C1
    C3 = a3 * C1
    D1 = K / 2.0

    q = (I1_bar - 3.)
    phi = C1 + 2.0 * C2 * q + 3.0 * C3 * q**2
    G_term = 2.0 * phi

    H = F - (I1 / 3) * F_inv_T
    P_iso = G_term * J_neg_23 * H
    P_vol = K * J * (J - 1.) * F_inv_T

    return P_iso + P_vol


# ----- other FEM integrands -----


@wp.fem.integrand
def inner_product_form(
    x: wp.fem.Sample,
    u: wp.fem.Field, # vector
    v: wp.fem.Field  # vector
):
    return wp.dot(u(x), v(x))


@wp.fem.integrand
def squared_error_form(
    x: wp.fem.Sample,
    u: wp.fem.Field, # vector
    v: wp.fem.Field, # vector
    w: wp.fem.Field  # scalar
):
    r_x = u(x) - v(x)
    return w(x) * wp.dot(r_x, r_x)


@wp.fem.integrand
def squared_norm_form(
    x: wp.fem.Sample,
    u: wp.fem.Field, # vector
    w: wp.fem.Field  # scalar
):
    u_x = u(x)
    return w(x) * wp.dot(u_x, u_x)


@wp.fem.integrand
def volume_form(x: wp.fem.Sample, w: wp.fem.Field):
    return w(x)


@wp.fem.integrand
def TV_reg_form(
    x: wp.fem.Sample,
    f: wp.fem.Field,
    eps_reg: float,
    eps_div: float
):
    '''
    Smooth penalty on L2 norm of log-parameter gradient.
    '''
    grad_log_f = wp.fem.grad(f, x) / (f(x) + eps_div)

    return wp.sqrt(
        wp.dot(grad_log_f, grad_log_f) + eps_reg * eps_reg
    )


@wp.fem.integrand
def det_F_form(x: wp.fem.Sample, u: wp.fem.Field):
    I = wp.identity(3, float)
    F = I + wp.fem.grad(u, x)
    return wp.determinant(F)

