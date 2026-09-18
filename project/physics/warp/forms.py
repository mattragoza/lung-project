# physics/warp/forms.py

import warp as wp
import warp.fem


# ----- FEM integrand factories -----


def build_residual_form(stress_func):

    def residual_form(
        x: wp.fem.Sample,
        u: wp.fem.Field,
        v: wp.fem.Field,
        mu: wp.fem.Field,
        lam: wp.fem.Field,
        rho: wp.fem.Field,
        g: wp.vec3
    ):
        stress = stress_func(x, u, mu, lam)

        internal = wp.ddot(stress, wp.fem.grad(v, x))
        external = rho(x) * wp.dot(g, v(x)) # gravity

        return external - internal

    name = f'{stress_func.func.__name__}_residual_form'
    residual_form.__name__ = name
    residual_form.__qualname__ = name

    return wp.fem.integrand(residual_form)


def build_jacobian_form(tangent_func):

    def jacobian_form(
        x: wp.fem.Sample,
        u: wp.fem.Field,
        v: wp.fem.Field,
        du: wp.fem.Field,
        mu: wp.fem.Field,
        lam: wp.fem.Field
    ):
        tangent = tangent_func(x, u, du, mu, lam)
        return wp.ddot(tangent, wp.fem.grad(v, x))

    name = f'{tangent_func.func.__name__}_jacobian_form'
    jacobian_form.__name__ = name
    jacobian_form.__qualname__ = name

    return wp.fem.integrand(jacobian_form)


# ----- linear elasticity model -----


@wp.fem.integrand
def linear_elastic_stress(
    x: wp.fem.Sample,
    u: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field
):
    I = wp.identity(3, dtype=float)

    # kinematic quantities
    grad_u = wp.fem.grad(u, x)
    eps_u = 0.5 * (grad_u + wp.transpose(grad_u))
    div_u = wp.trace(grad_u)

    # isotropic Cauchy stress (sigma)
    return 2.0 * mu(x) * eps_u + lam(x) * div_u * I


@wp.fem.integrand
def linear_elastic_tangent(
    x: wp.fem.Sample,
    u: wp.fem.Field,
    du: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field
):
    return linear_elastic_stress(x, du, mu, lam)


# ----- St. Venant-Kirchoff model -----


@wp.fem.integrand
def st_venant_kirchoff_stress(
    x: wp.fem.Sample,
    u: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field
):
    I = wp.identity(3, dtype=float)

    # kinematic quantities
    F = I + wp.fem.grad(u, x)
    C = wp.transpose(F) * F
    E = 0.5 * (C - I)

    # second Piola-Kirchoff stress
    S = 2.0 * mu(x) * E + lam(x) * wp.trace(E) * I

    # first Piola-Kirchoff stress (P)
    return F * S


@wp.fem.integrand
def st_venant_kirchoff_tangent(
    x: wp.fem.Sample,
    u: wp.fem.Field,
    du: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field
):
    I = wp.identity(3, dtype=float)

    # kinematic quantities
    F = I + wp.fem.grad(u, x)
    C = wp.transpose(F) * F
    E = 0.5 * (C - I)

    # second Piola-Kirchoff stress
    S = 2.0 * mu(x) * E + lam(x) * wp.trace(E) * I

    # dP = d(F * S) = dF * S + F * dS
    dF = wp.fem.grad(du, x)
    dE = 0.5 * (wp.transpose(dF) * F + wp.transpose(F) * dF)
    dS = 2.0 * mu(x) * dE + lam(x) * wp.trace(dE) * I

    return dF * S + F * dS


# ----- Neo-Hookean model -----


@wp.fem.integrand
def neo_hookean_stress(
    x: wp.fem.Sample,
    u: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field
):
    I = wp.identity(3, dtype=float)

    # kinematic quantities
    F = I + wp.fem.grad(u, x)
    J = wp.determinant(F)
    F_inv = wp.inverse(F)
    F_inv_T = wp.transpose(F_inv)

    # W = mu/2 (I1 - 3 - 2 ln J) + lam/4 (J*J - 1 - 2 ln J)
    # W = C1 (I1 - 3) for J = 1 (no volumetric deformation)

    # first Piola-Kirchoff stress (P)
    return mu(x) * (F - F_inv_T) + 0.5 * lam(x) * (J*J - 1.) * F_inv_T


@wp.fem.integrand
def neo_hookean_tangent(
    x: wp.fem.Sample,
    u: wp.fem.Field,
    du: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field
):
    I = wp.identity(3, dtype=float)

    # kinematic quantities
    F = I + wp.fem.grad(u, x)
    J = wp.determinant(F)
    F_inv = wp.inverse(F)
    F_inv_T = wp.transpose(F_inv)

    # differential values
    dF = wp.fem.grad(du, x)
    dJ = J * wp.trace(F_inv * dF)
    dF_inv_T = -F_inv_T * wp.transpose(dF) * F_inv_T

    return (
        mu(x) * (dF - dF_inv_T)
        + lam(x) * J * dJ * F_inv_T
        + 0.5 * lam(x) * (J*J - 1.) * dF_inv_T
    )


# ----- Yeoh hyperelastic model -----


@wp.fem.integrand
def yeoh_hyperelastic_stress(
    x: wp.fem.Sample,
    u: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    a2: float,
    a3: float
):
    I = wp.identity(3, dtype=float)

    # kinematic quantities
    F = I + wp.fem.grad(u, x)
    J = wp.determinant(F)
    F_inv = wp.inverse(F)
    F_inv_T = wp.transpose(F_inv)

    # isochoric first invariant
    J_neg_2_3 = wp.pow(J, -2/3)
    I1 = wp.ddot(F, F)
    I1_bar = J_neg_2_3 * I1
    q_term = (I1_bar - 3.)

    # Yeoh parameters
    C1 = 0.5 * mu(x)
    C2 = a2 * C1
    C3 = a3 * C1

    # bulk modulus
    K = lam(x) + (2/3) * mu(x)

    # W = C1 (I1 - 3) + C2 (I1 - 3)^2 + C3 (I1 - 3)^3

    # phi = dW/dq
    phi = C1 + 2. * C2 * q_term + 3. * C3 * q_term * q_term

    # first Piola-Kirchoff stress
    H = F - (I1 / 3) * F_inv_T
    P_iso = 2.0 * phi * J_neg_2_3 * H
    P_vol = K * J * (J - 1.) * F_inv_T

    return P_iso + P_vol


@wp.fem.integrand
def yeoh_hyperelastic_tangent(
    x: wp.fem.Sample,
    u: wp.fem.Field, 
    du: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    a2: float,
    a3: float
):
    I = wp.identity(3, dtype=float)

    # kinematic quantities
    F = I + wp.fem.grad(u, x)
    J = wp.determinant(F)
    F_inv = wp.inverse(F)
    F_inv_T = wp.transpose(F_inv)

    # basic differentials
    dF = wp.fem.grad(du, x)
    tr = wp.trace(F_inv * dF)
    dJ = J * tr
    dF_inv_T = -F_inv_T * wp.transpose(dF) * F_inv_T

    # first invariant differentials
    I1 = wp.ddot(F, F)
    dI1 = 2 * wp.ddot(F, dF)

    J_neg_2_3 = wp.pow(J, -2/3)
    dJ_neg_2_3 = -(2/3) * J_neg_2_3 * tr

    I1_bar = J_neg_2_3 * I1
    q_term = (I1_bar - 3.0)
    dq_term = dJ_neg_2_3 * I1 + J_neg_2_3 * dI1

    # Yeoh parameters
    C1 = mu(x) / 2
    C2 = a2 * C1
    C3 = a3 * C1

    # bulk modulus
    K = lam(x) + (2/3) * mu(x)

    # phi = dW/dq
    phi = C1 + 2 * C2 * q_term + 3 * C3 * q_term * q_term
    dphi = (2 * C2 + 6 * C3 * q_term) * dq_term

    # isochoric Piola stress tangent
    H = F - (I1 / 3) * F_inv_T
    dH = dF - (dI1 / 3) * F_inv_T - (I1/3) * dF_inv_T
    dP_iso = 2 * (dphi * J_neg_2_3 * H + phi * dJ_neg_2_3 * H + phi * J_neg_2_3 * dH)

    # volumetric Piola stress tangent
    sJ = J * (J - 1)
    dsJ = (2*J - 1) * dJ
    dP_vol = K * (dsJ * F_inv_T + sJ * dF_inv_T)

    return dP_iso + dP_vol


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
def TV_reg_form(x: wp.fem.Sample, mu: wp.fem.Field, eps_reg: float, eps_div: float):
    '''Smooth TV penalty on log-parameter gradient.'''
    grad_mu = wp.fem.grad(mu, x) / (mu(x) + eps_div)
    return wp.sqrt(wp.dot(grad_mu, grad_mu) + eps_reg * eps_reg)

