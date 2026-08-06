import warp as wp
import warp.fem


def _resolve_material_type(name: str):
    n = name.lower()
    if n in {'linear_elastic', 'linear', 'le'}:
        return LinearElasticMaterial
    elif n in {'st_venant_kirchoff', 'stvk', 'vk'}:
        return StVenantKirchoffMaterial
    elif n in {'neo_hookean', 'neohookean', 'nh'}:
        return NeoHookeanMaterial
    raise ValueError(f'Invalid material type: {name!r}')


# ----- linear elasticity -----


@wp.fem.integrand
def linear_elastic_residual_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    v: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    rho: wp.fem.Field,
    g: wp.vec3,
    I: wp.mat33
):
    eps_u = wp.fem.D(u, s) # infinitesimal strain
    div_u = wp.fem.div(u, s)

    # Cauchy stress tensor
    sigma = lam(s) * div_u * I + 2.0 * mu(s) * eps_u

    internal = wp.ddot(sigma, wp.fem.D(v, s))
    external = gravitational_load_form(s, v, rho, g)

    return external - internal


@wp.fem.integrand
def linear_elastic_jacobian_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,  # current state (not used in LE)
    du: wp.fem.Field, # Newton update / trial function
    v: wp.fem.Field,  # test function
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    I: wp.mat33
):
    eps_u = wp.fem.D(du, s) # infinitesimal strain
    div_u = wp.fem.div(du, s)

    # Cauchy stress tensor
    sigma = lam(s) * div_u * I + 2.0 * mu(s) * eps_u

    return wp.ddot(sigma, wp.fem.D(v, s))


# ----- geometric nonlinearity -----


@wp.fem.integrand
def st_venant_kirchoff_residual_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    v: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    rho: wp.fem.Field,
    g: wp.vec3,
    I: wp.mat33
):
    F = I + wp.fem.grad(u, s) # deformation gradient
    C = wp.transpose(F) * F   # right CG deformation
    E = 0.5 * (C - I)         # Lagrangian finite strain

    # second Piola-Kirchoff stress tensor
    S = lam(s) * wp.trace(E) * I + 2.0 * mu(s) * E

    # first Piola-Kirchoff stress tensor
    P = F * S

    internal = wp.ddot(P, wp.fem.grad(v, s))
    external = gravitational_load_form(s, v, rho, g)

    return external - internal


@wp.fem.integrand
def st_venant_kirchoff_jacobian_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    du: wp.fem.Field,
    v: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    I: wp.mat33
):
    F = I + wp.fem.grad(u, s) # deformation gradient
    C = wp.transpose(F) * F   # right CG deformation
    E = 0.5 * (C - I)         # Lagrangian finite strain

    # second Piola-Kirchoff stress tensor
    S = lam(s) * wp.trace(E) * I + 2.0 * mu(s) * E

    dF = wp.fem.grad(du, s)
    dE = 0.5 * (wp.transpose(dF) * F + wp.transpose(F) * dF)
    dS = lam(s) * wp.trace(dE) * I + 2.0 * mu(s) * dE
    dP = dF * S + F * dS

    return wp.ddot(dP, wp.fem.D(v, s))


# ----- material nonlinearity -----


@wp.fem.integrand
def neo_hookean_residual_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    v: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    rho: wp.fem.Field,
    g: wp.vec3,
    I: wp.mat33
):
    F = I + wp.fem.grad(u, s) # deformation gradient

    FinvT = wp.transpose(wp.inverse(F))
    J = wp.determinant(F)

    # first Piola-Kirchoff stress tensor
    P = (
        0.5 * lam(s) * (J*J - 1.0) * FinvT
        + mu(s) * (F - FinvT)
    )

    internal = wp.ddot(P, wp.fem.grad(v, s))
    external = gravitational_load_form(s, v, rho, g)

    return external - internal


@wp.fem.integrand
def neo_hookean_jacobian_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,  # current state
    du: wp.fem.Field, # Newton update / trial function
    v: wp.fem.Field,  # test function
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    I: wp.mat33
):
    F = I + wp.fem.grad(u, s)

    Finv = wp.inverse(F)
    FinvT = wp.transpose(Finv)
    J = wp.determinant(F)

    dF = wp.fem.grad(du, s)
    dJ = J * wp.trace(Finv * dF)
    dFinvT = -FinvT * wp.transpose(dF) * FinvT

    dP = (
        lam(s) * J * dJ * FinvT
        + 0.5 * lam(s) * (J*J - 1.0) * dFinvT
        + mu(s) * (dF - dFinvT)
    )
    return wp.ddot(dP, wp.fem.grad(v, s))


# ----- generic forms -----


@wp.fem.integrand
def gravitational_load_form(
    s: wp.fem.Sample,
    v: wp.fem.Field,
    rho: wp.fem.Field,
    g: wp.vec3
):
    return rho(s) * wp.dot(g, v(s))


@wp.fem.integrand
def inner_product_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    v: wp.fem.Field
):
    return wp.dot(u(s), v(s))


@wp.fem.integrand
def squared_error_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    v: wp.fem.Field,
    w: wp.fem.Field
):
    r_s = u(s) - v(s)
    return w(s) * wp.dot(r_s, r_s)


@wp.fem.integrand
def squared_norm_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    w: wp.fem.Field
):
    u_s = u(s)
    return w(s) * wp.dot(u_s, u_s)


@wp.fem.integrand
def l2_norm_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
):
    u_s = u(s)
    return wp.sqrt(wp.dot(u_s, u_s))


@wp.fem.integrand
def tv_regularization_form(
    s: wp.fem.Sample,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    rho: wp.fem.Field,
    eps_reg: float,
    eps_div: float,
):
    # TV regularization on gradient of log params
    grad_mu = wp.fem.grad(mu, s) / (mu(s) + eps_div)
    grad_lam = wp.fem.grad(lam, s) / (lam(s) + eps_div)
    grad_rho = wp.fem.grad(rho, s) / (rho(s) + eps_div)
    return (
        wp.sqrt(wp.dot(grad_mu, grad_mu) + eps_reg * eps_reg) +
        wp.sqrt(wp.dot(grad_lam, grad_lam) + eps_reg * eps_reg) +
        wp.sqrt(wp.dot(grad_rho, grad_rho) + eps_reg * eps_reg)
    )


# ----- material type registry -----


class WarpMaterial:

    @staticmethod
    def get_linear():
        return LinearElasticMaterial

    @staticmethod
    def get_subclass(name: str):
        return _resolve_material_type(name)


class LinearElasticMaterial(WarpMaterial):
    residual_form = linear_elastic_residual_form
    jacobian_form = linear_elastic_jacobian_form
    is_linear = True


class StVenantKirchoffMaterial(WarpMaterial):
    residual_form = st_venant_kirchoff_residual_form
    jacobian_form = st_venant_kirchoff_jacobian_form
    is_linear = False


class NeoHookeanMaterial(WarpMaterial):
    residual_form = neo_hookean_residual_form
    jacobian_form = neo_hookean_jacobian_form
    is_linear = False

