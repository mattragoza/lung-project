import warp as wp
import warp.fem


@wp.fem.integrand
def pde_bilinear_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    v: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    I: wp.mat33
):
    eps_u = wp.fem.D(u, s) # symmetric gradient
    eps_v = wp.fem.D(v, s)
    div_u = wp.fem.div(u, s)
    sigma_u = 2.0*mu(s)*eps_u + lam(s)*div_u*I
    return wp.ddot(sigma_u, eps_v)


@wp.fem.integrand
def pde_linear_form(
    s: wp.fem.Sample,
    v: wp.fem.Field,
    rho: wp.fem.Field,
    g: wp.vec3
):
    return rho(s) * wp.dot(g, v(s))


@wp.fem.integrand
def pde_residual_form(
    s: wp.fem.Sample,
    u: wp.fem.Field,
    v: wp.fem.Field,
    mu: wp.fem.Field,
    lam: wp.fem.Field,
    rho: wp.fem.Field,
    g: wp.vec3,
    I: wp.mat33
):
    lhs = pde_bilinear_form(s, u, v, mu, lam, I)
    rhs = pde_linear_form(s, v, rho, g)
    return rhs - lhs


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
def volume_form(s: wp.fem.Sample, w: wp.fem.Field):
    return w(s)


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

