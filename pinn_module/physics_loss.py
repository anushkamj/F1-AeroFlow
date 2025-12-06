import torch

def continuity_residual(u, v, x, y):
    """
    Enforce incompressibility: ∂u/∂x + ∂v/∂y = 0
    """
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    return (du_dx + dv_dy)**2


def momentum_residual(u, v, p, x, y, rho=1.225, nu=1.5e-5):
    """
    Enforce Navier–Stokes in 2D steady-state:
    u∂u/∂x + v∂u/∂y = -1/ρ ∂p/∂x + ν ∇²u
    u∂v/∂x + v∂v/∂y = -1/ρ ∂p/∂y + ν ∇²v
    """

    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    dv_dx = torch.autograd.grad(v, x, torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    dv_dy = torch.autograd.grad(v, y, torch.ones_like(v), retain_graph=True, create_graph=True)[0]

    dp_dx = torch.autograd.grad(p, x, torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    dp_dy = torch.autograd.grad(p, y, torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    # Laplacians (second derivatives)
    d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), retain_graph=True, create_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y, torch.ones_like(du_dy), retain_graph=True, create_graph=True)[0]
    lap_u = d2u_dx2 + d2u_dy2

    d2v_dx2 = torch.autograd.grad(dv_dx, x, torch.ones_like(dv_dx), retain_graph=True, create_graph=True)[0]
    d2v_dy2 = torch.autograd.grad(dv_dy, y, torch.ones_like(dv_dy), retain_graph=True, create_graph=True)[0]
    lap_v = d2v_dx2 + d2v_dy2

    # Residuals
    res_u = u * du_dx + v * du_dy + (1/rho) * dp_dx - nu * lap_u
    res_v = u * dv_dx + v * dv_dy + (1/rho) * dp_dy - nu * lap_v

    return res_u**2 + res_v**2
