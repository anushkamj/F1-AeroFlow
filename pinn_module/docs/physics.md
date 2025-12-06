# Incompressible Navier–Stokes Equations (2D)

# Continuity equation (mass conservation):
∂u/∂x + ∂v/∂y = 0

# Momentum equations (steady-state):
# u and v are velocity components, p is pressure, ρ is density, ν is kinematic viscosity.

# x-momentum:
u * (∂u/∂x) + v * (∂u/∂y) = -(1/ρ) * (∂p/∂x) + ν * ( ∂²u/∂x² + ∂²u/∂y² )

# y-momentum:
u * (∂v/∂x) + v * (∂v/∂y) = -(1/ρ) * (∂p/∂y) + ν * ( ∂²v/∂x² + ∂²v/∂y² )