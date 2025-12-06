# PINN Loss Function

# Total loss:
L_total = L_data + L_physics

# Data loss (match CFD results):
L_data = MSE(predicted_Cd, CFD_Cd) + MSE(predicted_Cl, CFD_Cl)

# Physics loss (enforce Navier–Stokes):
L_physics = MSE( continuity_residual ) + MSE( momentum_residual_x ) + MSE( momentum_residual_y )

# Final combined objective:
L_total = λ_data * L_data + λ_phys * L_physics
