# PINN Physics Simulator

This module provides a Physics-Informed Neural Network (PINN) for predicting aerodynamic coefficients (Cd, Cl) of F1 components.

## Files
- model.py – PINN architecture (4 layers, 128 neurons)
- physics_loss.py – Navier–Stokes + continuity residuals
- simulator.py – Inference wrapper returning (Cd, Cl)
- validation.py – CFD accuracy checker
- docs/ – Physics documentation

## Usage
from pinn_module.simulator import PINNSimulator

sim = PINNSimulator()
Cd, Cl = sim.simulate(params)
print(Cd, Cl)

INPUT PARAMS (dict):
{
  "angle": float,
  "chord": float,
  "thickness": float,
  ...
  (12 total parameters)
}

OUTPUT:
{
  "Cd": float,
  "Cl": float
}
