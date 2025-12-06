import json
from pinn_module import PINNSimulator, validate_against_cfd, load_baseline_case

# Load sample aerodynamic parameters
with open("pinn_module/sample_params.json", "r") as f:
    sample_params = json.load(f)

# Instantiate simulator
sim = PINNSimulator()

# Run PINN prediction
Cd, Cl = sim.simulate(sample_params)

print("\n--- PINN Simulation Results ---")
print("Cd (drag coefficient):", Cd)
print("Cl (lift coefficient):", Cl)

# Load CFD baseline
baseline = load_baseline_case("baseline_1")

# Validation
metrics = validate_against_cfd(
    pred_cd=Cd,
    pred_cl=Cl,
    baseline_cd=baseline["Cd"],
    baseline_cl=baseline["Cl"]
)

print("\n--- Validation vs CFD Baseline ---")
print(metrics)
print("\nSimulation complete.\n")
