import json
from pinn_module import PINNSimulator, validate_against_cfd, load_baseline_case

# Load sample aerodynamic parameters (6-parameter vector)
with open("pinn_module/sample_params.json", "r") as f:
    params = json.load(f)

# Load simulator (real model only)
sim = PINNSimulator(weights_path="pinn_weights.pth")

# Run simulation
Cd, Cl = sim.simulate(params)

print("\n--- PINN Output ---")
print("Cd:", Cd)
print("Cl:", Cl)

# Load CFD baseline for validation
baseline = load_baseline_case("baseline_1")

metrics = validate_against_cfd(
    pred_cd=Cd,
    pred_cl=Cl,
    baseline_cd=baseline["Cd"],
    baseline_cl=baseline["Cl"]
)

print("\n--- Validation Metrics ---")
print(metrics)
