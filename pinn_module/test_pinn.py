from pinn_module import PINNSimulator, validate_against_cfd, load_baseline_case

sim = PINNSimulator()

params = {f"param_{i}": 0.5 for i in range(12)}
Cd, Cl = sim.simulate(params)

print("Cd:", Cd)
print("Cl:", Cl)

baseline = load_baseline_case("baseline_1")
metrics = validate_against_cfd(Cd, Cl, baseline["Cd"], baseline["Cl"])

print("Validation metrics:", metrics)
