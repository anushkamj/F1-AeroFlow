import torch
from pinn_module.model import PINN
import os

class PINNSimulator:
    def __init__(self, weights_path=None, device="cpu"):
        self.device = device
        self.model = PINN().to(device)

        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            print("[PINN Simulator] Loaded trained weights successfully.")
            self.trained = True
        else:
            raise RuntimeError(
                "ERROR: No trained PINN weights found. "
                "This is not a demo version — training weights are required."
            )

        self.model.eval()

    def simulate(self, params: dict):
        """
        params should contain EXACTLY the 6 aerodynamic parameters:
            - alpha_norm
            - G_ratio
            - C_main_norm
            - lambda_ratio
            - theta_flap
            - tau_taper
        """

        expected_keys = [
            "alpha_norm",
            "G_ratio",
            "C_main_norm",
            "lambda_ratio",
            "theta_flap",
            "tau_taper"
        ]

        # Verify correct parameter set
        if list(params.keys()) != expected_keys:
            raise ValueError(
                f"Incorrect parameter keys.\nExpected: {expected_keys}\nReceived: {list(params.keys())}"
            )

        # Convert to tensor
        x = torch.tensor([list(params.values())], dtype=torch.float32).to(self.device)

        # RUN REAL PINN INFERENCE
        with torch.no_grad():
            Cd_pred, Cl_pred = self.model(x)[0]

        return float(Cd_pred), float(Cl_pred)


# Manual test hook
if __name__ == "__main__":
    sample_params = {
        "alpha_norm": 0.5,
        "G_ratio": 0.4,
        "C_main_norm": 0.85,
        "lambda_ratio": 0.5,
        "theta_flap": 12.0,
        "tau_taper": 0.8
    }

    sim = PINNSimulator(weights_path="pinn_weights.pth")
    Cd, Cl = sim.simulate(sample_params)
    print("Cd:", Cd)
    print("Cl:", Cl)
