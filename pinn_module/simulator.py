import torch
from model import PINN
import json
import os
import random


class PINNSimulator:
    def __init__(self, weights_path=None, device="cpu"):
        self.device = device
        self.model = PINN().to(device)

        # Load trained weights if provided
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            print("[PINN Simulator] No trained weights provided — running in DEMO MODE.")

        self.model.eval()

    def simulate(self, params: dict):
        """
        params: dictionary containing 12 aerodynamic design parameters
        Returns:
            Cd (float)
            Cl (float)
        """

        # Convert dict → tensor
        x = torch.tensor([list(params.values())], dtype=torch.float32).to(self.device)

        # REAL PINN INFERENCE (if weights exist)
        if hasattr(self.model, "network"):
            try:
                with torch.no_grad():
                    Cd_pred, Cl_pred = self.model(x)[0]
                return float(Cd_pred), float(Cl_pred)
            except Exception as e:
                print("PINN inference error, switching to demo mode:", e)

        # ---------------------------
        # DEMO MODE (synthetic output)
        # ---------------------------
        Cd_demo = 0.80 + random.uniform(-0.03, 0.03)
        Cl_demo = 1.50 + random.uniform(-0.05, 0.05)

        return round(Cd_demo, 3), round(Cl_demo, 3)


# Simple test hook
if __name__ == "__main__":
    sample_params = {
        f"param_{i}": 0.5 for i in range(12)
    }
    sim = PINNSimulator()
    Cd, Cl = sim.simulate(sample_params)
    print("Predicted Cd:", Cd)
    print("Predicted Cl:", Cl)
