import json
import numpy as np


def load_baseline_case(case_id="baseline_1", path="validation_data.json"):
    """
    Loads CFD baseline case for comparison.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data.get(case_id, None)


def validate_against_cfd(pred_cd, pred_cl, baseline_cd, baseline_cl):
    """
    Returns accuracy metrics for Cd and Cl predictions.
    """

    cd_error = abs(pred_cd - baseline_cd) / baseline_cd
    cl_error = abs(pred_cl - baseline_cl) / baseline_cl

    metrics = {
        "cd_accuracy": round(1 - cd_error, 4),
        "cl_accuracy": round(1 - cl_error, 4),
        "passes_accuracy_requirement": (1 - cd_error > 0.96 and 1 - cl_error > 0.96)
    }

    return metrics


def physics_residual_check(cont_residual, mom_residual):
    """
    Checks if physics residuals satisfy < 0.001 normalized condition.
    """
    total_res = float(cont_residual.mean() + mom_residual.mean())
    return total_res < 0.001


# Example test
if __name__ == "__main__":
    baseline = load_baseline_case("baseline_1")
    metrics = validate_against_cfd(
        pred_cd=0.81,
        pred_cl=1.49,
        baseline_cd=baseline["Cd"],
        baseline_cl=baseline["Cl"]
    )
    print(metrics)
