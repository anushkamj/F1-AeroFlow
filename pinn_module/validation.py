def validate_against_cfd(pred_cd, pred_cl, baseline_cd, baseline_cl):
    """
    Validation compares Cd and Cl predictions from REAL PINN inference
    against reference CFD values.

    This version assumes inputs come from the 6-parameter rear wing space:
        alpha_norm, G_ratio, C_main_norm, lambda_ratio, theta_flap, tau_taper
    """
    cd_error = abs(pred_cd - baseline_cd)
    cl_error = abs(pred_cl - baseline_cl)

    cd_percentage_error = (cd_error / baseline_cd) * 100 if baseline_cd != 0 else 0
    cl_percentage_error = (cl_error / baseline_cl) * 100 if baseline_cl != 0 else 0

    return {
        "Cd_Error": round(cd_error, 4),
        "Cl_Error": round(cl_error, 4),
        "Cd_Percentage_Error": round(cd_percentage_error, 2),
        "Cl_Percentage_Error": round(cl_percentage_error, 2)
    }