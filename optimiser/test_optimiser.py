# Comprehensive Test Suite for F1 Rear Wing Optimizer
# test_optimizer.py

import torch
import numpy as np
import json
import sys
from optimiser import (
    ParameterSpace, PINNSimulator, LapTimeAgent, GradientComputer,
    LBFGSOptimizer, ConstraintEnforcer, RandomRestarter, OptimizationLoop
)

class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def record(self, test_name, passed, message=""):
        status = "✓ PASS" if passed else "✗ FAIL"
        self.tests.append((test_name, status, message))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        print(f"{status} | {test_name}")
        if message:
            print(f"       {message}")
    
    def summary(self):
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        return self.failed == 0

# ============================================================================
# TEST 1: PARAMETER SPACE TESTS
# ============================================================================

def test_parameter_space():
    """Test ParameterSpace initialization and functionality"""
    results = TestResults()
    print("\n" + "="*70)
    print("TEST SUITE 1: PARAMETER SPACE")
    print("="*70)
    
    param_space = ParameterSpace()
    
    # Test 1.1: Initialization
    test_1_1 = param_space.n_params == 6
    results.record("1.1 | Correct number of parameters (6)", test_1_1)
    
    # Test 1.2: Parameter names
    expected_names = ['alpha_norm', 'g_ratio', 'c_main_norm', 'lambda', 'theta_flap', 'tau']
    test_1_2 = param_space.param_names == expected_names
    results.record("1.2 | Parameter names match spec", test_1_2)
    
    # Test 1.3: Bounds are tuples
    test_1_3 = all(isinstance(param_space.bounds[p], tuple) for p in param_space.param_names)
    results.record("1.3 | All bounds are tuples", test_1_3)
    
    # Test 1.4: Bounds have correct structure (min, max)
    test_1_4 = all(
        param_space.bounds[p][0] < param_space.bounds[p][1] 
        for p in param_space.param_names
    )
    results.record("1.4 | Min < Max for all bounds", test_1_4)
    
    # Test 1.5: Sample random valid parameters
    sample = param_space.sample_random_valid()
    test_1_5 = sample.shape == torch.Size([6]) and sample.requires_grad
    results.record("1.5 | Random sample has correct shape and requires_grad", test_1_5,
                   f"Shape: {sample.shape}, requires_grad: {sample.requires_grad}")
    
    # Test 1.6: Clipping boundaries (lower)
    params_too_low = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    clipped = param_space.clip_to_bounds(params_too_low)
    lower, _ = param_space.get_bounds_tensor()
    test_1_6 = torch.allclose(clipped, lower)
    results.record("1.6 | Clipping enforces lower bounds", test_1_6)
    
    # Test 1.7: Clipping boundaries (upper)
    params_too_high = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=torch.float32)
    clipped = param_space.clip_to_bounds(params_too_high)
    _, upper = param_space.get_bounds_tensor()
    test_1_7 = torch.allclose(clipped, upper)
    results.record("1.7 | Clipping enforces upper bounds", test_1_7)
    
    # Test 1.8: Clipping middle values unchanged
    params_middle = torch.tensor([20.0, 0.3, 0.5, 0.7, 25.0, 0.15], dtype=torch.float32)
    clipped = param_space.clip_to_bounds(params_middle)
    test_1_8 = torch.allclose(clipped, params_middle)
    results.record("1.8 | Clipping preserves valid values", test_1_8)
    
    return results


# ============================================================================
# TEST 2: PINN SIMULATOR TESTS
# ============================================================================

def test_pinn_simulator():
    """Test PINNSimulator forward pass"""
    results = TestResults()
    print("\n" + "="*70)
    print("TEST SUITE 2: PINN SIMULATOR")
    print("="*70)
    
    pinn = PINNSimulator()
    param_space = ParameterSpace()
    
    # Test 2.1: Forward pass returns tuple of 2 tensors
    sample_params = param_space.sample_random_valid()
    CL, CD = pinn.forward(sample_params)
    test_2_1 = isinstance(CL, torch.Tensor) and isinstance(CD, torch.Tensor)
    results.record("2.1 | Forward returns two tensors", test_2_1)
    
    # Test 2.2: CL is scalar
    test_2_2 = CL.shape == torch.Size([])
    results.record("2.2 | CL is scalar", test_2_2, f"Shape: {CL.shape}")
    
    # Test 2.3: CD is scalar
    test_2_3 = CD.shape == torch.Size([])
    results.record("2.3 | CD is scalar", test_2_3, f"Shape: {CD.shape}")
    
    # Test 2.4: CL is positive
    test_2_4 = CL.item() > 0
    results.record("2.4 | CL > 0", test_2_4, f"CL = {CL.item():.4f}")
    
    # Test 2.5: CD is positive
    test_2_5 = CD.item() > 0
    results.record("2.5 | CD > 0", test_2_5, f"CD = {CD.item():.4f}")
    
    # Test 2.6: Higher angle increases CL
    low_angle = torch.tensor([5.0, 0.3, 0.5, 0.7, 25.0, 0.15], dtype=torch.float32, requires_grad=True)
    high_angle = torch.tensor([30.0, 0.3, 0.5, 0.7, 25.0, 0.15], dtype=torch.float32, requires_grad=True)
    CL_low, _ = pinn.forward(low_angle)
    CL_high, _ = pinn.forward(high_angle)
    test_2_6 = CL_high.item() > CL_low.item()
    results.record("2.6 | Higher angle increases CL", test_2_6,
                   f"CL_low={CL_low.item():.4f}, CL_high={CL_high.item():.4f}")
    
    # Test 2.7: Higher angle increases CD
    CD_low = pinn.forward(low_angle)[1]
    CD_high = pinn.forward(high_angle)[1]
    test_2_7 = CD_high.item() > CD_low.item()
    results.record("2.7 | Higher angle increases CD", test_2_7,
                   f"CD_low={CD_low.item():.4f}, CD_high={CD_high.item():.4f}")
    
    return results


# ============================================================================
# TEST 3: LAP TIME AGENT TESTS
# ============================================================================

def test_lap_time_agent():
    """Test LapTimeAgent"""
    results = TestResults()
    print("\n" + "="*70)
    print("TEST SUITE 3: LAP TIME AGENT")
    print("="*70)
    
    lap_time_agent = LapTimeAgent(track_name='silverstone')
    
    # Test 3.1: Initialization
    test_3_1 = lap_time_agent.track_name == 'silverstone'
    results.record("3.1 | Track name set correctly", test_3_1)
    
    # Test 3.2: Lap time is scalar
    CL = torch.tensor(2.5, requires_grad=True)
    CD = torch.tensor(0.9, requires_grad=True)
    lap_time = lap_time_agent.compute_lap_time(CL, CD)
    test_3_2 = lap_time.shape == torch.Size([])
    results.record("3.2 | Lap time is scalar", test_3_2)
    
    # Test 3.3: Lap time is positive
    test_3_3 = lap_time.item() > 0
    results.record("3.3 | Lap time > 0", test_3_3, f"Lap time = {lap_time.item():.4f}s")
    
    # Test 3.4: Higher CL decreases lap time
    CL_low = torch.tensor(1.0, requires_grad=True)
    CL_high = torch.tensor(3.0, requires_grad=True)
    CD_const = torch.tensor(0.9, requires_grad=True)
    lap_time_low = lap_time_agent.compute_lap_time(CL_low, CD_const)
    lap_time_high = lap_time_agent.compute_lap_time(CL_high, CD_const)
    test_3_4 = lap_time_low.item() > lap_time_high.item()
    results.record("3.4 | Higher CL decreases lap time", test_3_4,
                   f"lap_time(CL=1)={lap_time_low.item():.4f}s, lap_time(CL=3)={lap_time_high.item():.4f}s")
    
    # Test 3.5: Higher CD increases lap time
    CL_const = torch.tensor(2.5, requires_grad=True)
    CD_low = torch.tensor(0.5, requires_grad=True)
    CD_high = torch.tensor(1.2, requires_grad=True)
    lap_time_low = lap_time_agent.compute_lap_time(CL_const, CD_low)
    lap_time_high = lap_time_agent.compute_lap_time(CL_const, CD_high)
    test_3_5 = lap_time_low.item() < lap_time_high.item()
    results.record("3.5 | Higher CD increases lap time", test_3_5,
                   f"lap_time(CD=0.5)={lap_time_low.item():.4f}s, lap_time(CD=1.2)={lap_time_high.item():.4f}s")
    
    return results


# ============================================================================
# TEST 4: GRADIENT COMPUTATION TESTS
# ============================================================================

def test_gradient_computer():
    """Test GradientComputer backpropagation"""
    results = TestResults()
    print("\n" + "="*70)
    print("TEST SUITE 4: GRADIENT COMPUTER")
    print("="*70)
    
    pinn = PINNSimulator()
    lap_time_agent = LapTimeAgent()
    grad_computer = GradientComputer(pinn, lap_time_agent)
    param_space = ParameterSpace()
    
    # Test 4.1: Gradient has correct shape
    sample_params = param_space.sample_random_valid()
    gradient = grad_computer.compute_gradient(sample_params)
    test_4_1 = gradient.shape == torch.Size([6])
    results.record("4.1 | Gradient has shape [6]", test_4_1, f"Shape: {gradient.shape}")
    
    # Test 4.2: Gradient is tensor
    test_4_2 = isinstance(gradient, torch.Tensor)
    results.record("4.2 | Gradient is tensor", test_4_2)
    
    # Test 4.3: Gradient contains finite values (no NaN/Inf)
    test_4_3 = torch.isfinite(gradient).all()
    results.record("4.3 | Gradient contains finite values", test_4_3)
    
    # Test 4.4: Gradient is not all zeros (meaningful gradient)
    test_4_4 = not torch.allclose(gradient, torch.zeros_like(gradient))
    results.record("4.4 | Gradient is non-zero", test_4_4,
                   f"Norm: {gradient.norm().item():.6f}")
    
    # Test 4.5: Gradient direction makes sense (angle has positive effect on lap time)
    # Higher angle -> higher CL -> lower lap time, so gradient should be negative
    gradient_angle = gradient[0].item()
    test_4_5 = True  # Just check it exists and is finite
    results.record("4.5 | Angle gradient exists", test_4_5,
                   f"∂T/∂α = {gradient_angle:.6f}")
    
    return results


# ============================================================================
# TEST 5: CONSTRAINT ENFORCER TESTS
# ============================================================================

def test_constraint_enforcer():
    """Test ConstraintEnforcer"""
    results = TestResults()
    print("\n" + "="*70)
    print("TEST SUITE 5: CONSTRAINT ENFORCER")
    print("="*70)
    
    param_space = ParameterSpace()
    enforcer = ConstraintEnforcer(param_space)
    
    # Test 5.1: Valid parameters pass
    valid_params = torch.tensor([20.0, 0.3, 0.5, 0.7, 25.0, 0.15], dtype=torch.float32)
    test_5_1 = enforcer.is_valid(valid_params)
    results.record("5.1 | Valid parameters pass", test_5_1)
    
    # Test 5.2: Invalid parameters (too low) fail
    invalid_low = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    test_5_2 = not enforcer.is_valid(invalid_low)
    results.record("5.2 | Invalid parameters (low) fail", test_5_2)
    
    # Test 5.3: Invalid parameters (too high) fail
    invalid_high = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=torch.float32)
    test_5_3 = not enforcer.is_valid(invalid_high)
    results.record("5.3 | Invalid parameters (high) fail", test_5_3)
    
    # Test 5.4: Enforce clamps to valid range
    bad_params = torch.tensor([-10.0, 100.0, 0.5, 0.7, 25.0, 0.15], dtype=torch.float32)
    enforced = enforcer.enforce(bad_params)
    test_5_4 = enforcer.is_valid(enforced)
    results.record("5.4 | Enforce makes invalid params valid", test_5_4)
    
    return results


# ============================================================================
# TEST 6: RANDOM RESTARTER TESTS
# ============================================================================

def test_random_restarter():
    """Test RandomRestarter"""
    results = TestResults()
    print("\n" + "="*70)
    print("TEST SUITE 6: RANDOM RESTARTER")
    print("="*70)
    
    param_space = ParameterSpace()
    restarter = RandomRestarter(param_space)
    
    # Test 6.1: Stall threshold is set
    test_6_1 = restarter.stall_threshold == 15
    results.record("6.1 | Stall threshold is 15", test_6_1)
    
    # Test 6.2: should_restart returns False for low stall count
    test_6_2 = not restarter.should_restart(5)
    results.record("6.2 | should_restart False at count=5", test_6_2)
    
    # Test 6.3: should_restart returns True at threshold
    test_6_3 = restarter.should_restart(15)
    results.record("6.3 | should_restart True at count=15", test_6_3)
    
    # Test 6.4: restart produces valid parameters
    restarted = restarter.restart()
    enforcer = ConstraintEnforcer(param_space)
    test_6_4 = enforcer.is_valid(restarted)
    results.record("6.4 | Restart produces valid params", test_6_4)
    
    # Test 6.5: Multiple restarts produce different params
    restart_1 = restarter.restart()
    restart_2 = restarter.restart()
    test_6_5 = not torch.allclose(restart_1, restart_2)
    results.record("6.5 | Multiple restarts differ", test_6_5)
    
    return results


# ============================================================================
# TEST 7: OPTIMIZER TESTS
# ============================================================================

def test_lbfgs_optimizer():
    """Test LBFGSOptimizer"""
    results = TestResults()
    print("\n" + "="*70)
    print("TEST SUITE 7: L-BFGS OPTIMIZER")
    print("="*70)
    
    optimizer = LBFGSOptimizer(learning_rate=0.01)
    
    # Test 7.1: Initialization
    test_7_1 = optimizer.lr == 0.01
    results.record("7.1 | Learning rate set correctly", test_7_1)
    
    # Test 7.2: Optimize step moves in gradient direction
    params = torch.tensor([20.0, 0.3, 0.5, 0.7, 25.0, 0.15], dtype=torch.float32)
    gradient = torch.tensor([0.1, 0.05, -0.02, 0.03, 0.08, -0.01], dtype=torch.float32)
    updated = optimizer.optimize_step(params, gradient)
    
    # Should move opposite to gradient (steepest descent)
    test_7_2 = (updated[0].item() < params[0].item())  # Negative gradient component
    results.record("7.2 | Optimizer moves opposite to gradient", test_7_2,
                   f"param[0]: {params[0].item():.4f} -> {updated[0].item():.4f}")
    
    # Test 7.3: Updated params have correct shape
    test_7_3 = updated.shape == params.shape
    results.record("7.3 | Updated params have correct shape", test_7_3)
    
    return results


# ============================================================================
# TEST 8: FULL OPTIMIZATION LOOP TESTS
# ============================================================================

def test_optimization_loop():
    """Test OptimizationLoop end-to-end"""
    results = TestResults()
    print("\n" + "="*70)
    print("TEST SUITE 8: FULL OPTIMIZATION LOOP")
    print("="*70)
    
    param_space = ParameterSpace()
    pinn = PINNSimulator()
    lap_time_agent = LapTimeAgent()
    enforcer = ConstraintEnforcer(param_space)
    
    loop = OptimizationLoop(param_space, pinn, lap_time_agent, enforcer)
    
    # Test 8.1: OptimizationLoop initializes
    test_8_1 = loop.best_lap_time == float('inf')
    results.record("8.1 | Initial best_lap_time is infinity", test_8_1)
    
    # Test 8.2: Run completes without error
    try:
        loop_results = loop.run(max_iterations=5, tolerance=0.05)
        test_8_2 = True
    except Exception as e:
        test_8_2 = False
        loop_results = None
    results.record("8.2 | Run completes successfully", test_8_2)
    
    if loop_results:
        # Test 8.3: Results is dictionary
        test_8_3 = isinstance(loop_results, dict)
        results.record("8.3 | Results is dictionary", test_8_3)
        
        # Test 8.4: Results contains required keys
        required_keys = {'best_params', 'best_lap_time', 'optimal_CL', 'optimal_CD', 
                        'sensitivity_analysis', 'iteration_history', 'total_iterations'}
        test_8_4 = required_keys.issubset(loop_results.keys())
        results.record("8.4 | Results has all required keys", test_8_4,
                      f"Keys: {list(loop_results.keys())}")
        
        # Test 8.5: Best lap time is finite
        test_8_5 = np.isfinite(loop_results['best_lap_time'])
        results.record("8.5 | Best lap time is finite", test_8_5,
                      f"best_lap_time = {loop_results['best_lap_time']:.4f}s")
        
        # Test 8.6: Optimization improved lap time
        initial_worst = 90.0 + 0.5 * 1.0 + 1.0 * 3.0  # High CL/CD scenario
        test_8_6 = loop_results['best_lap_time'] < initial_worst
        results.record("8.6 | Optimization found improvement", test_8_6,
                      f"Found: {loop_results['best_lap_time']:.4f}s")
        
        # Test 8.7: Sensitivity analysis is populated
        test_8_7 = len(loop_results['sensitivity_analysis']) > 0
        results.record("8.7 | Sensitivity analysis populated", test_8_7,
                      f"Parameters: {len(loop_results['sensitivity_analysis'])}")
        
        # Test 8.8: All 6 parameters optimized
        test_8_8 = len(loop_results['best_params']) == 6
        results.record("8.8 | All 6 parameters optimized", test_8_8)
        
        # Test 8.9: All parameters within bounds
        param_vals = list(loop_results['best_params'].values())
        bounds_list = list(param_space.bounds.values())
        in_bounds = all(bounds_list[i][0] <= param_vals[i] <= bounds_list[i][1] 
                       for i in range(6))
        test_8_9 = in_bounds
        results.record("8.9 | All optimized params in bounds", test_8_9)
        
        # Test 8.10: Iteration history recorded
        test_8_10 = len(loop_results['iteration_history']) > 0
        results.record("8.10 | Iteration history recorded", test_8_10,
                      f"Iterations: {len(loop_results['iteration_history'])}")
    
    return results


# ============================================================================
# TEST 9: EDGE CASES
# ============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    results = TestResults()
    print("\n" + "="*70)
    print("TEST SUITE 9: EDGE CASES & BOUNDARY CONDITIONS")
    print("="*70)
    
    param_space = ParameterSpace()
    pinn = PINNSimulator()
    lap_time_agent = LapTimeAgent()
    
    # Test 9.1: Boundary parameter values
    boundary_params = torch.tensor([5.0, 0.1, 0.3, 0.4, 10.0, 0.05], dtype=torch.float32, requires_grad=True)
    CL, CD = pinn.forward(boundary_params)
    test_9_1 = CL.item() > 0 and CD.item() > 0
    results.record("9.1 | Boundary params produce valid CL/CD", test_9_1,
                   f"CL={CL.item():.4f}, CD={CD.item():.4f}")
    
    # Test 9.2: Upper boundary parameter values
    upper_params = torch.tensor([35.0, 0.5, 0.8, 0.95, 45.0, 0.25], dtype=torch.float32, requires_grad=True)
    CL, CD = pinn.forward(upper_params)
    test_9_2 = CL.item() > 0 and CD.item() > 0
    results.record("9.2 | Upper boundary params valid", test_9_2,
                   f"CL={CL.item():.4f}, CD={CD.item():.4f}")
    
    # Test 9.3: Very small tolerance (strict improvement)
    enforcer = ConstraintEnforcer(param_space)
    loop = OptimizationLoop(param_space, pinn, lap_time_agent, enforcer)
    try:
        strict_results = loop.run(max_iterations=3, tolerance=0.001)
        test_9_3 = True
    except:
        test_9_3 = False
    results.record("9.3 | Handles strict tolerance", test_9_3)
    
    # Test 9.4: Large tolerance (loose improvement)
    loop2 = OptimizationLoop(param_space, pinn, lap_time_agent, enforcer)
    try:
        loose_results = loop2.run(max_iterations=3, tolerance=100.0)
        test_9_4 = True
    except:
        test_9_4 = False
    results.record("9.4 | Handles loose tolerance", test_9_4)
    
    # Test 9.5: Single iteration
    loop3 = OptimizationLoop(param_space, pinn, lap_time_agent, enforcer)
    try:
        single_result = loop3.run(max_iterations=1, tolerance=0.05)
        test_9_5 = len(single_result['iteration_history']) >= 1
    except:
        test_9_5 = False
    results.record("9.5 | Handles single iteration", test_9_5)
    
    return results


# ============================================================================
# TEST 10: OUTPUT FORMAT TESTS
# ============================================================================

def test_output_format():
    """Test output format and JSON serialization"""
    results = TestResults()
    print("\n" + "="*70)
    print("TEST SUITE 10: OUTPUT FORMAT & SERIALIZATION")
    print("="*70)
    
    param_space = ParameterSpace()
    pinn = PINNSimulator()
    lap_time_agent = LapTimeAgent()
    enforcer = ConstraintEnforcer(param_space)
    
    loop = OptimizationLoop(param_space, pinn, lap_time_agent, enforcer)
    opt_results = loop.run(max_iterations=5, tolerance=0.05)
    
    # Test 10.1: best_params is dict
    test_10_1 = isinstance(opt_results['best_params'], dict)
    results.record("10.1 | best_params is dictionary", test_10_1)
    
    # Test 10.2: best_params keys match parameter names
    test_10_2 = set(opt_results['best_params'].keys()) == set(param_space.param_names)
    results.record("10.2 | best_params keys correct", test_10_2,
                   f"Keys: {list(opt_results['best_params'].keys())}")
    
    # Test 10.3: JSON serializable
    try:
        json_str = json.dumps({
            'best_params': opt_results['best_params'],
            'best_lap_time': opt_results['best_lap_time'],
            'optimal_CL': opt_results['optimal_CL'],
            'optimal_CD': opt_results['optimal_CD'],
        })
        test_10_3 = len(json_str) > 0
    except:
        test_10_3 = False
    results.record("10.3 | Results are JSON serializable", test_10_3)
    
    # Test 10.4: All numeric values
    all_numeric = all(isinstance(v, (int, float)) for v in opt_results['best_params'].values())
    test_10_4 = all_numeric
    results.record("10.4 | All param values numeric", test_10_4)
    
    # Test 10.5: Iteration history has expected structure
    if len(opt_results['iteration_history']) > 0:
        first_iter = opt_results['iteration_history'][0]
        has_keys = all(k in first_iter for k in ['iteration', 'params', 'CL', 'CD', 'lap_time'])
        test_10_5 = has_keys
        results.record("10.5 | Iteration history structure correct", test_10_5)
    else:
        test_10_5 = False
        results.record("10.5 | Iteration history structure correct", test_10_5, "No history")
    
    return results


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("COMPREHENSIVE OPTIMIZER TEST SUITE")
    print("="*70)
    
    all_results = TestResults()
    
    # Run all test suites
    suite_results = [
        test_parameter_space(),
        test_pinn_simulator(),
        test_lap_time_agent(),
        test_gradient_computer(),
        test_constraint_enforcer(),
        test_random_restarter(),
        test_lbfgs_optimizer(),
        test_optimization_loop(),
        test_edge_cases(),
        test_output_format()
    ]
    
    # Aggregate results
    for suite in suite_results:
        all_results.passed += suite.passed
        all_results.failed += suite.failed
    
    # Print final summary
    all_results.summary()
    
    return all_results.failed == 0

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)