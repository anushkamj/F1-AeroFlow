# F1 Rear Wing Optimization Loop
# Ridwan's Optimizer Agent for SpoonOS Hackathon

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import json

# ============================================================================
# STEP 1: PARAMETER SPACE & BOUNDS
# ============================================================================

class ParameterSpace:
    """Define the 6D parameter space for the rear wing"""
    
    def __init__(self):
        # Parameter bounds (EXACT parameters from project spec)
        self.bounds = {
            'alpha_norm': (5.0, 35.0),           # Normalized Angle of Attack (degrees)
            'g_ratio': (0.1, 0.5),               # Spanwise Ratio
            'c_main_norm': (0.3, 0.8),           # Spanwise Component Chord Fraction
            'lambda': (0.4, 0.95),               # Flap Chord Ratio
            'theta_flap': (10.0, 45.0),          # Flap Camber Angle (degrees)
            'tau': (0.05, 0.25)                  # Spanwise Taper Ratio
        }
        
        self.param_names = list(self.bounds.keys())
        self.n_params = len(self.param_names)
    
    def get_bounds_tensor(self):
        """Return lower and upper bounds as tensors"""
        lower = torch.tensor([self.bounds[p][0] for p in self.param_names], dtype=torch.float32)
        upper = torch.tensor([self.bounds[p][1] for p in self.param_names], dtype=torch.float32)
        return lower, upper
    
    def clip_to_bounds(self, params: torch.Tensor) -> torch.Tensor:
        """Enforce constraint: clip parameters to valid ranges"""
        lower, upper = self.get_bounds_tensor()
        return torch.clamp(params, lower, upper)
    
    def sample_random_valid(self) -> torch.Tensor:
        """Sample a random valid parameter vector"""
        params = []
        for param_name in self.param_names:
            low, high = self.bounds[param_name]
            params.append(np.random.uniform(low, high))
        return torch.tensor(params, dtype=torch.float32, requires_grad=True)


# ============================================================================
# STEP 2: MOCK PINN & LAP TIME AGENT (Replace with real versions)
# ============================================================================

class PINNSimulator:
    """Mock PINN: parameters -> CL, CD"""
    
    def __init__(self):
        # In real hackathon, this is Anushka's trained neural network
        self.trained = True
    
    def forward(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mock PINN: takes parameters, returns CL and CD
        In reality, this calls Anushka's trained model
        
        Parameters:
        - params[0]: alpha_norm (Normalized Angle of Attack)
        - params[1]: g_ratio (Spanwise Ratio)
        - params[2]: c_main_norm (Spanwise Component Chord Fraction)
        - params[3]: lambda (Flap Chord Ratio)
        - params[4]: theta_flap (Flap Camber Angle)
        - params[5]: tau (Spanwise Taper Ratio)
        """
        # Simple mock: CL and CD depend on these 6 parameters
        alpha_norm = params[0]
        g_ratio = params[1]
        c_main_norm = params[2]
        lambda_param = params[3]
        theta_flap = params[4]
        tau = params[5]
        
        # Mock physics relationships
        # Higher angle -> more downforce, more drag
        # Higher flap angle -> more downforce, more drag
        # Chord and ratio parameters affect pressure distribution
        CL = 2.0 + 0.05 * alpha_norm + 0.3 * g_ratio - 0.2 * c_main_norm + 0.03 * theta_flap + 0.1 * lambda_param - 0.05 * tau
        CD = 0.8 + 0.01 * alpha_norm + 0.15 * g_ratio + 0.1 * c_main_norm + 0.015 * theta_flap + 0.08 * lambda_param + 0.03 * tau
        
        return CL, CD


class LapTimeAgent:
    """Mock Lap Time Agent: CL, CD -> lap_time"""
    
    def __init__(self, track_name: str = 'silverstone'):
        self.track_name = track_name
        
        # Track parameters (simplified)
        self.tracks = {
            'silverstone': {
                'total_length': 5891,  # meters
                'corners': [
                    {'radius': 150, 'speed_limit': 200},  # High speed
                    {'radius': 80, 'speed_limit': 120},   # Medium speed
                    {'radius': 50, 'speed_limit': 80}     # Low speed
                ]
            }
        }
    
    def compute_lap_time(self, CL: torch.Tensor, CD: torch.Tensor) -> torch.Tensor:
        """
        Mock lap time calculation: CL, CD -> lap_time (in seconds)
        
        Formula (simplified):
        - Higher CL (downforce) -> better cornering -> faster through corners
        - Higher CD (drag) -> slower on straights
        - Lap time = f(CL, CD)
        
        Real version uses full vehicle dynamics simulation
        """
        # Simplistic model: lap time depends on downforce and drag trade-off
        # Lower is better
        
        # Baseline lap time
        baseline_lap_time = 90.0  # seconds (realistic for Silverstone)
        
        # Downforce benefit: more downforce = faster corners
        downforce_benefit = -0.5 * CL  # negative = improvement
        
        # Drag penalty: more drag = slower straights
        drag_penalty = 1.0 * CD
        
        # Total lap time
        lap_time = baseline_lap_time + downforce_benefit + drag_penalty
        
        return lap_time


# ============================================================================
# STEP 3: GRADIENT COMPUTATION (Backprop)
# ============================================================================

class GradientComputer:
    """Compute gradient: d(lap_time) / d(params)"""
    
    def __init__(self, pinn: PINNSimulator, lap_time_agent: LapTimeAgent):
        self.pinn = pinn
        self.lap_time_agent = lap_time_agent
    
    def compute_gradient(self, params: torch.Tensor) -> torch.Tensor:
        """
        Backprop through: params -> PINN -> CL, CD -> Lap Time -> lap_time
        Returns: d(lap_time) / d(params)
        """
        # Ensure params require gradients
        params = params.clone().detach().requires_grad_(True)
        
        # Forward pass through PINN
        CL, CD = self.pinn.forward(params)
        
        # Forward pass through Lap Time Agent
        lap_time = self.lap_time_agent.compute_lap_time(CL, CD)
        
        # Backward pass: compute gradient
        lap_time.backward()
        
        gradient = params.grad
        return gradient


# ============================================================================
# STEP 4: L-BFGS OPTIMIZER
# ============================================================================

class LBFGSOptimizer:
    """Wrapper around PyTorch's L-BFGS optimizer"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
    
    def optimize_step(self, params: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """
        Take one optimization step using L-BFGS logic
        (simplified: just gradient descent with line search)
        """
        # Simple gradient descent with adaptive step size
        step_size = self.lr
        updated_params = params - step_size * gradient
        
        return updated_params


# ============================================================================
# STEP 5: CONSTRAINT ENFORCEMENT (Referee)
# ============================================================================

class ConstraintEnforcer:
    """Enforce F1 regulatory constraints"""
    
    def __init__(self, param_space: ParameterSpace):
        self.param_space = param_space
    
    def enforce(self, params: torch.Tensor) -> torch.Tensor:
        """
        Check and clip parameters to valid ranges
        (Referee function)
        """
        return self.param_space.clip_to_bounds(params)
    
    def is_valid(self, params: torch.Tensor) -> bool:
        """Check if design is legal"""
        lower, upper = self.param_space.get_bounds_tensor()
        return torch.all(params >= lower) and torch.all(params <= upper)


# ============================================================================
# STEP 6: RANDOM RESTART LOGIC
# ============================================================================

class RandomRestarter:
    """Handle random restarts when optimizer gets stuck"""
    
    def __init__(self, param_space: ParameterSpace):
        self.param_space = param_space
        self.stall_threshold = 15  # iterations without improvement
    
    def should_restart(self, stall_counter: int) -> bool:
        """Check if we should reset and try a new random initialization"""
        return stall_counter >= self.stall_threshold
    
    def restart(self) -> torch.Tensor:
        """Sample a new random valid parameter vector"""
        return self.param_space.sample_random_valid()


# ============================================================================
# STEP 7: MAIN OPTIMIZATION LOOP
# ============================================================================

class OptimizationLoop:
    """Main optimization engine: orchestrates the entire feedback loop"""
    
    def __init__(self, 
                 param_space: ParameterSpace,
                 pinn: PINNSimulator,
                 lap_time_agent: LapTimeAgent,
                 constraint_enforcer: ConstraintEnforcer):
        
        self.param_space = param_space
        self.pinn = pinn
        self.lap_time_agent = lap_time_agent
        self.constraint_enforcer = constraint_enforcer
        
        self.gradient_computer = GradientComputer(pinn, lap_time_agent)
        self.optimizer = LBFGSOptimizer(learning_rate=0.01)
        self.restarter = RandomRestarter(param_space)
        
        # History tracking
        self.iteration_history = []
        self.best_params = None
        self.best_lap_time = float('inf')
        
    def run(self, max_iterations: int = 50, tolerance: float = 0.05) -> Dict:
        """
        Main optimization loop
        
        Args:
            max_iterations: Max number of iterations
            tolerance: Minimum improvement to count as success (seconds)
        
        Returns:
            Dictionary with results
        """
        
        # Initialize with random valid parameters
        current_params = self.param_space.sample_random_valid()
        stall_counter = 0
        
        print("\n" + "="*70)
        print("STARTING OPTIMIZATION LOOP")
        print("="*70 + "\n")
        
        for iteration in range(max_iterations):
            # STEP 1: Evaluate current parameters
            with torch.no_grad():
                CL, CD = self.pinn.forward(current_params)
                lap_time = self.lap_time_agent.compute_lap_time(CL, CD)
            
            lap_time_scalar = lap_time.item()
            
            # STEP 2: Check for improvement
            improvement = self.best_lap_time - lap_time_scalar
            
            if improvement > tolerance:
                # Good improvement!
                self.best_params = current_params.clone().detach()
                self.best_lap_time = lap_time_scalar
                stall_counter = 0
                status = "✓ IMPROVED"
            else:
                stall_counter += 1
                status = f"✗ STALL ({stall_counter})"
            
            # STEP 3: Log iteration
            self.iteration_history.append({
                'iteration': iteration,
                'params': current_params.detach().numpy().tolist(),
                'CL': CL.item(),
                'CD': CD.item(),
                'lap_time': lap_time_scalar,
                'improvement': improvement,
                'best_lap_time': self.best_lap_time
            })
            
            # Print progress
            print(f"Iter {iteration:3d} | Lap Time: {lap_time_scalar:7.3f}s | "
                  f"Δ: {improvement:+7.3f}s | {status}")
            
            # STEP 4: Check termination
            if self.restarter.should_restart(stall_counter):
                print(f"\n  → Stalled for {stall_counter} iterations. RESTARTING...\n")
                current_params = self.restarter.restart()
                stall_counter = 0
                continue
            
            # STEP 5: Compute gradient
            gradient = self.gradient_computer.compute_gradient(current_params)
            
            # STEP 6: Optimization step (L-BFGS)
            current_params = self.optimizer.optimize_step(current_params, gradient)
            
            # STEP 7: Enforce constraints (Referee)
            current_params = self.constraint_enforcer.enforce(current_params)
        
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70 + "\n")
        
        # Return results
        return self._compile_results()
    
    def _compile_results(self) -> Dict:
        """Compile final results and diagnostics"""
        
        # Compute sensitivity analysis
        sensitivity = self._compute_sensitivity()
        
        # Get optimal CL, CD
        with torch.no_grad():
            optimal_CL, optimal_CD = self.pinn.forward(self.best_params)
        
        results = {
            'best_params': {
                name: float(value)
                for name, value in zip(self.param_space.param_names, self.best_params.numpy())
            },
            'best_lap_time': float(self.best_lap_time),
            'optimal_CL': float(optimal_CL),
            'optimal_CD': float(optimal_CD),
            'sensitivity_analysis': sensitivity,
            'iteration_history': self.iteration_history,
            'total_iterations': len(self.iteration_history),
        }
        
        return results
    
    def _compute_sensitivity(self) -> Dict:
        """Compute parameter sensitivity: which parameters matter most?"""
        
        sensitivity = {}
        base_lap_time = self.best_lap_time
        
        for i, param_name in enumerate(self.param_space.param_names):
            # Perturb parameter by 1%
            perturbed = self.best_params.clone()
            original_value = perturbed[i].item()
            perturbation = original_value * 0.01
            perturbed[i] += perturbation
            
            # Evaluate
            with torch.no_grad():
                CL, CD = self.pinn.forward(perturbed)
                new_lap_time = self.lap_time_agent.compute_lap_time(CL, CD).item()
            
            # Sensitivity = change in lap time / change in parameter
            sensitivity[param_name] = (new_lap_time - base_lap_time) / perturbation
        
        return sensitivity


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Run the complete optimization"""
    
    # Initialize components
    param_space = ParameterSpace()
    pinn = PINNSimulator()
    lap_time_agent = LapTimeAgent(track_name='silverstone')
    constraint_enforcer = ConstraintEnforcer(param_space)
    
    # Create and run optimization loop
    optimizer = OptimizationLoop(
        param_space=param_space,
        pinn=pinn,
        lap_time_agent=lap_time_agent,
        constraint_enforcer=constraint_enforcer
    )
    
    results = optimizer.run(max_iterations=50, tolerance=0.05)
    
    # Display results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nBest Lap Time: {results['best_lap_time']:.4f} seconds")
    print(f"Optimal CL: {results['optimal_CL']:.4f}")
    print(f"Optimal CD: {results['optimal_CD']:.4f}")
    print(f"Total Iterations: {results['total_iterations']}")
    
    print("\n" + "-"*70)
    print("OPTIMIZED PARAMETERS")
    print("-"*70)
    print(f"  alpha_norm (Normalized Angle of Attack):     {results['best_params']['alpha_norm']:8.4f}°")
    print(f"  g_ratio (Spanwise Ratio):                    {results['best_params']['g_ratio']:8.4f}")
    print(f"  c_main_norm (Spanwise Component Chord):      {results['best_params']['c_main_norm']:8.4f}")
    print(f"  lambda (Flap Chord Ratio):                   {results['best_params']['lambda']:8.4f}")
    print(f"  theta_flap (Flap Camber Angle):              {results['best_params']['theta_flap']:8.4f}°")
    print(f"  tau (Spanwise Taper Ratio):                  {results['best_params']['tau']:8.4f}")
    
    print("\n" + "-"*70)
    print("PARAMETER SENSITIVITY (most influential first)")
    print("-"*70)
    sorted_sensitivity = sorted(
        results['sensitivity_analysis'].items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for param_name, sensitivity_value in sorted_sensitivity:
        print(f"  {param_name:15s}: {sensitivity_value:+8.4f}")
    
    # Save results to JSON
    results_json = {
        'best_params': results['best_params'],
        'best_lap_time': results['best_lap_time'],
        'optimal_CL': results['optimal_CL'],
        'optimal_CD': results['optimal_CD'],
        'sensitivity_analysis': {k: v for k, v in results['sensitivity_analysis'].items()},
    }
    
    with open('optimization_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n✓ Results saved to optimization_results.json\n")
    
    return results


if __name__ == '__main__':
    main()