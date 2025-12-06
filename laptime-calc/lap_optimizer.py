#!/usr/bin/env python3
"""
F1 AeroFlow - Interactive Lap Time Calculator
Original optimization logic preserved exactly
Interactive input with validation ranges
Comprehensive JSON output with detailed console formatting
"""

import json
import os
import sys
from typing import Dict, List, Tuple
from datetime import datetime


class LapOptimizer:
    """Calculates lap time based on aerodynamic parameters"""
    
    def __init__(self):
        """Initialize optimizer with circuit data"""
        self.circuit_data_all = self._load_circuit_data()
        self.baseline_cd = 0.064
        self.baseline_cl = 0.70
        
        # CD and CL validation ranges
        self.cd_min, self.cd_max = 0.040, 0.100
        self.cl_min, self.cl_max = 0.40, 1.30
        self.ratio_min, self.ratio_max = 6.0, 16.0
    
    def _load_circuit_data(self) -> Dict:
        """Load circuit data from JSON file"""
        json_file = "F1_2024_COMPLETE_CIRCUIT_DATA.json"
        
        if not os.path.exists(json_file):
            print(f"ERROR: {json_file} not found")
            sys.exit(1)
        
        with open(json_file, 'r') as f:
            return json.load(f)
    
    def get_available_circuits(self) -> List[str]:
        """Get list of available circuits"""
        return sorted(self.circuit_data_all.keys())
    
    def validate_parameters(self, cd: float, cl: float) -> Tuple[bool, List[str]]:
        """Validate CD and CL are in realistic ranges"""
        errors = []
        
        if not (self.cd_min <= cd <= self.cd_max):
            errors.append(f"CD {cd} out of range [{self.cd_min}, {self.cd_max}]")
        
        if not (self.cl_min <= cl <= self.cl_max):
            errors.append(f"CL {cl} out of range [{self.cl_min}, {self.cl_max}]")
        
        # Check CL/CD ratio (F1 aerodynamic law)
        if cd > 0:
            ratio = cl / cd
            if not (self.ratio_min <= ratio <= self.ratio_max):
                errors.append(f"CL/CD ratio {ratio:.2f} unrealistic (expect {self.ratio_min}-{self.ratio_max} for F1)")
        
        return len(errors) == 0, errors
    
    def adjust_corner_time(self, baseline_time: float,
                          apex_speed: float, cl_delta: float) -> float:
        """
        Adjust corner time based on CL change
        Higher CL → more grip → faster apex speed → shorter time
        
        Relationship: Δv ≈ ΔCL * 5 km/h
        Time reduction from apex speed increase: t ∝ 1/v
        """
        if cl_delta == 0:
            return baseline_time
        
        # Speed gain from downforce increase (km/h)
        speed_delta = cl_delta * 5.0
        
        # New apex speed
        new_apex_speed = apex_speed + speed_delta
        new_apex_speed = max(new_apex_speed, apex_speed * 0.90)  # Floor at 90%
        
        # Time reduction proportional to speed increase
        if apex_speed > 0:
            speed_ratio = new_apex_speed / apex_speed
            adjusted_time = baseline_time / speed_ratio
        else:
            adjusted_time = baseline_time
        
        return adjusted_time
    
    def adjust_straight_time(self, baseline_time: float,
                            exit_speed: float, cd_delta: float) -> float:
        """
        Adjust straight time based on CD change
        Higher CD → more drag → lower top speed → longer time
        
        Relationship: v_max ∝ (1/CD)^(1/3)
        dv/v ≈ (-1/3) * (dCD/CD)
        """
        if cd_delta == 0:
            return baseline_time
        
        # Speed loss from drag increase
        # Roughly: -1% speed per 0.01 CD increase
        speed_delta = -exit_speed * (cd_delta / self.baseline_cd) / 3.0
        
        # New top speed
        new_top_speed = exit_speed + speed_delta
        new_top_speed = max(new_top_speed, exit_speed * 0.95)  # Floor at 95%
        
        # Time increase from speed reduction
        if exit_speed > 0:
            speed_ratio = new_top_speed / exit_speed
            adjusted_time = baseline_time / speed_ratio
        else:
            adjusted_time = baseline_time
        
        return adjusted_time
    
    def calculate_lap_time(self, circuit: str, cd: float, cl: float) -> Dict:
        """
        Calculate complete lap time with all aerodynamic adjustments
        Original logic preserved exactly
        """
        
        # Validate inputs
        valid, errors = self.validate_parameters(cd, cl)
        if not valid:
            print("\nPARAMETER VALIDATION ERRORS:")
            for error in errors:
                print(f"  • {error}")
            return None
        
        # Get circuit data
        circuit = circuit.lower()
        if circuit not in self.circuit_data_all:
            print(f"\nERROR: Circuit '{circuit}' not found")
            return None
        
        circuit_data = self.circuit_data_all[circuit]
        pole_time = circuit_data["pole_time"]
        pole_driver = circuit_data.get("pole_driver", "Unknown")
        corners = circuit_data["corners"]
        straights = circuit_data["straights"]
        
        # Calculate deltas
        cl_delta = cl - self.baseline_cl
        cd_delta = cd - self.baseline_cd
        
        # Process all corners
        baseline_corner_time_total = 0.0
        adjusted_corner_time_total = 0.0
        corner_results = []
        
        for corner in corners:
            baseline_corner_time = corner["time_seconds"]
            adjusted_corner_time = self.adjust_corner_time(
                baseline_corner_time,
                corner["apex_speed_kmh"],
                cl_delta
            )
            
            time_delta = adjusted_corner_time - baseline_corner_time
            corner_results.append({
                "name": corner["name"],
                "baseline": round(baseline_corner_time, 3),
                "adjusted": round(adjusted_corner_time, 3),
                "delta": round(time_delta, 3)
            })
            
            baseline_corner_time_total += baseline_corner_time
            adjusted_corner_time_total += adjusted_corner_time
        
        # Process all straights
        baseline_straight_time_total = 0.0
        adjusted_straight_time_total = 0.0
        straight_results = []
        
        for straight in straights:
            baseline_straight_time = straight["time_seconds"]
            adjusted_straight_time = self.adjust_straight_time(
                baseline_straight_time,
                straight["exit_speed_kmh"],
                cd_delta
            )
            
            time_delta = adjusted_straight_time - baseline_straight_time
            straight_results.append({
                "name": straight["name"],
                "baseline": round(baseline_straight_time, 3),
                "adjusted": round(adjusted_straight_time, 3),
                "delta": round(time_delta, 3)
            })
            
            baseline_straight_time_total += baseline_straight_time
            adjusted_straight_time_total += adjusted_straight_time
        
        # Calculate totals
        baseline_calculated_lap = baseline_corner_time_total + baseline_straight_time_total
        calculated_lap_time = adjusted_corner_time_total + adjusted_straight_time_total
        
        # Calculate improvement relative to baseline segments
        improvement = baseline_calculated_lap - calculated_lap_time
        improvement_percent = (improvement / baseline_calculated_lap) * 100
        
        # Status assessment
        if improvement > 0.5:
            status = "✓ GOOD - significant improvement"
        elif improvement > 0.2:
            status = "✓ MODERATE - good improvement"
        elif improvement > 0.0:
            status = "✓ VALID - modest improvement"
        elif improvement > -0.1:
            status = "✓ VALID - minimal change"
        elif improvement > -0.3:
            status = "⚠ TRADE-OFF - small loss (higher drag outweighs corner gain)"
        else:
            status = "✗ LOSS - setup worsens lap time"
        
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "circuit": circuit.upper(),
                "pole_driver": pole_driver,
                "pole_time": round(pole_time, 3)
            },
            "aero_config": {
                "cd": cd,
                "cl": cl,
                "cd_delta": round(cd - self.baseline_cd, 4),
                "cl_delta": round(cl - self.baseline_cl, 4),
                "ratio": round(cl / cd if cd > 0 else 0, 2)
            },
            "baseline_aero": {
                "cd": self.baseline_cd,
                "cl": self.baseline_cl,
                "ratio": round(self.baseline_cl / self.baseline_cd, 2)
            },
            "lap_analysis": {
                "baseline_calculated_lap": round(baseline_calculated_lap, 3),
                "calculated_lap_time": round(calculated_lap_time, 3),
                "improvement_seconds": round(improvement, 3),
                "improvement_percent": round(improvement_percent, 2),
                "status": status
            },
            "corner_analysis": {
                "total_corners": len(corners),
                "baseline_time": round(baseline_corner_time_total, 3),
                "adjusted_time": round(adjusted_corner_time_total, 3),
                "total_delta": round(adjusted_corner_time_total - baseline_corner_time_total, 3),
                "corners": corner_results
            },
            "straight_analysis": {
                "total_straights": len(straights),
                "baseline_time": round(baseline_straight_time_total, 3),
                "adjusted_time": round(adjusted_straight_time_total, 3),
                "total_delta": round(adjusted_straight_time_total - baseline_straight_time_total, 3),
                "straights": straight_results
            }
        }
    
    def print_results(self, results: Dict):
        """Pretty-print results in original format"""
        
        print("\n" + "="*80)
        print(f"  F1 LAP TIME CALCULATION: {results['metadata']['circuit']}")
        print("="*80)
        
        print(f"\nCIRCUIT:")
        print(f"  Pole Time (from telemetry): {results['metadata']['pole_time']:.3f}s ({results['metadata']['pole_driver']})")
        print(f"  (Note: Segment times may not sum to pole due to data granularity)")
        
        aero = results['aero_config']
        baseline = results['baseline_aero']
        print(f"\nAERODYNAMIC SETUP:")
        print(f"  CD (Drag):      {aero['cd']:.4f}  (baseline: {baseline['cd']:.4f})  [{aero['cd_delta']:+.4f}]")
        print(f"  CL (Downforce): {aero['cl']:.4f}  (baseline: {baseline['cl']:.4f})  [{aero['cl_delta']:+.4f}]")
        print(f"  CL/CD Ratio:    {aero['ratio']:.2f}  (F1 range: {self.ratio_min}-{self.ratio_max})")
        
        # Corner analysis
        corner_data = results['corner_analysis']
        print(f"\nCORNER ANALYSIS ({corner_data['total_corners']} corners):")
        print(f"  {'Corner':<30} {'Baseline':>10} {'Adjusted':>10} {'Change':>10}")
        print(f"  {'-'*60}")
        
        for corner in corner_data['corners']:
            print(f"  {corner['name']:<30} {corner['baseline']:>10.3f}s {corner['adjusted']:>10.3f}s {corner['delta']:>+10.3f}s")
        
        print(f"  {'-'*60}")
        print(f"  {'TOTAL':<30} {corner_data['baseline_time']:>10.3f}s {corner_data['adjusted_time']:>10.3f}s {corner_data['total_delta']:>+10.3f}s")
        
        # Straight analysis
        straight_data = results['straight_analysis']
        print(f"\nSTRAIGHT ANALYSIS ({straight_data['total_straights']} straights):")
        print(f"  {'Straight':<30} {'Baseline':>10} {'Adjusted':>10} {'Change':>10}")
        print(f"  {'-'*60}")
        
        for straight in straight_data['straights']:
            print(f"  {straight['name']:<30} {straight['baseline']:>10.3f}s {straight['adjusted']:>10.3f}s {straight['delta']:>+10.3f}s")
        
        print(f"  {'-'*60}")
        print(f"  {'TOTAL':<30} {straight_data['baseline_time']:>10.3f}s {straight_data['adjusted_time']:>10.3f}s {straight_data['total_delta']:>+10.3f}s")
        
        # Results
        lap = results['lap_analysis']
        print(f"\nRESULTS:")
        print(f"  Baseline Lap Time:        {lap['baseline_calculated_lap']:.3f}s (sum of segments)")
        print(f"  Calculated Lap Time:      {lap['calculated_lap_time']:.3f}s (with CD/CL adjustments)")
        print(f"  {'─'*50}")
        print(f"  Improvement:              {lap['improvement_seconds']:>+.3f}s ({lap['improvement_percent']:+.2f}%)")
        print(f"  Status:                   {lap['status']}")
        print("\n" + "="*80 + "\n")
    
    def save_results(self, results: Dict, circuit: str) -> str:
        """Save results to JSON file"""
        os.makedirs("results", exist_ok=True)
        
        filename = f"results/{circuit.lower()}_cd{results['aero_config']['cd']:.3f}_cl{results['aero_config']['cl']:.2f}.json"
        filename = filename.replace(".", "")
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filename


def get_circuit_input(optimizer: LapOptimizer) -> str:
    """Get circuit name from user with validation"""
    available = optimizer.get_available_circuits()
    
    while True:
        print(f"\nAvailable circuits: {', '.join(available)}")
        circuit = input("Enter circuit name: ").strip().lower()
        
        if circuit in available:
            return circuit
        else:
            print(f"ERROR: '{circuit}' not found. Please enter a valid circuit name.")


def get_cd_input(optimizer: LapOptimizer) -> float:
    """Get CD value from user with validation"""
    cd_min = optimizer.cd_min
    cd_max = optimizer.cd_max
    
    while True:
        try:
            cd_str = input(f"Enter CD (Drag coefficient) [{cd_min}-{cd_max}]: ").strip()
            cd = float(cd_str)
            
            if cd_min <= cd <= cd_max:
                return cd
            else:
                print(f"ERROR: CD must be between {cd_min} and {cd_max}")
        except ValueError:
            print("ERROR: Please enter a valid number")


def get_cl_input(optimizer: LapOptimizer) -> float:
    """Get CL value from user with validation"""
    cl_min = optimizer.cl_min
    cl_max = optimizer.cl_max
    
    while True:
        try:
            cl_str = input(f"Enter CL (Downforce coefficient) [{cl_min}-{cl_max}]: ").strip()
            cl = float(cl_str)
            
            if cl_min <= cl <= cl_max:
                return cl
            else:
                print(f"ERROR: CL must be between {cl_min} and {cl_max}")
        except ValueError:
            print("ERROR: Please enter a valid number")


def main():
    """Main interactive loop"""
    
    print("\n" + "="*80)
    print("  F1 AEROFLOW - INTERACTIVE LAP TIME CALCULATOR")
    print("="*80)
    
    optimizer = LapOptimizer()
    
    while True:
        try:
            # Get user inputs
            circuit = get_circuit_input(optimizer)
            cd = get_cd_input(optimizer)
            cl = get_cl_input(optimizer)
            
            # Calculate
            results = optimizer.calculate_lap_time(circuit, cd, cl)
            
            if results:
                # Display results
                optimizer.print_results(results)
                
                # Save to JSON
                output_file = optimizer.save_results(results, circuit)
                print(f"✓ Results saved to: {output_file}\n")
            
            # Ask if they want to continue
            again = input("Calculate another lap time? (y/n): ").strip().lower()
            if again != 'y':
                print("\nThank you for using F1 AeroFlow! 🏁\n")
                break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()