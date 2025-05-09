"""
Run all PyTorch neural network classification experiments
"""

import time
import subprocess
import sys

def run_experiment(script_name):
    """Run a single experiment script and measure execution time."""
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True,
            capture_output=False
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"{script_name} completed successfully in {execution_time:.2f} seconds")
        print(f"{'='*80}\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"Error running {script_name}: {e}")
        print(f"{'='*80}\n")
        
        return False

def main():
    """Run all experiments."""
    experiments = [
        "binary_classification.py",
        "regression_validation.py",
        "multiclass_classification.py"
    ]
    
    print("\nPyTorch Neural Network Classification Experiments")
    print("================================================\n")
    print("This script will run all experiments in sequence.")
    
    results = {}
    for exp in experiments:
        success = run_experiment(exp)
        results[exp] = success
    
    # Print summary
    print("\nExecution Summary:")
    print("=================")
    for exp, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{exp}: {status}")
    
    # Check if all experiments succeeded
    if all(results.values()):
        print("\nAll experiments completed successfully!")
    else:
        print("\nSome experiments failed. Check the output above for details.")

if __name__ == "__main__":
    main()
