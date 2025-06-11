#!/usr/bin/env python3
"""
Simple WGF-GMM Runner
Works with the existing experimental framework.
"""

import jax
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Import existing framework
from src.problems.toy import Banana, Multimodal, XShape
from src.utils import make_step_and_carry, config_to_parameters, parse_config
from src.trainers.trainer import trainer


def run_wgf_gmm_experiment(config_name='wgf_gmm_moderate', 
                          problem_name='banana',
                          n_updates=500,
                          seed=42):
    """
    Run a WGF-GMM experiment using the existing framework.
    
    Args:
        config_name: Name of algorithm configuration to use
        problem_name: Name of problem ('banana', 'multimodal', 'xshape')
        n_updates: Number of training updates
        seed: Random seed
    """
    
    # Problem mapping
    PROBLEMS = {
        'banana': Banana,
        'multimodal': Multimodal,
        'xshape': XShape
    }
    
    if problem_name not in PROBLEMS:
        print(f"Error: Unknown problem '{problem_name}'. Available: {list(PROBLEMS.keys())}")
        return None
    
    # Load configuration
    config_path = Path("scripts/wgf_gmm/wgf_gmm_config.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print("Please save the configuration file first.")
        return None
    
    try:
        config = parse_config(config_path)
    except Exception as e:
        print(f"Error parsing config: {e}")
        return None
    
    if config_name not in config:
        print(f"Error: Config '{config_name}' not found. Available: {list(config.keys())}")
        return None
    
    # Setup problem and parameters
    target = PROBLEMS[problem_name]()
    key = jax.random.PRNGKey(seed)
    
    try:
        parameters = config_to_parameters(config, config_name)
        print(f"✓ Loaded parameters for {config_name}")
    except Exception as e:
        print(f"Error creating parameters: {e}")
        return None
    
    # Initialize
    init_key, train_key = jax.random.split(key)
    
    try:
        step, carry = make_step_and_carry(init_key, parameters, target)
        print(f"✓ Created step function and carry for {parameters.algorithm}")
    except Exception as e:
        print(f"Error initializing: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Run training
    print(f"\nRunning {n_updates} updates on {problem_name} with {config_name}...")
    
    try:
        history, final_carry = trainer(
            key=train_key,
            carry=carry,
            target=target,
            ys=None,
            step=step,
            max_epochs=n_updates,
            metrics=None,
            use_jit=True
        )
        
        print(f"✓ Training completed successfully")
        
        # Print results
        losses = history['loss']
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Loss improvement: {losses[0] - losses[-1]:.4f}")
        
        return {
            'config_name': config_name,
            'problem_name': problem_name,
            'history': history,
            'final_carry': final_carry,
            'parameters': parameters
        }
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_algorithms(problem_name='banana', n_updates=500, seed=42):
    """
    Compare different WGF-GMM configurations against standard PVI.
    """
    
    algorithms_to_test = [
        'pvi',
        'wgf_gmm_conservative', 
        'wgf_gmm_moderate',
        'wgf_gmm_aggressive'
    ]
    
    results = {}
    
    for alg in algorithms_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {alg}")
        print(f"{'='*60}")
        
        result = run_wgf_gmm_experiment(
            config_name=alg,
            problem_name=problem_name,
            n_updates=n_updates,
            seed=seed
        )
        
        if result is not None:
            results[alg] = result
        else:
            print(f"✗ {alg} failed")
    
    # Create comparison plot
    if len(results) > 0:
        create_comparison_plot(results, problem_name)
        print_comparison_summary(results)
    
    return results


def create_comparison_plot(results, problem_name):
    """Create a comparison plot of the results."""
    
    plt.figure(figsize=(12, 8))
    
    for alg, result in results.items():
        losses = result['history']['loss']
        # Sample losses for plotting (too many points can be slow)
        if len(losses) > 100:
            indices = np.linspace(0, len(losses)-1, 100, dtype=int)
            plot_losses = [losses[i] for i in indices]
            plot_x = indices
        else:
            plot_losses = losses
            plot_x = range(len(losses))
        
        plt.plot(plot_x, plot_losses, label=alg, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Algorithm Comparison on {problem_name.title()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_dir = Path("output/wgf_gmm_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / f"comparison_{problem_name}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"comparison_{problem_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison plot saved to {output_dir}")


def print_comparison_summary(results):
    """Print a summary of the comparison results."""
    
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    summary_data = []
    
    for alg, result in results.items():
        losses = result['history']['loss']
        times = result['history']['time']
        
        summary_data.append({
            'Algorithm': alg,
            'Initial Loss': f"{losses[0]:.4f}",
            'Final Loss': f"{losses[-1]:.4f}",
            'Improvement': f"{losses[0] - losses[-1]:.4f}",
            'Total Time': f"{times[-1]:.2f}s" if times else "N/A"
        })
    
    # Find best performing algorithm
    best_alg = min(results.keys(), key=lambda alg: results[alg]['history']['loss'][-1])
    
    # Print table
    for data in summary_data:
        print(f"\n{data['Algorithm']}:")
        for key, value in data.items():
            if key != 'Algorithm':
                print(f"  {key}: {value}")
    
    print(f"\n🏆 Best performing algorithm: {best_alg}")
    print(f"   Final loss: {results[best_alg]['history']['loss'][-1]:.4f}")


def quick_test():
    """Run a quick test to verify everything works."""
    
    print("Running quick WGF-GMM test...")
    
    result = run_wgf_gmm_experiment(
        config_name='wgf_gmm_quick_test',
        problem_name='banana',
        n_updates=50,  # Very small for quick test
        seed=42
    )
    
    if result is not None:
        print("✓ Quick test passed!")
        losses = result['history']['loss']
        print(f"Loss went from {losses[0]:.4f} to {losses[-1]:.4f}")
        return True
    else:
        print("✗ Quick test failed!")
        return False


def main():
    """Main function."""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python simple_wgf_gmm_runner.py test                    # Quick test")
        print("  python simple_wgf_gmm_runner.py run [config] [problem] # Run single experiment")
        print("  python simple_wgf_gmm_runner.py compare [problem]      # Compare algorithms")
        print("")
        print("Available configs: pvi, wgf_gmm_conservative, wgf_gmm_moderate, wgf_gmm_aggressive")
        print("Available problems: banana, multimodal, xshape")
        return
    
    command = sys.argv[1]
    
    if command == 'test':
        success = quick_test()
        sys.exit(0 if success else 1)
    
    elif command == 'run':
        config_name = sys.argv[2] if len(sys.argv) > 2 else 'wgf_gmm_moderate'
        problem_name = sys.argv[3] if len(sys.argv) > 3 else 'banana'
        
        result = run_wgf_gmm_experiment(
            config_name=config_name,
            problem_name=problem_name,
            n_updates=1000,
            seed=42
        )
        
        if result is not None:
            print("✓ Experiment completed successfully!")
        else:
            print("✗ Experiment failed!")
            sys.exit(1)
    
    elif command == 'compare':
        problem_name = sys.argv[2] if len(sys.argv) > 2 else 'banana'
        
        results = compare_algorithms(
            problem_name=problem_name,
            n_updates=1000,
            seed=42
        )
        
        if len(results) > 0:
            print("✓ Comparison completed successfully!")
        else:
            print("✗ Comparison failed!")
            sys.exit(1)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
