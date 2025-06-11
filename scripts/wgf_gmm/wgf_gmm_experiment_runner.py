#!/usr/bin/env python3
"""
Fixed WGF-GMM Experiment Runner
Works with the existing framework and available functions.
"""

import jax
import jax.numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
from collections import defaultdict
from itertools import product
import sys

# Import your existing modules
from src.problems.toy import Banana, Multimodal, XShape
from src.utils import make_step_and_carry, config_to_parameters, parse_config
from src.base import Parameters, ModelParameters, ThetaOptParameters, ROptParameters, PIDParameters

# Import the WGF-GMM implementation
try:
    from src.trainers.wgf_gmm import (
        wgf_gmm_pvi_step_with_monitoring,
        WGFGMMMetrics,
        WGFGMMHyperparams,
        wgf_gmm_pvi_step
    )
    WGF_GMM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import WGF-GMM implementation: {e}")
    WGF_GMM_AVAILABLE = False


def test_single_config():
    """Test a single configuration to verify everything works."""
    
    if not WGF_GMM_AVAILABLE:
        print("Error: WGF-GMM implementation not available.")
        return False
    
    print("Testing single WGF-GMM configuration...")
    
    target = Banana()
    key = jax.random.PRNGKey(42)
    
    parameters = Parameters(
        algorithm='wgf_gmm',
        model_parameters=ModelParameters(
            d_z=2,
            use_particles=True,
            n_particles=20,  # Small for testing
            kernel='norm_fixed_var_w_skip',
            n_hidden=128
        ),
        theta_opt_parameters=ThetaOptParameters(
            lr=1e-4,
            optimizer='rmsprop',
            lr_decay=False,
            regularization=1e-8,
            clip=False
        ),
        r_opt_parameters=ROptParameters(
            lr=1e-2,
            regularization=1e-8
        ),
        extra_alg_parameters=PIDParameters(mc_n_samples=50)  # Small for testing
    )
    
    init_key, train_key = jax.random.split(key)
    step, carry = make_step_and_carry(init_key, parameters, target)
    
    # Extract optim from step function  
    optim = None
    if hasattr(step, 'keywords') and 'optim' in step.keywords:
        optim = step.keywords['optim']
    elif hasattr(step, 'func') and hasattr(step.func, 'keywords'):
        optim = step.func.keywords.get('optim')
    
    if optim is None:
        print("Error: Could not extract optimizer from step function")
        return False
    
    print("Running 10 steps...")
    losses = []
    
    for i in range(10):
        train_key, step_key = jax.random.split(train_key)
        
        try:
            lval, carry, metrics = wgf_gmm_pvi_step_with_monitoring(
                step_key, carry, target, None, optim, 
                parameters.extra_alg_parameters,
                lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01
            )
            losses.append(float(lval))
            
            if (i + 1) % 5 == 0:
                print(f"Step {i+1}: Loss = {lval:.4f}")
                print(f"         ELBO = {metrics.elbo:.4f}")
                print(f"         W_dist = {metrics.wasserstein_distance:.4f}")
                
        except Exception as e:
            print(f"Error at step {i+1}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if losses:
        print(f"Test completed successfully!")
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        return True
    else:
        print("Test failed!")
        return False


def run_quick_test():
    """Run a quick test with a subset of hyperparameters."""
    
    if not WGF_GMM_AVAILABLE:
        print("Error: WGF-GMM implementation not available.")
        return None
    
    print("Running quick WGF-GMM test...")
    
    # Reduced hyperparameter space for testing
    LAMBDA_REG_VALUES = [0.1, 0.5]
    LR_MEAN_VALUES = [0.01, 0.05]
    
    # Just test one problem
    target = Banana()
    N_UPDATES = 50  # Very small for quick test
    SEED = 42
    
    output_dir = Path("output/wgf_gmm_quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    key = jax.random.PRNGKey(SEED)
    
    # Create parameters
    parameters = Parameters(
        algorithm='wgf_gmm',
        model_parameters=ModelParameters(
            d_z=2,
            use_particles=True,
            n_particles=20,  # Small for speed
            kernel='norm_fixed_var_w_skip',
            n_hidden=128    # Small for speed
        ),
        theta_opt_parameters=ThetaOptParameters(
            lr=1e-4,
            optimizer='rmsprop',
            lr_decay=False,
            regularization=1e-8,
            clip=False
        ),
        r_opt_parameters=ROptParameters(
            lr=1e-2,
            regularization=1e-8
        ),
        extra_alg_parameters=PIDParameters(mc_n_samples=50)  # Small for speed
    )
    
    # Initialize
    init_key, exp_key = jax.random.split(key)
    step, initial_carry = make_step_and_carry(init_key, parameters, target)
    
    # Extract optim
    optim = None
    if hasattr(step, 'keywords') and 'optim' in step.keywords:
        optim = step.keywords['optim']
    elif hasattr(step, 'func') and hasattr(step.func, 'keywords'):
        optim = step.func.keywords.get('optim')
    
    if optim is None:
        print("Error: Could not extract optimizer")
        return None
    
    results = []
    
    # Test all combinations
    for lambda_reg in LAMBDA_REG_VALUES:
        for lr_mean in LR_MEAN_VALUES:
            
            print(f"Testing: λ={lambda_reg}, lr_mean={lr_mean}")
            
            # Reset for this test
            carry = initial_carry
            exp_key, run_key = jax.random.split(exp_key)
            
            losses = []
            
            # Quick training loop
            for update_idx in range(N_UPDATES):
                run_key, step_key = jax.random.split(run_key)
                
                try:
                    lval, carry, metrics = wgf_gmm_pvi_step_with_monitoring(
                        step_key, carry, target, None, optim, 
                        parameters.extra_alg_parameters,
                        lambda_reg=lambda_reg, lr_mean=lr_mean, 
                        lr_cov=0.001, lr_weight=0.01
                    )
                    losses.append(float(lval))
                except Exception as e:
                    print(f"  Error at step {update_idx}: {e}")
                    break
            
            if losses:
                final_loss = losses[-1]
                print(f"  Final loss: {final_loss:.4f}")
                
                result = {
                    'lambda_reg': lambda_reg,
                    'lr_mean': lr_mean,
                    'final_loss': final_loss,
                    'losses': losses,
                    'success': True
                }
            else:
                print(f"  Failed!")
                result = {
                    'lambda_reg': lambda_reg,
                    'lr_mean': lr_mean,
                    'final_loss': float('inf'),
                    'losses': [],
                    'success': False
                }
            
            results.append(result)
    
    # Save quick test results
    with open(output_dir / "quick_test_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Find best from quick test
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['final_loss'])
        print(f"\nBest quick test result:")
        print(f"Final Loss: {best_result['final_loss']:.4f}")
        print(f"λ={best_result['lambda_reg']}, lr_mean={best_result['lr_mean']}")
    else:
        print("No successful runs in quick test!")
    
    return results


def run_comprehensive_experiments():
    """Run comprehensive experiments with more hyperparameter combinations."""
    
    if not WGF_GMM_AVAILABLE:
        print("Error: WGF-GMM implementation not available.")
        return None, None
    
    # Hyperparameter values to test
    LAMBDA_REG_VALUES = [0.01, 0.1, 0.5]
    LR_MEAN_VALUES = [0.005, 0.01, 0.05]
    LR_COV_VALUES = [0.001, 0.005]
    LR_WEIGHT_VALUES = [0.01, 0.02]
    
    # Problems to test
    PROBLEMS = {
        'banana': Banana,
        'multimodal': Multimodal, 
    }
    
    # Experiment settings
    N_UPDATES = 200
    N_PARTICLES = 50
    SEED = 42
    
    # Output directory
    output_dir = Path("output/wgf_gmm_comprehensive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    
    print("Starting comprehensive WGF-GMM experiments...")
    total_combinations = len(LAMBDA_REG_VALUES) * len(LR_MEAN_VALUES) * len(LR_COV_VALUES) * len(LR_WEIGHT_VALUES)
    print(f"Testing {total_combinations} combinations per problem")
    
    for problem_name, problem_class in PROBLEMS.items():
        print(f"\n{'='*60}")
        print(f"Running experiments for {problem_name.upper()}")
        print(f"{'='*60}")
        
        # Setup problem
        target = problem_class()
        key = jax.random.PRNGKey(SEED)
        
        # Create base parameters
        parameters = Parameters(
            algorithm='wgf_gmm',
            model_parameters=ModelParameters(
                d_z=2,
                use_particles=True,
                n_particles=N_PARTICLES,
                kernel='norm_fixed_var_w_skip',
                n_hidden=256
            ),
            theta_opt_parameters=ThetaOptParameters(
                lr=1e-4,
                optimizer='rmsprop',
                lr_decay=False,
                regularization=1e-8,
                clip=False
            ),
            r_opt_parameters=ROptParameters(
                lr=1e-2,
                regularization=1e-8
            ),
            extra_alg_parameters=PIDParameters(mc_n_samples=100)
        )
        
        # Initialize step and carry
        init_key, exp_key = jax.random.split(key)
        step, initial_carry = make_step_and_carry(init_key, parameters, target)
        
        # Extract optim
        optim = None
        if hasattr(step, 'keywords') and 'optim' in step.keywords:
            optim = step.keywords['optim']
        elif hasattr(step, 'func') and hasattr(step.func, 'keywords'):
            optim = step.func.keywords.get('optim')
        
        if optim is None:
            print(f"Error: Could not extract optimizer for {problem_name}")
            continue
        
        # Create all hyperparameter combinations
        param_combinations = list(product(
            LAMBDA_REG_VALUES, LR_MEAN_VALUES, LR_COV_VALUES, LR_WEIGHT_VALUES
        ))
        
        print(f"Total combinations to test: {len(param_combinations)}")
        
        # Progress bar for this problem
        pbar = tqdm(param_combinations, desc=f"{problem_name}")
        
        for lambda_reg, lr_mean, lr_cov, lr_weight in pbar:
            # Update progress bar description
            pbar.set_description(f"{problem_name} λ={lambda_reg:.2f} m={lr_mean:.3f}")
            
            # Reset carry and key for this experiment
            carry = initial_carry
            exp_key, run_key = jax.random.split(exp_key)
            
            # Storage for this experiment
            losses = []
            elbo_values = []
            wasserstein_distances = []
            
            # Run training
            for update_idx in range(N_UPDATES):
                run_key, step_key = jax.random.split(run_key)
                
                try:
                    # Perform WGF-GMM step
                    lval, carry, metrics = wgf_gmm_pvi_step_with_monitoring(
                        step_key, carry, target, None, optim, 
                        parameters.extra_alg_parameters,
                        lambda_reg=lambda_reg, lr_mean=lr_mean, 
                        lr_cov=lr_cov, lr_weight=lr_weight
                    )
                    
                    losses.append(float(lval))
                    elbo_values.append(metrics.elbo)
                    wasserstein_distances.append(metrics.wasserstein_distance)
                        
                except Exception as e:
                    print(f"\nError at iteration {update_idx}: {e}")
                    break
            
            # Create experiment identifier
            exp_id = f"{problem_name}_lambda{lambda_reg}_mean{lr_mean}_cov{lr_cov}_weight{lr_weight}"
            
            # Save results for this configuration
            result_data = {
                'problem': problem_name,
                'lambda_reg': lambda_reg,
                'lr_mean': lr_mean,
                'lr_cov': lr_cov,
                'lr_weight': lr_weight,
                'final_loss': losses[-1] if losses else float('inf'),
                'final_elbo': elbo_values[-1] if elbo_values else None,
                'final_wasserstein_distance': wasserstein_distances[-1] if wasserstein_distances else None,
                'mean_loss': np.mean(losses) if losses else float('inf'),
                'std_loss': np.std(losses) if losses else 0.0,
                'converged': len(losses) == N_UPDATES,
                'exp_id': exp_id
            }
            
            all_results.append(result_data)
            
            # Save individual experiment results
            exp_dir = output_dir / problem_name / exp_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            detailed_results = {
                **result_data,
                'losses': losses,
                'elbo_values': elbo_values,
                'wasserstein_distances': wasserstein_distances
            }
            
            with open(exp_dir / "results.pkl", "wb") as f:
                pickle.dump(detailed_results, f)
            
            # Plot and save individual results
            if losses:
                plot_individual_experiment(detailed_results, exp_dir)
    
    # Save comprehensive results
    results_df = pd.DataFrame([{k: v for k, v in result.items() 
                               if k not in ['losses', 'elbo_values', 'wasserstein_distances']} 
                              for result in all_results])
    results_df.to_csv(output_dir / "comprehensive_results.csv", index=False)
    
    # Create summary analysis
    create_summary_analysis(all_results, output_dir)
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Total experiments run: {len(all_results)}")
    
    return all_results, results_df


def plot_individual_experiment(result_data, save_dir):
    """Plot results for an individual experiment."""
    exp_id = result_data['exp_id']
    losses = result_data.get('losses', [])
    elbo_values = result_data.get('elbo_values', [])
    wasserstein_distances = result_data.get('wasserstein_distances', [])
    
    if not losses:
        return
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss curve
    axes[0].plot(losses, color='purple', linewidth=1.5)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # ELBO curve
    if elbo_values and any(v is not None for v in elbo_values):
        valid_elbo = [v for v in elbo_values if v is not None]
        axes[1].plot(valid_elbo, color='blue', linewidth=1.5)
        axes[1].set_title('ELBO')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('ELBO')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'ELBO data not available', 
                       ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('ELBO')
    
    # Wasserstein distance
    if wasserstein_distances and any(w > 0 for w in wasserstein_distances):
        axes[2].plot(wasserstein_distances, color='green', linewidth=1.5)
        axes[2].set_title('Wasserstein Distance')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Distance')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Wasserstein data not available', 
                       ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Wasserstein Distance')
    
    # Add hyperparameters as title
    plt.suptitle(f'{exp_id}\nλ={result_data["lambda_reg"]}, lr_mean={result_data["lr_mean"]}, '
                 f'lr_cov={result_data["lr_cov"]}, lr_weight={result_data["lr_weight"]}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / f"training_curves_{exp_id}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / f"training_curves_{exp_id}.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_analysis(all_results, output_dir):
    """Create summary analysis across all experiments."""
    
    # Group results by problem
    by_problem = {}
    for result in all_results:
        problem = result['problem']
        if problem not in by_problem:
            by_problem[problem] = []
        by_problem[problem].append(result)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (problem_name, results) in enumerate(by_problem.items()):
        if i >= 4:  # Limit to 4 problems
            break
            
        # Extract data (filter out failed experiments)
        successful_results = [r for r in results if r['converged']]
        if not successful_results:
            continue
            
        lambda_vals = [r['lambda_reg'] for r in successful_results]
        lr_mean_vals = [r['lr_mean'] for r in successful_results]
        final_losses = [r['final_loss'] for r in successful_results]
        
        # Loss vs Lambda regularization
        scatter = axes[i].scatter(lambda_vals, final_losses, alpha=0.6, c=lr_mean_vals, 
                                   cmap='viridis', s=20)
        axes[i].set_xlabel('Lambda Regularization')
        axes[i].set_ylabel('Final Loss')
        axes[i].set_title(f'{problem_name.title()} - Loss vs Lambda')
        axes[i].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[i], label='LR Mean')
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "summary_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find and print best configurations
    print("\n" + "="*60)
    print("BEST CONFIGURATIONS BY PROBLEM")
    print("="*60)
    
    for problem_name, results in by_problem.items():
        successful_results = [r for r in results if r['converged']]
        if not successful_results:
            print(f"\n{problem_name.upper()}: No successful runs")
            continue
            
        print(f"\n{problem_name.upper()}:")
        
        # Sort by final loss
        best_results = sorted(successful_results, key=lambda x: x['final_loss'])[:3]
        
        print("Top 3 configurations by final loss:")
        for j, result in enumerate(best_results, 1):
            print(f"  {j}. Loss: {result['final_loss']:.4f} | "
                  f"λ={result['lambda_reg']}, lr_mean={result['lr_mean']}, "
                  f"lr_cov={result['lr_cov']}, lr_weight={result['lr_weight']}")


def main():
    """Main function to handle different experiment modes."""
    
    print("WGF-GMM Hyperparameter Experiments")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'test'  # Default mode
    
    if mode == 'test':
        print("Running single configuration test...")
        success = test_single_config()
        if success:
            print("✓ Single configuration test passed!")
        else:
            print("✗ Single configuration test failed!")
        return
    
    elif mode == 'quick':
        print("Running quick hyperparameter test...")
        results = run_quick_test()
        if results:
            print("✓ Quick test completed!")
        else:
            print("✗ Quick test failed!")
        return
    
    elif mode == 'full':
        print("Running comprehensive hyperparameter search...")
        all_results, results_df = run_comprehensive_experiments()
        
        if all_results is None:
            print("✗ Comprehensive experiments failed!")
            return
        
        print("✓ Comprehensive experiments completed!")
        print(f"Total configurations tested: {len(all_results)}")
        
        # Print overall best configurations
        print("\n" + "="*60)
        print("OVERALL BEST CONFIGURATIONS")
        print("="*60)
        
        # Filter successful results
        successful_results = [r for r in all_results if r['converged']]
        
        if successful_results:
            # Best overall (lowest loss)
            best_overall = min(successful_results, key=lambda x: x['final_loss'])
            print(f"\nBest overall configuration:")
            print(f"Problem: {best_overall['problem']}")
            print(f"Final Loss: {best_overall['final_loss']:.4f}")
            print(f"λ={best_overall['lambda_reg']}, lr_mean={best_overall['lr_mean']}, "
                  f"lr_cov={best_overall['lr_cov']}, lr_weight={best_overall['lr_weight']}")
            
            print(f"\n✓ All analyses completed! Results saved to: output/wgf_gmm_comprehensive")
        else:
            print("\n✗ No successful experiments completed!")
    
    else:
        print("Usage: python run_wgf_gmm_experiments.py [test|quick|full]")
        print("  test  - Test single configuration")
        print("  quick - Quick hyperparameter search")
        print("  full  - Full comprehensive search")


if __name__ == "__main__":
    main()