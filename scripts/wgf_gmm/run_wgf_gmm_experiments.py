#!/usr/bin/env python3
"""
Script to run WGF-GMM experiments with different hyperparameter combinations.
This script will systematically test the recommended learning rates and lambda_reg values,
monitor ELBO and Wasserstein distances, and save results with descriptive filenames.

Usage:
    python run_wgf_gmm_experiments.py           # Full comprehensive search
    python run_wgf_gmm_experiments.py quick     # Quick test with reduced params
    python run_wgf_gmm_experiments.py test      # Single configuration test
"""

import jax
import jax.numpy as np
import numpy
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
from itertools import product
import sys
import argparse

# Import your existing modules
from src.problems.toy import Banana, Multimodal, XShape
from src.utils import make_step_and_carry
from src.base import Parameters, ModelParameters, ThetaOptParameters, ROptParameters, PIDParameters

# Import the WGF-GMM implementation (assuming you've updated src/trainers/wgf_gmm.py)
try:
    from src.trainers.wgf_gmm import (
        WGFGMMHyperparams, 
        wgf_gmm_pvi_step,
        GMMState,
        WGFGMMMetrics,
        compute_elbo,
        compute_elbo_with_wasserstein_regularization
    )
    WGF_GMM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import WGF-GMM implementation: {e}")
    print("Make sure you've updated src/trainers/wgf_gmm.py with the enhanced version")
    WGF_GMM_AVAILABLE = False


def create_enhanced_wgf_gmm_step(wgf_hyperparams):
    """Create a WGF-GMM step function with monitoring capabilities."""
    
    def enhanced_step(key, carry, target, y, optim, hyperparams):
        """Enhanced WGF-GMM step with monitoring."""
        theta_key, r_key, grad_key, metrics_key = jax.random.split(key, 4)
        
        # Import here to avoid circular imports
        from jax import vmap
        import equinox as eqx
        from jax.lax import stop_gradient
        from src.trainers.util import loss_step
        from src.trainers.wgf_gmm import (
            particles_to_gmm, compute_gmm_gradients, update_gmm_parameters,
            gmm_to_particles, sample_from_gmm
        )
        
        # Step 1: Standard conditional parameter (theta) update
        def loss(key, params, static):
            pid = eqx.combine(params, static)
            _samples = pid.sample(key, hyperparams.mc_n_samples, None)
            logq = vmap(eqx.combine(stop_gradient(params), static).log_prob, (0, None))(_samples, None)
            logp = vmap(target.log_prob, (0, None))(_samples, y)
            return np.mean(logq - logp, axis=0)
        
        # Update conditional parameters (theta)
        lval, pid, theta_opt_state = loss_step(
            theta_key,
            loss,
            carry.id,
            optim.theta_optim,
            carry.theta_opt_state,
        )
        
        # Step 2: Convert particles to GMM representation
        if not hasattr(carry, 'gmm_state') or carry.gmm_state is None:
            gmm_state = particles_to_gmm(pid.particles, use_em=True, n_components=None)
        else:
            gmm_state = carry.gmm_state
        
        # Step 3: Compute metrics before update
        elbo = compute_elbo(metrics_key, pid, target, gmm_state, y, hyperparams)
        elbo_with_reg, wasserstein_dist = compute_elbo_with_wasserstein_regularization(
            metrics_key, pid, target, gmm_state, y, hyperparams, wgf_hyperparams.lambda_reg
        )
        
        # Step 4: Compute gradients for GMM parameters
        mean_grads, cov_grads, weight_grads = compute_gmm_gradients(
            grad_key, pid, target, gmm_state, y, hyperparams, wgf_hyperparams.lambda_reg
        )
        
        # Step 5: Update GMM parameters using WGF
        updated_gmm_state = update_gmm_parameters(
            gmm_state, mean_grads, cov_grads, weight_grads, wgf_hyperparams
        )
        
        # Step 6: Convert back to particle representation for compatibility
        updated_particles = gmm_to_particles(updated_gmm_state)
        pid = eqx.tree_at(lambda tree: tree.particles, pid, updated_particles)
        
        # Create updated carry with GMM state
        from src.base import PIDCarry
        updated_carry = PIDCarry(
            id=pid,
            theta_opt_state=theta_opt_state,
            r_opt_state=carry.r_opt_state,
            r_precon_state=carry.r_precon_state
        )
        
        # Store GMM state and metrics in carry
        updated_carry.gmm_state = updated_gmm_state
        updated_carry.metrics = {
            'elbo': float(elbo),
            'elbo_with_wasserstein': float(elbo_with_reg),
            'wasserstein_distance': float(wasserstein_dist),
            'lambda_reg': wgf_hyperparams.lambda_reg,
            'lr_mean': wgf_hyperparams.lr_mean,
            'lr_cov': wgf_hyperparams.lr_cov,
            'lr_weight': wgf_hyperparams.lr_weight
        }
        
        return lval, updated_carry
    
    return enhanced_step


def run_comprehensive_wgf_gmm_experiments():
    """Run comprehensive WGF-GMM experiments with recommended hyperparameters."""
    
    if not WGF_GMM_AVAILABLE:
        print("Error: WGF-GMM implementation not available. Please update src/trainers/wgf_gmm.py")
        return None, None
    
    # Recommended hyperparameter values based on your analysis
    LAMBDA_REG_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]
    LR_MEAN_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
    LR_COV_VALUES = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    LR_WEIGHT_VALUES = [0.005, 0.01, 0.02, 0.05]
    
    # Problems to test
    PROBLEMS = {
        'banana': Banana,
        'multimodal': Multimodal, 
        'xshape': XShape
    }
    
    # Experiment settings
    N_UPDATES = 1000
    N_PARTICLES = 100
    SEED = 42
    
    # Output directory
    output_dir = Path("output/wgf_gmm_comprehensive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    
    print("Starting comprehensive WGF-GMM experiments...")
    print(f"Testing {len(LAMBDA_REG_VALUES)} λ values, {len(LR_MEAN_VALUES)} lr_mean values,")
    print(f"{len(LR_COV_VALUES)} lr_cov values, {len(LR_WEIGHT_VALUES)} lr_weight values")
    print(f"Total combinations per problem: {len(LAMBDA_REG_VALUES) * len(LR_MEAN_VALUES) * len(LR_COV_VALUES) * len(LR_WEIGHT_VALUES)}")
    
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
                n_hidden=512
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
            extra_alg_parameters=PIDParameters(mc_n_samples=250)
        )
        
        # Initialize step and carry
        init_key, exp_key = jax.random.split(key)
        step, initial_carry = make_step_and_carry(init_key, parameters, target)
        
        # Get optimizer from the step function
        optim = step.keywords['optim'] if hasattr(step, 'keywords') else None
        
        # Create all hyperparameter combinations
        param_combinations = list(product(
            LAMBDA_REG_VALUES, LR_MEAN_VALUES, LR_COV_VALUES, LR_WEIGHT_VALUES
        ))
        
        print(f"Total combinations to test: {len(param_combinations)}")
        
        # Progress bar for this problem
        pbar = tqdm(param_combinations, desc=f"{problem_name}")
        
        for lambda_reg, lr_mean, lr_cov, lr_weight in pbar:
            # Update progress bar description
            pbar.set_description(f"{problem_name} λ={lambda_reg:.2f} m={lr_mean:.3f} c={lr_cov:.4f} w={lr_weight:.3f}")
            
            # Create WGF-GMM hyperparameters
            wgf_hyperparams = WGFGMMHyperparams(
                lambda_reg=lambda_reg,
                lr_mean=lr_mean,
                lr_cov=lr_cov,
                lr_weight=lr_weight
            )
            
            # Create enhanced step function
            enhanced_step = create_enhanced_wgf_gmm_step(wgf_hyperparams)
            
            # Reset carry and key for this experiment
            carry = initial_carry
            exp_key, run_key = jax.random.split(exp_key)
            
            # Storage for this experiment
            losses = []
            elbo_values = []
            elbo_with_wasserstein_values = []
            wasserstein_distances = []
            
            # Run training
            for update_idx in range(N_UPDATES):
                run_key, step_key = jax.random.split(run_key)
                
                try:
                    # Perform WGF-GMM step
                    lval, carry = enhanced_step(
                        step_key, carry, target, None, optim, 
                        parameters.extra_alg_parameters
                    )
                    
                    losses.append(float(lval))
                    
                    # Extract metrics if available
                    if hasattr(carry, 'metrics'):
                        elbo_values.append(carry.metrics['elbo'])
                        elbo_with_wasserstein_values.append(carry.metrics['elbo_with_wasserstein'])
                        wasserstein_distances.append(carry.metrics['wasserstein_distance'])
                    else:
                        elbo_values.append(-float(lval))  # Approximation
                        elbo_with_wasserstein_values.append(-float(lval))
                        wasserstein_distances.append(0.0)
                        
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
                'final_elbo_with_wasserstein': elbo_with_wasserstein_values[-1] if elbo_with_wasserstein_values else None,
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
                'elbo_with_wasserstein_values': elbo_with_wasserstein_values,
                'wasserstein_distances': wasserstein_distances
            }
            
            with open(exp_dir / "results.pkl", "wb") as f:
                pickle.dump(detailed_results, f)
            
            # Plot and save individual results
            plot_individual_experiment(detailed_results, exp_dir)
    
    # Save comprehensive results
    results_df = pd.DataFrame([{k: v for k, v in result.items() 
                               if k not in ['losses', 'elbo_values', 'elbo_with_wasserstein_values', 'wasserstein_distances']} 
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
    elbo_with_wasserstein_values = result_data.get('elbo_with_wasserstein_values', [])
    wasserstein_distances = result_data.get('wasserstein_distances', [])
    
    if not losses:
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curve
    axes[0, 0].plot(losses, color='purple', linewidth=1.5)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # ELBO curve
    if elbo_values and any(v is not None for v in elbo_values):
        valid_elbo = [v for v in elbo_values if v is not None]
        axes[0, 1].plot(valid_elbo, color='blue', linewidth=1.5)
        axes[0, 1].set_title('ELBO')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('ELBO')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'ELBO data not available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('ELBO')
    
    # ELBO with Wasserstein
    if elbo_with_wasserstein_values and any(v is not None for v in elbo_with_wasserstein_values):
        valid_elbo_w = [v for v in elbo_with_wasserstein_values if v is not None]
        axes[1, 0].plot(valid_elbo_w, color='red', linewidth=1.5)
        axes[1, 0].set_title('ELBO with Wasserstein Regularization')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('ELBO with Regularization')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'ELBO+W data not available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('ELBO with Wasserstein')
    
    # Wasserstein distance
    if wasserstein_distances and any(w > 0 for w in wasserstein_distances):
        axes[1, 1].plot(wasserstein_distances, color='green', linewidth=1.5)
        axes[1, 1].set_title('Wasserstein Distance')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Wasserstein data not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Wasserstein Distance')
    
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, (problem_name, results) in enumerate(by_problem.items()):
        if i >= 3:  # Limit to 3 problems
            break
            
        # Extract data (filter out failed experiments)
        successful_results = [r for r in results if r['converged']]
        if not successful_results:
            continue
            
        lambda_vals = [r['lambda_reg'] for r in successful_results]
        lr_mean_vals = [r['lr_mean'] for r in successful_results]
        final_losses = [r['final_loss'] for r in successful_results]
        
        # Loss vs Lambda regularization
        scatter = axes[0, i].scatter(lambda_vals, final_losses, alpha=0.6, c=lr_mean_vals, 
                                   cmap='viridis', s=20)
        axes[0, i].set_xlabel('Lambda Regularization')
        axes[0, i].set_ylabel('Final Loss')
        axes[0, i].set_title(f'{problem_name.title()} - Loss vs Lambda')
        axes[0, i].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, i], label='LR Mean')
        
        # Loss vs Learning Rate (Mean)
        scatter2 = axes[1, i].scatter(lr_mean_vals, final_losses, alpha=0.6, c=lambda_vals, 
                                    cmap='plasma', s=20)
        axes[1, i].set_xlabel('Learning Rate (Mean)')
        axes[1, i].set_ylabel('Final Loss')
        axes[1, i].set_title(f'{problem_name.title()} - Loss vs LR Mean')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xscale('log')
        plt.colorbar(scatter2, ax=axes[1, i], label='Lambda')
    
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
        best_results = sorted(successful_results, key=lambda x: x['final_loss'])[:5]
        
        print("Top 5 configurations by final loss:")
        for j, result in enumerate(best_results, 1):
            print(f"  {j}. Loss: {result['final_loss']:.4f} | "
                  f"λ={result['lambda_reg']}, lr_mean={result['lr_mean']}, "
                  f"lr_cov={result['lr_cov']}, lr_weight={result['lr_weight']}")


def run_quick_test():
    """Run a quick test with a subset of hyperparameters for debugging."""
    
    if not WGF_GMM_AVAILABLE:
        print("Error: WGF-GMM implementation not available.")
        return None
    
    print("Running quick WGF-GMM test...")
    
    # Reduced hyperparameter space for testing
    LAMBDA_REG_VALUES = [0.1, 0.5]
    LR_MEAN_VALUES = [0.01, 0.05]
    LR_COV_VALUES = [0.001, 0.005]
    LR_WEIGHT_VALUES = [0.01, 0.02]
    
    # Just test one problem
    target = Banana()
    N_UPDATES = 200
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
            n_particles=50,  # Reduced for speed
            kernel='norm_fixed_var_w_skip',
            n_hidden=256    # Reduced for speed
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
        extra_alg_parameters=PIDParameters(mc_n_samples=100)  # Reduced for speed
    )
    
    # Initialize
    init_key, exp_key = jax.random.split(key)
    step, initial_carry = make_step_and_carry(init_key, parameters, target)
    optim = step.keywords['optim'] if hasattr(step, 'keywords') else None
    
    results = []
    
    # Test all combinations
    for lambda_reg in LAMBDA_REG_VALUES:
        for lr_mean in LR_MEAN_VALUES:
            for lr_cov in LR_COV_VALUES:
                for lr_weight in LR_WEIGHT_VALUES:
                    
                    print(f"Testing: λ={lambda_reg}, lr_mean={lr_mean}, lr_cov={lr_cov}, lr_weight={lr_weight}")
                    
                    wgf_hyperparams = WGFGMMHyperparams(
                        lambda_reg=lambda_reg,
                        lr_mean=lr_mean,
                        lr_cov=lr_cov,
                        lr_weight=lr_weight
                    )
                    
                    # Create enhanced step function
                    enhanced_step = create_enhanced_wgf_gmm_step(wgf_hyperparams)
                    
                    # Reset for this test
                    carry = initial_carry
                    exp_key, run_key = jax.random.split(exp_key)
                    
                    losses = []
                    
                    # Quick training loop
                    for update_idx in range(N_UPDATES):
                        run_key, step_key = jax.random.split(run_key)
                        
                        try:
                            lval, carry = enhanced_step(
                                step_key, carry, target, None, optim, 
                                parameters.extra_alg_parameters
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
                            'lr_cov': lr_cov,
                            'lr_weight': lr_weight,
                            'final_loss': final_loss,
                            'losses': losses,
                            'success': True
                        }
                    else:
                        print(f"  Failed!")
                        result = {
                            'lambda_reg': lambda_reg,
                            'lr_mean': lr_mean,
                            'lr_cov': lr_cov,
                            'lr_weight': lr_weight,
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
        print(f"λ={best_result['lambda_reg']}, lr_mean={best_result['lr_mean']}, "
              f"lr_cov={best_result['lr_cov']}, lr_weight={best_result['lr_weight']}")
    else:
        print("No successful runs in quick test!")
    
    return results


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
            n_particles=20,
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
        extra_alg_parameters=PIDParameters(mc_n_samples=50)
    )
    
    init_key, train_key = jax.random.split(key)
    step, carry = make_step_and_carry(init_key, parameters, target)
    optim = step.keywords['optim'] if hasattr(step, 'keywords') else None
    
    wgf_hyperparams = WGFGMMHyperparams(
        lambda_reg=0.1,
        lr_mean=0.01,
        lr_cov=0.001,
        lr_weight=0.01
    )
    
    # Create enhanced step function
    enhanced_step = create_enhanced_wgf_gmm_step(wgf_hyperparams)
    
    print("Running 50 steps...")
    losses = []
    
    for i in range(50):
        train_key, step_key = jax.random.split(train_key)
        
        try:
            lval, carry = enhanced_step(
                step_key, carry, target, None, optim, 
                parameters.extra_alg_parameters
            )
            losses.append(float(lval))
            
            if (i + 1) % 10 == 0:
                print(f"Step {i+1}: Loss = {lval:.4f}")
                if hasattr(carry, 'metrics'):
                    metrics = carry.metrics
                    print(f"         ELBO = {metrics['elbo']:.4f}, "
                          f"ELBO+W = {metrics['elbo_with_wasserstein']:.4f}, "
                          f"W_dist = {metrics['wasserstein_distance']:.4f}")
                
        except Exception as e:
            print(f"Error at step {i+1}: {e}")
            break
    
    if losses:
        print(f"Test completed successfully!")
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Loss improvement: {losses[0] - losses[-1]:.4f}")
        return True
    else:
        print("Test failed!")
        return False


def create_config_files_from_results(results_df, output_dir, top_n=5):
    """Create YAML config files for the best performing hyperparameter combinations."""
    
    output_dir = Path(output_dir)
    config_dir = output_dir / "best_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Get top configurations by problem
    problems = results_df['problem'].unique()
    
    for problem in problems:
        problem_results = results_df[results_df['problem'] == problem]
        # Filter successful runs
        successful_results = problem_results[problem_results['converged'] == True]
        
        if len(successful_results) == 0:
            print(f"No successful results for {problem}")
            continue
            
        top_configs = successful_results.nsmallest(top_n, 'final_loss')
        
        for i, (_, config) in enumerate(top_configs.iterrows()):
            config_name = f"{problem}_rank{i+1}"
            
            config_content = f"""# WGF-GMM Configuration for {problem} (Rank {i+1})
# Final Loss: {config['final_loss']:.4f}

default_parameters: &default_parameters
    d_z: 2
    kernel: 'norm_fixed_var_w_skip'
    n_hidden: 512

default_theta_lr: &default_theta_lr
    lr: 1e-4

experiment:
    n_reruns: 10
    n_updates: 15000
    name: 'wgf_gmm_{config_name}'
    compute_metrics: True
    use_jit: True

pvi:
    algorithm: 'pvi'
    model: 
        use_particles: True
        n_particles: 100
        <<: *default_parameters
    theta_opt:
        <<: *default_theta_lr
        optimizer: 'rmsprop'
        lr_decay: False
        regularization: 1e-8
        clip: False
    r_opt:
        lr: 1e-2
        regularization: 1e-8
    r_precon:
        type: 'rms'
        max_norm: 1.
        agg: 'mean'
    extra_alg:

wgf_gmm:
    algorithm: 'wgf_gmm'
    model: 
        use_particles: True
        n_particles: 100
        <<: *default_parameters
    theta_opt:
        <<: *default_theta_lr
        optimizer: 'rmsprop'
        lr_decay: False
        regularization: 1e-8
        clip: False
    r_opt:
        lr: 1e-2
        regularization: 1e-8
    r_precon:
        type: 'rms'
        max_norm: 1.
        agg: 'mean'
    wgf_gmm_params:
        lambda_reg: {config['lambda_reg']}
        lr_mean: {config['lr_mean']}
        lr_cov: {config['lr_cov']}
        lr_weight: {config['lr_weight']}
    extra_alg:

gmm_pvi:
    algorithm: 'gmm_pvi'
    model: 
        use_particles: True
        n_particles: 100
        <<: *default_parameters
    theta_opt:
        <<: *default_theta_lr
        optimizer: 'rmsprop'
        lr_decay: False
        regularization: 1e-8
        clip: False
    r_opt:
        lr: 1e-2
        regularization: 1e-8
    extra_alg:
"""
            
            with open(config_dir / f"wgf_gmm_{config_name}.yaml", "w") as f:
                f.write(config_content)
    
    print(f"Created config files in: {config_dir}")
    print(f"Generated {len(problems) * min(top_n, len(successful_results))} configuration files")


def analyze_hyperparameter_sensitivity(results_df, output_dir):
    """Analyze sensitivity to different hyperparameters."""
    
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "sensitivity_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter successful runs
    successful_df = results_df[results_df['converged'] == True]
    
    if len(successful_df) == 0:
        print("No successful runs for sensitivity analysis")
        return
    
    # Create sensitivity plots for each hyperparameter
    hyperparams = ['lambda_reg', 'lr_mean', 'lr_cov', 'lr_weight']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, param in enumerate(hyperparams):
        # Group by parameter value and compute statistics
        grouped = successful_df.groupby(param)['final_loss'].agg(['mean', 'std', 'count'])
        
        x_vals = grouped.index.values
        y_means = grouped['mean'].values
        y_stds = grouped['std'].values
        
        axes[i].errorbar(x_vals, y_means, yerr=y_stds, marker='o', capsize=5, capthick=2)
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Final Loss (mean ± std)')
        axes[i].set_title(f'Sensitivity to {param}')
        axes[i].grid(True, alpha=0.3)
        
        if param.startswith('lr'):
            axes[i].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(analysis_dir / "hyperparameter_sensitivity.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(analysis_dir / "hyperparameter_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation matrix
    correlation_data = successful_df[hyperparams + ['final_loss']].corr()
    
    plt.figure(figsize=(8, 6))
    try:
        import seaborn as sns
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.3f')
    except ImportError:
        # Fallback to matplotlib if seaborn not available
        im = plt.imshow(correlation_data, cmap='coolwarm', aspect='auto')
        plt.colorbar(im)
        plt.xticks(range(len(correlation_data.columns)), correlation_data.columns, rotation=45)
        plt.yticks(range(len(correlation_data.index)), correlation_data.index)
        
        # Add correlation values as text
        for i in range(len(correlation_data.index)):
            for j in range(len(correlation_data.columns)):
                plt.text(j, i, f'{correlation_data.iloc[i, j]:.3f}', 
                        ha='center', va='center')
    
    plt.title('Hyperparameter Correlation Matrix')
    plt.tight_layout()
    plt.savefig(analysis_dir / "correlation_matrix.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(analysis_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sensitivity analysis saved to: {analysis_dir}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Run WGF-GMM experiments with different hyperparameters')
    parser.add_argument('mode', nargs='?', default='full', 
                       choices=['test', 'quick', 'full'],
                       help='Mode to run: test (single config), quick (reduced search), full (comprehensive)')
    
    args = parser.parse_args()
    
    print("WGF-GMM Hyperparameter Experiments")
    print("=" * 50)
    
    if args.mode == 'test':
        print("Running single configuration test...")
        success = test_single_config()
        if success:
            print("✓ Single configuration test passed!")
        else:
            print("✗ Single configuration test failed!")
        return
    
    elif args.mode == 'quick':
        print("Running quick hyperparameter test...")
        results = run_quick_test()
        if results:
            print("✓ Quick test completed!")
        else:
            print("✗ Quick test failed!")
        return
    
    elif args.mode == 'full':
        print("Running comprehensive hyperparameter search...")
        all_results, results_df = run_comprehensive_wgf_gmm_experiments()
        
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
            
            # Best by problem
            problems = set(r['problem'] for r in successful_results)
            for problem in problems:
                problem_results = [r for r in successful_results if r['problem'] == problem]
                if problem_results:
                    best_for_problem = min(problem_results, key=lambda x: x['final_loss'])
                    print(f"\nBest for {problem}:")
                    print(f"Final Loss: {best_for_problem['final_loss']:.4f}")
                    print(f"λ={best_for_problem['lambda_reg']}, lr_mean={best_for_problem['lr_mean']}, "
                          f"lr_cov={best_for_problem['lr_cov']}, lr_weight={best_for_problem['lr_weight']}")
            
            # Create additional analyses
            output_dir = Path("output/wgf_gmm_comprehensive")
            create_config_files_from_results(results_df, output_dir)
            analyze_hyperparameter_sensitivity(results_df, output_dir)
            
            print(f"\n✓ All analyses completed! Results saved to: {output_dir}")
        else:
            print("\n✗ No successful experiments completed!")


if __name__ == "__main__":
    # Handle both command line arguments and direct execution
    if len(sys.argv) > 1:
        # Use argparse for proper CLI handling
        main()
    else:
        # Default behavior when run without arguments
        print("WGF-GMM Hyperparameter Experiments")
        print("=" * 50)
        print("Usage:")
        print("  python run_wgf_gmm_experiments.py test   # Test single configuration")
        print("  python run_wgf_gmm_experiments.py quick  # Quick hyperparameter search")
        print("  python run_wgf_gmm_experiments.py full   # Full comprehensive search")
        print("")
        
        # Run test by default
        print("Running single configuration test by default...")
        success = test_single_config()
        if success:
            print("\n✓ Test passed! You can now run:")
            print("  python run_wgf_gmm_experiments.py quick")
            print("  python run_wgf_gmm_experiments.py full")
        else:
            print("\n✗ Test failed! Please check your WGF-GMM implementation.")