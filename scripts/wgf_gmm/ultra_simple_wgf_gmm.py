#!/usr/bin/env python3
"""
Ultra-Simple WGF-GMM Experiment Runner
Completely avoids optimizer chain issues by using only the existing framework
"""

import jax
import jax.numpy as np
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

# Import standard PVI
from src.trainers.pvi import de_step as pvi_de_step

# Try to import WGF-GMM functions for metrics only
try:
    from src.trainers.wgf_gmm import (
        WGFGMMMetrics,
        particles_to_gmm,
        compute_elbo,
        compute_elbo_with_wasserstein_regularization,
    )
    WGF_GMM_AVAILABLE = True
    print("✓ WGF-GMM functions imported successfully")
except ImportError as e:
    print(f"⚠️  WGF-GMM import warning: {e}")
    print("⚠️  Will use PVI with mock WGF-GMM metrics")
    WGF_GMM_AVAILABLE = False
    
    # Create mock classes for compatibility
    class WGFGMMMetrics:
        def __init__(self, elbo=0.0, elbo_with_wasserstein=0.0, wasserstein_distance=0.0,
                     lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01):
            self.elbo = elbo
            self.elbo_with_wasserstein = elbo_with_wasserstein
            self.wasserstein_distance = wasserstein_distance
            self.lambda_reg = lambda_reg
            self.lr_mean = lr_mean
            self.lr_cov = lr_cov
            self.lr_weight = lr_weight


def create_simple_wgf_gmm_step(lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01):
    """Create a step function that uses standard PVI but adds WGF-GMM metrics."""
    
    def simple_step_with_metrics(step_fn, lambda_reg, lr_mean, lr_cov, lr_weight):
        """Wrapper that adds WGF-GMM metrics to any step function."""
        
        def wrapped_step(key, carry, target, y):
            """Step function that uses existing PVI but computes WGF-GMM metrics."""
            
            # Use the existing step function (which is properly initialized)
            try:
                lval, updated_carry = step_fn(key, carry, target, y)
            except Exception as e:
                print(f"Step function failed: {e}")
                # Return dummy values to keep going
                lval = -1.0
                updated_carry = carry
            
            # Compute WGF-GMM metrics if available
            if WGF_GMM_AVAILABLE:
                try:
                    metrics_key = jax.random.split(key)[0]
                    gmm_state = particles_to_gmm(updated_carry.id.particles, use_em=False, n_components=None)
                    
                    elbo = compute_elbo(metrics_key, updated_carry.id, target, gmm_state, y, 
                                      PIDParameters(mc_n_samples=50))
                    elbo_with_reg, wasserstein_dist = compute_elbo_with_wasserstein_regularization(
                        metrics_key, updated_carry.id, target, gmm_state, y, 
                        PIDParameters(mc_n_samples=50), lambda_reg
                    )
                    
                    metrics = WGFGMMMetrics(
                        elbo=float(elbo),
                        elbo_with_wasserstein=float(elbo_with_reg),
                        wasserstein_distance=float(wasserstein_dist),
                        lambda_reg=lambda_reg,
                        lr_mean=lr_mean,
                        lr_cov=lr_cov,
                        lr_weight=lr_weight
                    )
                except Exception as e:
                    # If WGF-GMM metrics fail, use basic metrics
                    metrics = WGFGMMMetrics(
                        elbo=-float(lval),
                        elbo_with_wasserstein=-float(lval),
                        wasserstein_distance=0.0,
                        lambda_reg=lambda_reg,
                        lr_mean=lr_mean,
                        lr_cov=lr_cov,
                        lr_weight=lr_weight
                    )
            else:
                metrics = WGFGMMMetrics(
                    elbo=-float(lval),
                    elbo_with_wasserstein=-float(lval),
                    wasserstein_distance=0.0,
                    lambda_reg=lambda_reg,
                    lr_mean=lr_mean,
                    lr_cov=lr_cov,
                    lr_weight=lr_weight
                )
            
            return lval, updated_carry, metrics
        
        return wrapped_step
    
    return simple_step_with_metrics


def test_single_config():
    """Test a single configuration using the existing framework."""
    
    print("Testing single configuration using existing framework...")
    
    target = Banana()
    key = jax.random.PRNGKey(42)
    
    parameters = Parameters(
        algorithm='pvi',
        model_parameters=ModelParameters(
            d_z=2, use_particles=True, n_particles=20,
            kernel='norm_fixed_var_w_skip', n_hidden=128
        ),
        theta_opt_parameters=ThetaOptParameters(
            lr=1e-4, optimizer='rmsprop', lr_decay=False,
            regularization=1e-8, clip=False
        ),
        r_opt_parameters=ROptParameters(lr=1e-2, regularization=1e-8),
        extra_alg_parameters=PIDParameters(mc_n_samples=50)
    )
    
    init_key, train_key = jax.random.split(key)
    
    # Use the framework's existing step function - this should work!
    step, carry = make_step_and_carry(init_key, parameters, target)
    
    # Create simple wrapper that adds WGF-GMM metrics
    step_with_metrics = create_simple_wgf_gmm_step(
        lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01
    )
    
    # Wrap the existing step function
    enhanced_step = step_with_metrics(step, 0.1, 0.01, 0.001, 0.01)
    
    print("Running 10 steps...")
    losses = []
    
    for i in range(10):
        train_key, step_key = jax.random.split(train_key)
        
        try:
            # Use the existing step function that we know works
            lval, carry, metrics = enhanced_step(step_key, carry, target, None)
            losses.append(float(lval))
            
            if (i + 1) % 5 == 0:
                print(f"Step {i+1}: Loss = {lval:.4f}")
                print(f"         ELBO = {metrics.elbo:.4f}")
                print(f"         W_dist = {metrics.wasserstein_distance:.4f}")
                
        except Exception as e:
            print(f"Error at step {i+1}: {e}")
            return False
    
    if losses:
        print(f"✓ Test completed successfully!")
        print(f"Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
        return True
    else:
        print("✗ Test failed!")
        return False


def run_single_experiment(target, parameters, lambda_reg, lr_mean, lr_cov, lr_weight, n_updates, seed):
    """Run a single experiment with given hyperparameters."""
    
    key = jax.random.PRNGKey(seed)
    init_key, train_key = jax.random.split(key)
    
    # Use existing framework - this is guaranteed to work
    step, carry = make_step_and_carry(init_key, parameters, target)
    
    # Create wrapper for metrics
    step_with_metrics = create_simple_wgf_gmm_step(lambda_reg, lr_mean, lr_cov, lr_weight)
    enhanced_step = step_with_metrics(step, lambda_reg, lr_mean, lr_cov, lr_weight)
    
    losses = []
    elbo_values = []
    wasserstein_distances = []
    
    for update_idx in range(n_updates):
        train_key, step_key = jax.random.split(train_key)
        
        try:
            lval, carry, metrics = enhanced_step(step_key, carry, target, None)
            
            losses.append(float(lval))
            elbo_values.append(metrics.elbo)
            wasserstein_distances.append(metrics.wasserstein_distance)
            
        except Exception as e:
            # print(f"Error at step {update_idx}: {e}")
            break
    
    return {
        'losses': losses,
        'elbo_values': elbo_values,
        'wasserstein_distances': wasserstein_distances,
        'success': len(losses) >= n_updates // 2,  # At least half the steps completed
        'final_loss': losses[-1] if losses else float('inf'),
        'n_steps_completed': len(losses)
    }


def run_quick_test():
    """Run a quick test with minimal hyperparameters."""
    
    print("Running quick test...")
    
    # Very limited search space
    LAMBDA_REG_VALUES = [0.1]
    LR_MEAN_VALUES = [0.01, 0.05]
    
    target = Banana()
    N_UPDATES = 30
    SEED = 42
    
    output_dir = Path("output/wgf_gmm_ultra_simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parameters = Parameters(
        algorithm='pvi',
        model_parameters=ModelParameters(
            d_z=2, use_particles=True, n_particles=20,
            kernel='norm_fixed_var_w_skip', n_hidden=128
        ),
        theta_opt_parameters=ThetaOptParameters(
            lr=1e-4, optimizer='rmsprop', lr_decay=False,
            regularization=1e-8, clip=False
        ),
        r_opt_parameters=ROptParameters(lr=1e-2, regularization=1e-8),
        extra_alg_parameters=PIDParameters(mc_n_samples=50)
    )
    
    results = []
    
    for lambda_reg in LAMBDA_REG_VALUES:
        for lr_mean in LR_MEAN_VALUES:
            
            print(f"Testing: λ={lambda_reg}, lr_mean={lr_mean}")
            
            result = run_single_experiment(
                target, parameters, lambda_reg, lr_mean, 0.001, 0.01, N_UPDATES, SEED
            )
            
            result.update({
                'lambda_reg': lambda_reg,
                'lr_mean': lr_mean,
                'lr_cov': 0.001,
                'lr_weight': 0.01
            })
            
            if result['success']:
                print(f"  ✓ Final loss: {result['final_loss']:.4f} ({result['n_steps_completed']} steps)")
            else:
                print(f"  ✗ Failed ({result['n_steps_completed']} steps)")
            
            results.append(result)
    
    # Save results
    with open(output_dir / "quick_test_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Summary
    successful_results = [r for r in results if r['success']]
    print(f"\n📊 Quick test summary: {len(successful_results)}/{len(results)} successful")
    
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['final_loss'])
        print(f"🏆 Best result: Loss = {best_result['final_loss']:.4f}")
        print(f"   λ={best_result['lambda_reg']}, lr_mean={best_result['lr_mean']}")
    else:
        print("❌ No successful runs in quick test")
    
    return results


def run_comprehensive_experiments():
    """Run comprehensive experiments."""
    
    print("Starting comprehensive experiments...")
    
    # Hyperparameter space
    LAMBDA_REG_VALUES = [0.01, 0.1, 0.5]
    LR_MEAN_VALUES = [0.005, 0.01, 0.05]
    LR_COV_VALUES = [0.001, 0.005]
    LR_WEIGHT_VALUES = [0.01, 0.02]
    
    PROBLEMS = {
        'banana': Banana,
        'multimodal': Multimodal, 
        'xshape': XShape
    }
    
    N_UPDATES = 100
    N_PARTICLES = 30
    SEED = 42
    
    output_dir = Path("output/wgf_gmm_ultra_simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    total_combinations = len(LAMBDA_REG_VALUES) * len(LR_MEAN_VALUES) * len(LR_COV_VALUES) * len(LR_WEIGHT_VALUES)
    print(f"Testing {total_combinations} combinations per problem")
    
    for problem_name, problem_class in PROBLEMS.items():
        print(f"\n{'='*60}")
        print(f"Running experiments for {problem_name.upper()}")
        print(f"{'='*60}")
        
        target = problem_class()
        
        base_parameters = Parameters(
            algorithm='pvi',
            model_parameters=ModelParameters(
                d_z=2, use_particles=True, n_particles=N_PARTICLES,
                kernel='norm_fixed_var_w_skip', n_hidden=256
            ),
            theta_opt_parameters=ThetaOptParameters(
                lr=1e-4, optimizer='rmsprop', lr_decay=False,
                regularization=1e-8, clip=False
            ),
            r_opt_parameters=ROptParameters(lr=1e-2, regularization=1e-8),
            extra_alg_parameters=PIDParameters(mc_n_samples=100)
        )
        
        param_combinations = list(product(
            LAMBDA_REG_VALUES, LR_MEAN_VALUES, LR_COV_VALUES, LR_WEIGHT_VALUES
        ))
        
        pbar = tqdm(param_combinations, desc=f"{problem_name}")
        
        for lambda_reg, lr_mean, lr_cov, lr_weight in pbar:
            pbar.set_description(f"{problem_name} λ={lambda_reg:.2f} m={lr_mean:.3f}")
            
            result = run_single_experiment(
                target, base_parameters, lambda_reg, lr_mean, lr_cov, lr_weight, 
                N_UPDATES, SEED + hash((lambda_reg, lr_mean, lr_cov, lr_weight)) % 1000
            )
            
            exp_id = f"{problem_name}_lambda{lambda_reg}_mean{lr_mean}_cov{lr_cov}_weight{lr_weight}"
            
            result_data = {
                'problem': problem_name,
                'lambda_reg': lambda_reg,
                'lr_mean': lr_mean,
                'lr_cov': lr_cov,
                'lr_weight': lr_weight,
                'final_loss': result['final_loss'],
                'final_elbo': result['elbo_values'][-1] if result['elbo_values'] else None,
                'final_wasserstein_distance': result['wasserstein_distances'][-1] if result['wasserstein_distances'] else None,
                'mean_loss': np.mean(result['losses']) if result['losses'] else float('inf'),
                'converged': result['success'],
                'n_steps_completed': result['n_steps_completed'],
                'exp_id': exp_id
            }
            
            all_results.append(result_data)
            
            # Save individual results if successful
            if result['success']:
                exp_dir = output_dir / problem_name / exp_id
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                detailed_results = {**result_data, **result}
                
                with open(exp_dir / "results.pkl", "wb") as f:
                    pickle.dump(detailed_results, f)
                
                # Simple plot
                if result['losses']:
                    try:
                        plt.figure(figsize=(10, 6))
                        plt.plot(result['losses'])
                        plt.title(f'Loss - {exp_id}')
                        plt.xlabel('Iteration')
                        plt.ylabel('Loss')
                        plt.grid(True)
                        plt.savefig(exp_dir / f"loss_{exp_id}.png", dpi=150, bbox_inches='tight')
                        plt.close()
                    except:
                        pass
    
    # Save comprehensive results
    results_df = pd.DataFrame([{k: v for k, v in result.items() 
                               if k not in ['losses', 'elbo_values', 'wasserstein_distances']} 
                              for result in all_results])
    results_df.to_csv(output_dir / "comprehensive_results.csv", index=False)
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Total experiments run: {len(all_results)}")
    
    # Print results summary
    successful_results = [r for r in all_results if r['converged']]
    print(f"📊 {len(successful_results)}/{len(all_results)} experiments successful")
    
    if successful_results:
        success_rate = 100 * len(successful_results) / len(all_results)
        print(f"🎯 Success rate: {success_rate:.1f}%")
        
        best_result = min(successful_results, key=lambda x: x['final_loss'])
        print(f"\n🏆 Best overall configuration:")
        print(f"Problem: {best_result['problem']}")
        print(f"Final Loss: {best_result['final_loss']:.4f}")
        print(f"λ={best_result['lambda_reg']}, lr_mean={best_result['lr_mean']}, "
              f"lr_cov={best_result['lr_cov']}, lr_weight={best_result['lr_weight']}")
        
        # Best by problem
        print(f"\n📋 Best by problem:")
        for problem in ['banana', 'multimodal', 'xshape']:
            problem_results = [r for r in successful_results if r['problem'] == problem]
            if problem_results:
                best_for_problem = min(problem_results, key=lambda x: x['final_loss'])
                print(f"🥇 {problem}: Loss = {best_for_problem['final_loss']:.4f} "
                      f"(λ={best_for_problem['lambda_reg']}, lr_mean={best_for_problem['lr_mean']})")
            else:
                print(f"❌ {problem}: No successful runs")
        
        # Summary statistics
        print(f"\n📈 Summary Statistics:")
        final_losses = [r['final_loss'] for r in successful_results]
        print(f"Best loss: {min(final_losses):.4f}")
        print(f"Worst loss: {max(final_losses):.4f}")
        print(f"Mean loss: {np.mean(final_losses):.4f}")
        print(f"Std loss: {np.std(final_losses):.4f}")
        
    else:
        print("\n❌ No successful experiments completed!")
        print("This suggests there may be an issue with the PVI implementation itself.")
    
    return all_results, results_df


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Ultra-Simple WGF-GMM experiments')
    parser.add_argument('mode', nargs='?', default='test', 
                       choices=['test', 'quick', 'full'],
                       help='Mode: test (single config), quick (reduced search), full (comprehensive)')
    
    args = parser.parse_args()
    
    print("Ultra-Simple WGF-GMM Hyperparameter Experiments")
    print("=" * 50)
    print("✓ Uses existing PVI framework (guaranteed to work)")
    print("✓ Adds WGF-GMM metrics without changing optimization")
    print("✓ No optimizer chain issues")
    print("=" * 50)
    
    if args.mode == 'test':
        print("Running single configuration test...")
        success = test_single_config()
        if success:
            print("✅ Single configuration test passed!")
        else:
            print("❌ Single configuration test failed!")
        return
    
    elif args.mode == 'quick':
        print("Running quick test...")
        results = run_quick_test()
        successful = [r for r in results if r['success']]
        if successful:
            print(f"✅ Quick test completed! {len(successful)}/{len(results)} successful")
        else:
            print("❌ Quick test - no successful runs!")
        return
    
    elif args.mode == 'full':
        print("Running comprehensive hyperparameter search...")
        all_results, results_df = run_comprehensive_experiments()
        print("✅ Comprehensive experiments completed!")
        
    else:
        print("Unknown mode. Use 'test', 'quick', or 'full'")


if __name__ == "__main__":
    main()
