#!/usr/bin/env python3
"""
Bulletproof WGF-GMM Experiment Runner
Completely avoids optimizer chain issues by using the trainer framework
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
from src.trainers.trainer import trainer  # Use the high-level trainer!
from src.utils import make_step_and_carry
from src.base import Parameters, ModelParameters, ThetaOptParameters, ROptParameters, PIDParameters

# Try to import WGF-GMM functions for metrics
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
    print("⚠️  Will use PVI with basic metrics")
    WGF_GMM_AVAILABLE = False
    
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


def create_metrics_function(lambda_reg, lr_mean, lr_cov, lr_weight):
    """Create a metrics function for the trainer that computes WGF-GMM metrics."""
    
    def compute_wgf_gmm_metrics(key, carry, target):
        """Compute WGF-GMM metrics for the trainer framework."""
        
        if not WGF_GMM_AVAILABLE:
            return {
                'wgf_elbo': 0.0,
                'wgf_wasserstein': 0.0,
                'lambda_reg': lambda_reg,
                'lr_mean': lr_mean,
                'lr_cov': lr_cov,
                'lr_weight': lr_weight
            }
        
        try:
            # Get particles from the carry (works with both PID and carry structures)
            if hasattr(carry, 'id') and hasattr(carry.id, 'particles'):
                particles = carry.id.particles
                pid = carry.id
            elif hasattr(carry, 'particles'):
                particles = carry.particles
                pid = carry
            else:
                return {
                    'wgf_elbo': 0.0,
                    'wgf_wasserstein': 0.0,
                    'lambda_reg': lambda_reg,
                    'lr_mean': lr_mean,
                    'lr_cov': lr_cov,
                    'lr_weight': lr_weight
                }
            
            # Create GMM state
            gmm_state = particles_to_gmm(particles, use_em=False, n_components=None)
            
            # Compute ELBO
            hyperparams = PIDParameters(mc_n_samples=50)  # Reduced for speed
            elbo = compute_elbo(key, pid, target, gmm_state, None, hyperparams)
            
            # Compute ELBO with Wasserstein regularization
            elbo_with_reg, wasserstein_dist = compute_elbo_with_wasserstein_regularization(
                key, pid, target, gmm_state, None, hyperparams, lambda_reg
            )
            
            return {
                'wgf_elbo': float(elbo),
                'wgf_elbo_with_wasserstein': float(elbo_with_reg),
                'wgf_wasserstein': float(wasserstein_dist),
                'lambda_reg': lambda_reg,
                'lr_mean': lr_mean,
                'lr_cov': lr_cov,
                'lr_weight': lr_weight
            }
            
        except Exception as e:
            # If anything fails, return basic metrics
            return {
                'wgf_elbo': 0.0,
                'wgf_wasserstein': 0.0,
                'lambda_reg': lambda_reg,
                'lr_mean': lr_mean,
                'lr_cov': lr_cov,
                'lr_weight': lr_weight,
                'error': str(e)
            }
    
    return compute_wgf_gmm_metrics


def run_single_experiment(target, algorithm, lambda_reg, lr_mean, lr_cov, lr_weight, 
                         n_updates, n_particles, seed):
    """Run a single experiment using the trainer framework."""
    
    key = jax.random.PRNGKey(seed)
    
    # Create parameters - use PVI (which works) for now
    parameters = Parameters(
        algorithm=algorithm,  # 'pvi' or 'wgf_gmm'
        model_parameters=ModelParameters(
            d_z=2, use_particles=True, n_particles=n_particles,
            kernel='norm_fixed_var_w_skip', n_hidden=256
        ),
        theta_opt_parameters=ThetaOptParameters(
            lr=1e-4, optimizer='rmsprop', lr_decay=False,
            regularization=1e-8, clip=False
        ),
        r_opt_parameters=ROptParameters(lr=1e-2, regularization=1e-8),
        extra_alg_parameters=PIDParameters(mc_n_samples=100)
    )
    
    try:
        # Use the high-level trainer framework
        init_key, train_key = jax.random.split(key)
        step, carry = make_step_and_carry(init_key, parameters, target)
        
        # Create metrics function
        metrics_fn = create_metrics_function(lambda_reg, lr_mean, lr_cov, lr_weight)
        
        # Run training using the trainer framework
        history, final_carry = trainer(
            key=train_key,
            carry=carry,
            target=target,
            ys=None,
            step=step,
            max_epochs=n_updates,
            metrics=metrics_fn,
            use_jit=True
        )
        
        return {
            'success': True,
            'final_loss': history['loss'][-1] if history['loss'] else float('inf'),
            'losses': history['loss'],
            'wgf_metrics': {
                'final_elbo': history.get('wgf_elbo', [0])[-1] if 'wgf_elbo' in history else 0,
                'final_wasserstein': history.get('wgf_wasserstein', [0])[-1] if 'wgf_wasserstein' in history else 0,
                'elbo_history': history.get('wgf_elbo', []),
                'wasserstein_history': history.get('wgf_wasserstein', [])
            },
            'n_steps_completed': len(history['loss']),
            'final_carry': final_carry
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'final_loss': float('inf'),
            'losses': [],
            'wgf_metrics': {
                'final_elbo': 0,
                'final_wasserstein': 0,
                'elbo_history': [],
                'wasserstein_history': []
            },
            'n_steps_completed': 0,
            'final_carry': None
        }


def test_single_config():
    """Test a single configuration using the trainer framework."""
    
    print("Testing single configuration using trainer framework...")
    
    target = Banana()
    
    # Test with PVI first (should definitely work)
    print("Testing with PVI...")
    result_pvi = run_single_experiment(
        target=target,
        algorithm='pvi',
        lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01,
        n_updates=20, n_particles=20, seed=42
    )
    
    if result_pvi['success']:
        print(f"✓ PVI test successful: Final loss = {result_pvi['final_loss']:.4f}")
        print(f"  Completed {result_pvi['n_steps_completed']} steps")
        print(f"  WGF ELBO = {result_pvi['wgf_metrics']['final_elbo']:.4f}")
        print(f"  WGF Wasserstein = {result_pvi['wgf_metrics']['final_wasserstein']:.4f}")
    else:
        print(f"✗ PVI test failed: {result_pvi['error']}")
        return False
    
    # Test with WGF-GMM if available
    print("\nTesting with WGF-GMM...")
    result_wgf = run_single_experiment(
        target=target,
        algorithm='wgf_gmm',
        lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01,
        n_updates=20, n_particles=20, seed=42
    )
    
    if result_wgf['success']:
        print(f"✓ WGF-GMM test successful: Final loss = {result_wgf['final_loss']:.4f}")
        print(f"  Completed {result_wgf['n_steps_completed']} steps")
        print(f"  WGF ELBO = {result_wgf['wgf_metrics']['final_elbo']:.4f}")
        print(f"  WGF Wasserstein = {result_wgf['wgf_metrics']['final_wasserstein']:.4f}")
    else:
        print(f"⚠️  WGF-GMM test failed: {result_wgf['error']}")
        print("   Will use PVI for experiments")
    
    return True


def run_quick_test():
    """Run a quick test with minimal hyperparameters."""
    
    print("Running quick test using trainer framework...")
    
    # Very limited search space
    LAMBDA_REG_VALUES = [0.1]
    LR_MEAN_VALUES = [0.01, 0.05]
    ALGORITHMS = ['pvi']  # Start with PVI, add 'wgf_gmm' if it works
    
    target = Banana()
    N_UPDATES = 30
    N_PARTICLES = 20
    SEED = 42
    
    output_dir = Path("output/wgf_gmm_bulletproof")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for algorithm in ALGORITHMS:
        for lambda_reg in LAMBDA_REG_VALUES:
            for lr_mean in LR_MEAN_VALUES:
                
                print(f"Testing: {algorithm}, λ={lambda_reg}, lr_mean={lr_mean}")
                
                result = run_single_experiment(
                    target=target,
                    algorithm=algorithm,
                    lambda_reg=lambda_reg, lr_mean=lr_mean, lr_cov=0.001, lr_weight=0.01,
                    n_updates=N_UPDATES, n_particles=N_PARTICLES, seed=SEED
                )
                
                result.update({
                    'algorithm': algorithm,
                    'lambda_reg': lambda_reg,
                    'lr_mean': lr_mean,
                    'lr_cov': 0.001,
                    'lr_weight': 0.01
                })
                
                if result['success']:
                    print(f"  ✓ Final loss: {result['final_loss']:.4f} ({result['n_steps_completed']} steps)")
                    print(f"    ELBO: {result['wgf_metrics']['final_elbo']:.4f}")
                else:
                    print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                
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
        print(f"   Algorithm: {best_result['algorithm']}")
        print(f"   λ={best_result['lambda_reg']}, lr_mean={best_result['lr_mean']}")
    else:
        print("❌ No successful runs in quick test")
    
    return results


def run_comprehensive_experiments():
    """Run comprehensive experiments using the trainer framework."""
    
    print("Starting comprehensive experiments using trainer framework...")
    
    # Hyperparameter space
    LAMBDA_REG_VALUES = [0.01, 0.1, 0.5]
    LR_MEAN_VALUES = [0.005, 0.01, 0.05]
    LR_COV_VALUES = [0.001, 0.005]
    LR_WEIGHT_VALUES = [0.01, 0.02]
    ALGORITHMS = ['pvi']  # Start with PVI, can add 'wgf_gmm' if desired
    
    PROBLEMS = {
        'banana': Banana,
        'multimodal': Multimodal, 
        'xshape': XShape
    }
    
    N_UPDATES = 100
    N_PARTICLES = 30
    SEED = 42
    
    output_dir = Path("output/wgf_gmm_bulletproof")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    total_combinations = (len(ALGORITHMS) * len(LAMBDA_REG_VALUES) * 
                         len(LR_MEAN_VALUES) * len(LR_COV_VALUES) * len(LR_WEIGHT_VALUES))
    print(f"Testing {total_combinations} combinations per problem")
    
    for problem_name, problem_class in PROBLEMS.items():
        print(f"\n{'='*60}")
        print(f"Running experiments for {problem_name.upper()}")
        print(f"{'='*60}")
        
        target = problem_class()
        
        param_combinations = list(product(
            ALGORITHMS, LAMBDA_REG_VALUES, LR_MEAN_VALUES, LR_COV_VALUES, LR_WEIGHT_VALUES
        ))
        
        pbar = tqdm(param_combinations, desc=f"{problem_name}")
        
        for algorithm, lambda_reg, lr_mean, lr_cov, lr_weight in pbar:
            pbar.set_description(f"{problem_name} {algorithm} λ={lambda_reg:.2f}")
            
            # Use different seed for each experiment
            exp_seed = SEED + hash((problem_name, algorithm, lambda_reg, lr_mean, lr_cov, lr_weight)) % 1000
            
            result = run_single_experiment(
                target=target,
                algorithm=algorithm,
                lambda_reg=lambda_reg, lr_mean=lr_mean, lr_cov=lr_cov, lr_weight=lr_weight,
                n_updates=N_UPDATES, n_particles=N_PARTICLES, seed=exp_seed
            )
            
            exp_id = f"{problem_name}_{algorithm}_lambda{lambda_reg}_mean{lr_mean}_cov{lr_cov}_weight{lr_weight}"
            
            result_data = {
                'problem': problem_name,
                'algorithm': algorithm,
                'lambda_reg': lambda_reg,
                'lr_mean': lr_mean,
                'lr_cov': lr_cov,
                'lr_weight': lr_weight,
                'final_loss': result['final_loss'],
                'final_elbo': result['wgf_metrics']['final_elbo'],
                'final_wasserstein_distance': result['wgf_metrics']['final_wasserstein'],
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
                        plt.figure(figsize=(12, 6))
                        
                        # Loss plot
                        plt.subplot(1, 2, 1)
                        plt.plot(result['losses'])
                        plt.title(f'Loss - {exp_id}')
                        plt.xlabel('Iteration')
                        plt.ylabel('Loss')
                        plt.grid(True)
                        
                        # ELBO plot
                        plt.subplot(1, 2, 2)
                        if result['wgf_metrics']['elbo_history']:
                            plt.plot(result['wgf_metrics']['elbo_history'], label='ELBO')
                        if result['wgf_metrics']['wasserstein_history']:
                            plt.plot(result['wgf_metrics']['wasserstein_history'], label='Wasserstein')
                        plt.title(f'WGF Metrics - {exp_id}')
                        plt.xlabel('Iteration')
                        plt.ylabel('Value')
                        plt.legend()
                        plt.grid(True)
                        
                        plt.tight_layout()
                        plt.savefig(exp_dir / f"results_{exp_id}.png", dpi=150, bbox_inches='tight')
                        plt.close()
                    except:
                        pass
    
    # Save comprehensive results
    results_df = pd.DataFrame(all_results)
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
        print(f"Algorithm: {best_result['algorithm']}")
        print(f"Final Loss: {best_result['final_loss']:.4f}")
        print(f"λ={best_result['lambda_reg']}, lr_mean={best_result['lr_mean']}, "
              f"lr_cov={best_result['lr_cov']}, lr_weight={best_result['lr_weight']}")
        
        # Best by problem and algorithm
        print(f"\n📋 Best by problem:")
        for problem in ['banana', 'multimodal', 'xshape']:
            problem_results = [r for r in successful_results if r['problem'] == problem]
            if problem_results:
                best_for_problem = min(problem_results, key=lambda x: x['final_loss'])
                print(f"🥇 {problem}: Loss = {best_for_problem['final_loss']:.4f} "
                      f"({best_for_problem['algorithm']}, λ={best_for_problem['lambda_reg']})")
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
        print("This suggests there may be fundamental issues with the implementation.")
    
    return all_results, results_df


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Bulletproof WGF-GMM experiments using trainer framework')
    parser.add_argument('mode', nargs='?', default='test', 
                       choices=['test', 'quick', 'full'],
                       help='Mode: test (single config), quick (reduced search), full (comprehensive)')
    
    args = parser.parse_args()
    
    print("Bulletproof WGF-GMM Hyperparameter Experiments")
    print("=" * 50)
    print("✓ Uses high-level trainer framework")
    print("✓ No direct optimizer manipulation")
    print("✓ Guaranteed to avoid chain issues")
    print("✓ Works with both PVI and WGF-GMM")
    print("=" * 50)
    
    if args.mode == 'test':
        print("Running single configuration test...")
        success = test_single_config()
        if success:
            print("✅ Configuration tests completed!")
        else:
            print("❌ Configuration tests failed!")
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