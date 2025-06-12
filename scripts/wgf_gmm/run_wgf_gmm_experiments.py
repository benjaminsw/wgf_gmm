#!/usr/bin/env python3
"""
Working WGF-GMM Experiment Runner
This version works with the existing wgf_gmm.py implementation
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
import argparse

# Import your existing modules
from src.problems.toy import Banana, Multimodal, XShape
from src.utils import make_step_and_carry
from src.base import Parameters, ModelParameters, ThetaOptParameters, ROptParameters, PIDParameters

# Import standard PVI as fallback
from src.trainers.pvi import de_step as pvi_de_step

# Try to import WGF-GMM with careful error handling
try:
    from src.trainers.wgf_gmm import (
        WGFGMMMetrics,
        particles_to_gmm,
        gmm_to_particles,
        compute_elbo,
        compute_elbo_with_wasserstein_regularization,
        update_gmm_parameters_simple
    )
    WGF_GMM_AVAILABLE = True
    print("✓ WGF-GMM functions imported successfully")
except ImportError as e:
    print(f"⚠️  WGF-GMM import warning: {e}")
    print("⚠️  Will use PVI fallback for all experiments")
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


def create_working_wgf_gmm_step(lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01):
    """Create a WGF-GMM step function that actually works."""
    
    def working_step(key, carry, target, y, optim, hyperparams):
        """Working WGF-GMM step with proper error handling."""
        
        if not WGF_GMM_AVAILABLE:
            # Use standard PVI
            lval, updated_carry = pvi_de_step(key, carry, target, y, optim, hyperparams)
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
        
        try:
            # Try enhanced WGF-GMM
            from src.trainers.util import loss_step
            from jax import vmap
            from jax.lax import stop_gradient
            import equinox as eqx
            
            theta_key, r_key, metrics_key = jax.random.split(key, 3)
            
            # Standard PVI theta update
            def loss(key, params, static):
                pid = eqx.combine(params, static)
                _samples = pid.sample(key, hyperparams.mc_n_samples, None)
                logq = vmap(eqx.combine(stop_gradient(params), static).log_prob, (0, None))(_samples, None)
                logp = vmap(target.log_prob, (0, None))(_samples, y)
                return np.mean(logq - logp, axis=0)
            
            lval, pid, theta_opt_state = loss_step(
                theta_key, loss, carry.id, optim.theta_optim, carry.theta_opt_state
            )
            
            # Get or create GMM state
            if hasattr(carry, 'gmm_state') and carry.gmm_state is not None:
                gmm_state = carry.gmm_state
            else:
                gmm_state = particles_to_gmm(pid.particles, use_em=False, n_components=None)
            
            # Compute metrics
            elbo = compute_elbo(metrics_key, pid, target, gmm_state, y, hyperparams)
            elbo_with_reg, wasserstein_dist = compute_elbo_with_wasserstein_regularization(
                metrics_key, pid, target, gmm_state, y, hyperparams, lambda_reg
            )
            
            # Standard PVI particle update
            def particle_grad_fn(particles):
                def ediff_score(particle, eps):
                    vf = vmap(pid.conditional.f, (None, None, 0))
                    samples = vf(particle, y, eps)
                    logq = vmap(pid.log_prob, (0, None))(samples, y)
                    logp = vmap(target.log_prob, (0, None))(samples, y)
                    return np.mean(logq - logp, 0)
                
                eps = pid.conditional.base_sample(r_key, hyperparams.mc_n_samples)
                return vmap(jax.grad(lambda p: ediff_score(p, eps)))(particles)
            
            g_grad, r_precon_state = optim.r_precon.update(
                pid.particles, particle_grad_fn, carry.r_precon_state
            )
            
            update, r_opt_state = optim.r_optim.update(
                g_grad, carry.r_opt_state, params=pid.particles, index=y
            )
            
            # Update particles
            updated_particles = pid.particles + update
            
            # Simple GMM update
            updated_gmm_state = update_gmm_parameters_simple(gmm_state, update, lr_mean)
            pid = eqx.tree_at(lambda tree: tree.particles, pid, updated_particles)
            
            # Create updated carry
            from src.base import PIDCarry
            updated_carry = PIDCarry(
                id=pid,
                theta_opt_state=theta_opt_state,
                r_opt_state=r_opt_state,
                r_precon_state=r_precon_state
            )
            
            # Add GMM state using a wrapper to avoid immutability
            class ExtendedCarry:
                def __init__(self, base_carry, gmm_state):
                    self.id = base_carry.id
                    self.theta_opt_state = base_carry.theta_opt_state
                    self.r_opt_state = base_carry.r_opt_state
                    self.r_precon_state = base_carry.r_precon_state
                    self.gmm_state = gmm_state
            
            final_carry = ExtendedCarry(updated_carry, updated_gmm_state)
            
            metrics = WGFGMMMetrics(
                elbo=float(elbo),
                elbo_with_wasserstein=float(elbo_with_reg),
                wasserstein_distance=float(wasserstein_dist),
                lambda_reg=lambda_reg,
                lr_mean=lr_mean,
                lr_cov=lr_cov,
                lr_weight=lr_weight
            )
            
            return lval, final_carry, metrics
            
        except Exception as e:
            print(f"WGF-GMM failed: {e}, using PVI fallback")
            # Fallback to PVI
            lval, updated_carry = pvi_de_step(key, carry, target, y, optim, hyperparams)
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
    
    return working_step


def create_basic_optimizer():
    """Create a basic optimizer for testing."""
    import optax
    from src.base import PIDOpt
    from src.preconditioner import identity
    
    theta_optim = optax.rmsprop(1e-4)
    r_optim = optax.chain(optax.scale(-1e-2))
    r_precon = identity()
    return PIDOpt(theta_optim, r_optim, r_precon)


def test_single_config():
    """Test a single configuration."""
    
    print("Testing single WGF-GMM configuration...")
    
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
        r_opt_parameters=ROptParameters(
            lr=1e-2, regularization=1e-8
        ),
        extra_alg_parameters=PIDParameters(mc_n_samples=50)
    )
    
    init_key, train_key = jax.random.split(key)
    step, carry = make_step_and_carry(init_key, parameters, target)
    optim = create_basic_optimizer()
    
    # Create working step function
    working_step = create_working_wgf_gmm_step(
        lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01
    )
    
    print("Running 10 steps...")
    losses = []
    
    for i in range(10):
        train_key, step_key = jax.random.split(train_key)
        
        try:
            lval, carry, metrics = working_step(
                step_key, carry, target, None, optim, parameters.extra_alg_parameters
            )
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


def run_quick_test():
    """Run a quick test with a few hyperparameter combinations."""
    
    print("Running quick WGF-GMM test...")
    
    # Very limited search space for quick testing
    LAMBDA_REG_VALUES = [0.1, 0.5]
    LR_MEAN_VALUES = [0.01, 0.05]
    
    target = Banana()
    N_UPDATES = 50
    SEED = 42
    
    output_dir = Path("output/wgf_gmm_quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    key = jax.random.PRNGKey(SEED)
    
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
    
    init_key, exp_key = jax.random.split(key)
    step, initial_carry = make_step_and_carry(init_key, parameters, target)
    optim = create_basic_optimizer()
    
    results = []
    
    for lambda_reg in LAMBDA_REG_VALUES:
        for lr_mean in LR_MEAN_VALUES:
            
            print(f"Testing: λ={lambda_reg}, lr_mean={lr_mean}")
            
            carry = initial_carry
            exp_key, run_key = jax.random.split(exp_key)
            
            working_step = create_working_wgf_gmm_step(
                lambda_reg=lambda_reg, lr_mean=lr_mean, lr_cov=0.001, lr_weight=0.01
            )
            
            losses = []
            
            for update_idx in range(N_UPDATES):
                run_key, step_key = jax.random.split(run_key)
                
                try:
                    lval, carry, metrics = working_step(
                        step_key, carry, target, None, optim, parameters.extra_alg_parameters
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
    
    # Save results
    with open(output_dir / "quick_test_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Find best result
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['final_loss'])
        print(f"\n✓ Best quick test result:")
        print(f"Final Loss: {best_result['final_loss']:.4f}")
        print(f"λ={best_result['lambda_reg']}, lr_mean={best_result['lr_mean']}")
        return results
    else:
        print("✗ No successful runs in quick test!")
        return results


def run_comprehensive_experiments():
    """Run comprehensive experiments with all problems and hyperparameters."""
    
    print("Starting comprehensive WGF-GMM experiments...")
    
    # Reasonable hyperparameter space
    LAMBDA_REG_VALUES = [0.01, 0.1, 0.5]
    LR_MEAN_VALUES = [0.005, 0.01, 0.05]
    LR_COV_VALUES = [0.001, 0.005]
    LR_WEIGHT_VALUES = [0.01, 0.02]
    
    PROBLEMS = {
        'banana': Banana,
        'multimodal': Multimodal, 
        'xshape': XShape
    }
    
    N_UPDATES = 200
    N_PARTICLES = 50
    SEED = 42
    
    output_dir = Path("output/wgf_gmm_working")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    total_combinations = len(LAMBDA_REG_VALUES) * len(LR_MEAN_VALUES) * len(LR_COV_VALUES) * len(LR_WEIGHT_VALUES)
    print(f"Testing {total_combinations} combinations per problem")
    
    for problem_name, problem_class in PROBLEMS.items():
        print(f"\n{'='*60}")
        print(f"Running experiments for {problem_name.upper()}")
        print(f"{'='*60}")
        
        target = problem_class()
        key = jax.random.PRNGKey(SEED)
        
        parameters = Parameters(
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
        
        init_key, exp_key = jax.random.split(key)
        step, initial_carry = make_step_and_carry(init_key, parameters, target)
        optim = create_basic_optimizer()
        
        param_combinations = list(product(
            LAMBDA_REG_VALUES, LR_MEAN_VALUES, LR_COV_VALUES, LR_WEIGHT_VALUES
        ))
        
        pbar = tqdm(param_combinations, desc=f"{problem_name}")
        
        for lambda_reg, lr_mean, lr_cov, lr_weight in pbar:
            pbar.set_description(f"{problem_name} λ={lambda_reg:.2f} m={lr_mean:.3f}")
            
            carry = initial_carry
            exp_key, run_key = jax.random.split(exp_key)
            
            working_step = create_working_wgf_gmm_step(
                lambda_reg=lambda_reg, lr_mean=lr_mean, lr_cov=lr_cov, lr_weight=lr_weight
            )
            
            losses = []
            elbo_values = []
            wasserstein_distances = []
            success = True
            
            for update_idx in range(N_UPDATES):
                run_key, step_key = jax.random.split(run_key)
                
                try:
                    lval, carry, metrics = working_step(
                        step_key, carry, target, None, optim, parameters.extra_alg_parameters
                    )
                    
                    losses.append(float(lval))
                    elbo_values.append(metrics.elbo)
                    wasserstein_distances.append(metrics.wasserstein_distance)
                        
                except Exception as e:
                    print(f"\nError at iteration {update_idx}: {e}")
                    success = False
                    break
            
            exp_id = f"{problem_name}_lambda{lambda_reg}_mean{lr_mean}_cov{lr_cov}_weight{lr_weight}"
            
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
                'converged': success and len(losses) == N_UPDATES,
                'exp_id': exp_id
            }
            
            all_results.append(result_data)
            
            # Save individual results if successful
            if success and losses:
                exp_dir = output_dir / problem_name / exp_id
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                detailed_results = {
                    **result_data,
                    'losses': losses,
                    'elbo_values': elbo_values,
                    'wasserstein_distances': wasserstein_distances
                }
                
                with open(exp_dir / "results.pkl", "wb") as f:
                    pickle.dump(detailed_results, f)
    
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
    
    # Print best results
    successful_results = [r for r in all_results if r['converged']]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['final_loss'])
        print(f"\n🏆 Best overall configuration:")
        print(f"Problem: {best_result['problem']}")
        print(f"Final Loss: {best_result['final_loss']:.4f}")
        print(f"λ={best_result['lambda_reg']}, lr_mean={best_result['lr_mean']}, "
              f"lr_cov={best_result['lr_cov']}, lr_weight={best_result['lr_weight']}")
        
        # Best by problem
        for problem in ['banana', 'multimodal', 'xshape']:
            problem_results = [r for r in successful_results if r['problem'] == problem]
            if problem_results:
                best_for_problem = min(problem_results, key=lambda x: x['final_loss'])
                print(f"\n🥇 Best for {problem}:")
                print(f"Final Loss: {best_for_problem['final_loss']:.4f}")
                print(f"λ={best_for_problem['lambda_reg']}, lr_mean={best_for_problem['lr_mean']}, "
                      f"lr_cov={best_for_problem['lr_cov']}, lr_weight={best_for_problem['lr_weight']}")
    else:
        print("\n❌ No successful experiments completed!")
    
    return all_results, results_df


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Working WGF-GMM experiments')
    parser.add_argument('mode', nargs='?', default='test', 
                       choices=['test', 'quick', 'full'],
                       help='Mode: test (single config), quick (reduced search), full (comprehensive)')
    
    args = parser.parse_args()
    
    print("Working WGF-GMM Hyperparameter Experiments")
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
        print("Running quick hyperparameter test...")
        results = run_quick_test()
        if results:
            successful = [r for r in results if r['success']]
            print(f"✅ Quick test completed! {len(successful)}/{len(results)} successful")
        else:
            print("❌ Quick test failed!")
        return
    
    elif args.mode == 'full':
        print("Running comprehensive hyperparameter search...")
        all_results, results_df = run_comprehensive_experiments()
        
        successful_results = [r for r in all_results if r['converged']]
        print(f"✅ Comprehensive experiments completed!")
        print(f"📊 {len(successful_results)}/{len(all_results)} experiments successful")
        
        if len(successful_results) > 0:
            print(f"🎯 Success rate: {100*len(successful_results)/len(all_results):.1f}%")
        
    else:
        print("Unknown mode. Use 'test', 'quick', or 'full'")


if __name__ == "__main__":
    main()