#!/usr/bin/env python3
"""
WGF-GMM Experiment Runner
Integrates the enhanced WGF-GMM with hyperparameter search into the existing experiment framework.
"""

import jax
import jax.numpy as np
from pathlib import Path
from tqdm import tqdm
import typer
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from src.problems.toy import Banana, Multimodal, XShape
from src.utils import make_step_and_carry, config_to_parameters, parse_config
from src.base import Parameters, ModelParameters, ThetaOptParameters, ROptParameters, PIDParameters
from wgf_gmm_hyperparam_search import (
    run_wgf_gmm_hyperparameter_search,
    wgf_gmm_pvi_step_with_monitoring,
    WGFGMMMetrics,
    create_wgf_gmm_config_file
)

app = typer.Typer()

PROBLEMS = {
    'banana': Banana,
    'multimodal': Multimodal,
    'xshape': XShape,
}

# Recommended hyperparameter grids based on your analysis
RECOMMENDED_HYPERPARAMS = {
    'conservative': {
        'lambda_reg_values': [0.05, 0.1, 0.2],
        'lr_mean_values': [0.005, 0.01],
        'lr_cov_values': [0.0005, 0.001],
        'lr_weight_values': [0.005, 0.01]
    },
    'aggressive': {
        'lambda_reg_values': [0.01, 0.05, 0.1, 0.5, 1.0],
        'lr_mean_values': [0.01, 0.05, 0.1],
        'lr_cov_values': [0.001, 0.005, 0.01],
        'lr_weight_values': [0.01, 0.02, 0.05]
    },
    'focused': {
        'lambda_reg_values': [0.1],
        'lr_mean_values': [0.01, 0.05],
        'lr_cov_values': [0.001, 0.005],
        'lr_weight_values': [0.01, 0.02]
    }
}


@app.command()
def run_single_experiment(
    problem_name: str = 'banana',
    lambda_reg: float = 0.1,
    lr_mean: float = 0.01,
    lr_cov: float = 0.001,
    lr_weight: float = 0.01,
    n_updates: int = 1000,
    n_particles: int = 100,
    seed: int = 42,
    output_dir: str = "output/wgf_gmm_single"
):
    """
    Run a single WGF-GMM experiment with specified hyperparameters.
    """
    print(f"Running WGF-GMM experiment: {problem_name}")
    print(f"Hyperparameters: λ={lambda_reg}, lr_mean={lr_mean}, lr_cov={lr_cov}, lr_weight={lr_weight}")
    
    # Setup
    key = jax.random.PRNGKey(seed)
    target = PROBLEMS[problem_name]()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create parameters
    parameters = Parameters(
        algorithm='wgf_gmm',
        model_parameters=ModelParameters(
            d_z=2,
            use_particles=True,
            n_particles=n_particles,
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
    
    # Initialize
    init_key, train_key = jax.random.split(key)
    step, carry = make_step_and_carry(init_key, parameters, target)
    
    # Training loop with monitoring
    metrics_history = []
    losses = []
    
    pbar = tqdm(range(n_updates), desc="Training")
    for update_idx in pbar:
        train_key, step_key = jax.random.split(train_key)
        
        # Use the monitoring version of the step function
        lval, carry, metrics = wgf_gmm_pvi_step_with_monitoring(
            step_key, carry, target, None, step.keywords['optim'], 
            parameters.extra_alg_parameters,
            lambda_reg=lambda_reg, lr_mean=lr_mean, lr_cov=lr_cov, lr_weight=lr_weight
        )
        
        losses.append(float(lval))
        metrics_history.append(metrics)
        
        if (update_idx + 1) % 100 == 0:
            pbar.set_postfix({
                'Loss': f'{lval:.4f}',
                'ELBO': f'{metrics.elbo:.4f}',
                'W_dist': f'{metrics.wasserstein_distance:.4f}'
            })
    
    # Save results
    exp_id = f"{problem_name}_lambda{lambda_reg}_mean{lr_mean}_cov{lr_cov}_weight{lr_weight}"
    
    with open(output_path / f"metrics_{exp_id}.pkl", "wb") as f:
        pickle.dump(metrics_history, f)
    
    with open(output_path / f"losses_{exp_id}.pkl", "wb") as f:
        pickle.dump(losses, f)
    
    # Plot results
    plot_single_experiment_results(metrics_history, losses, output_path, exp_id)
    
    print(f"Results saved to: {output_path}")
    print(f"Final ELBO: {metrics_history[-1].elbo:.4f}")
    print(f"Final ELBO with Wasserstein: {metrics_history[-1].elbo_with_wasserstein:.4f}")


@app.command()
def run_hyperparameter_search(
    problem_name: str = 'banana',
    search_type: str = 'conservative',
    n_updates: int = 1000,
    n_particles: int = 100,
    seed: int = 42,
    output_dir: str = "output/wgf_gmm_search"
):
    """
    Run comprehensive hyperparameter search for WGF-GMM.
    
    Args:
        problem_name: Which problem to run ('banana', 'multimodal', 'xshape')
        search_type: Type of search ('conservative', 'aggressive', 'focused')
        n_updates: Number of training updates per experiment
        n_particles: Number of particles
        seed: Random seed
        output_dir: Output directory
    """
    print(f"Running WGF-GMM hyperparameter search: {problem_name}")
    print(f"Search type: {search_type}")
    
    if search_type not in RECOMMENDED_HYPERPARAMS:
        raise ValueError(f"Unknown search type: {search_type}. Choose from: {list(RECOMMENDED_HYPERPARAMS.keys())}")
    
    # Setup
    key = jax.random.PRNGKey(seed)
    target = PROBLEMS[problem_name]()
    output_path = Path(output_dir) / problem_name / search_type
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create parameters
    parameters = Parameters(
        algorithm='wgf_gmm',
        model_parameters=ModelParameters(
            d_z=2,
            use_particles=True,
            n_particles=n_particles,
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
    init_key, _ = jax.random.split(key)
    step, initial_carry = make_step_and_carry(init_key, parameters, target)
    
    # Get hyperparameter grid
    hyperparam_grid = RECOMMENDED_HYPERPARAMS[search_type]
    
    # Run search
    results_df = run_wgf_gmm_hyperparameter_search(
        target=target,
        initial_carry=initial_carry,
        optim=step.keywords['optim'],
        hyperparams=parameters.extra_alg_parameters,
        n_updates=n_updates,
        output_dir=output_path,
        seed=seed,
        **hyperparam_grid
    )
    
    print(f"Hyperparameter search completed!")
    print(f"Results saved to: {output_path}")
    
    # Extract best configurations
    best_configs = extract_best_configurations(results_df)
    
    # Create config files for best settings
    create_wgf_gmm_config_file(output_path, best_configs)
    
    return results_df


@app.command()
def run_comparison_study(
    n_updates: int = 1000,
    n_reruns: int = 5,
    seed: int = 42,
    output_dir: str = "output/wgf_gmm_comparison"
):
    """
    Run a comparison study across all problems with recommended hyperparameters.
    """
    print("Running WGF-GMM comparison study across all problems")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Best hyperparameters found from previous experiments (you can update these)
    best_hyperparams = {
        'banana': {'lambda_reg': 0.1, 'lr_mean': 0.01, 'lr_cov': 0.001, 'lr_weight': 0.01},
        'multimodal': {'lambda_reg': 0.05, 'lr_mean': 0.05, 'lr_cov': 0.005, 'lr_weight': 0.02},
        'xshape': {'lambda_reg': 0.1, 'lr_mean': 0.01, 'lr_cov': 0.001, 'lr_weight': 0.01}
    }
    
    all_results = defaultdict(list)
    
    for problem_name, problem_class in PROBLEMS.items():
        print(f"\nRunning {problem_name} experiments...")
        
        hyperparams = best_hyperparams.get(problem_name, best_hyperparams['banana'])
        
        for run_idx in range(n_reruns):
            print(f"  Run {run_idx + 1}/{n_reruns}")
            
            # Setup for this run
            run_seed = seed + run_idx
            key = jax.random.PRNGKey(run_seed)
            target = problem_class()
            
            # Create parameters
            parameters = Parameters(
                algorithm='wgf_gmm',
                model_parameters=ModelParameters(
                    d_z=2,
                    use_particles=True,
                    n_particles=100,
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
            
            # Initialize
            init_key, train_key = jax.random.split(key)
            step, carry = make_step_and_carry(init_key, parameters, target)
            
            # Training
            metrics_history = []
            losses = []
            
            for update_idx in range(n_updates):
                train_key, step_key = jax.random.split(train_key)
                
                lval, carry, metrics = wgf_gmm_pvi_step_with_monitoring(
                    step_key, carry, target, None, step.keywords['optim'],
                    parameters.extra_alg_parameters,
                    **hyperparams
                )
                
                losses.append(float(lval))
                metrics_history.append(metrics)
            
            # Store results
            all_results[problem_name].append({
                'run_idx': run_idx,
                'final_elbo': metrics_history[-1].elbo,
                'final_elbo_with_wasserstein': metrics_history[-1].elbo_with_wasserstein,
                'final_wasserstein_distance': metrics_history[-1].wasserstein_distance,
                'final_loss': losses[-1],
                'metrics_history': metrics_history,
                'losses': losses
            })
    
    # Save and analyze results
    comparison_results = analyze_comparison_results(all_results, output_path)
    plot_comparison_results(comparison_results, output_path)
    
    print(f"Comparison study completed! Results saved to: {output_path}")
    
    return comparison_results


def extract_best_configurations(results_df):
    """Extract the best configurations based on different metrics."""
    best_configs = {
        'best_elbo': results_df.loc[results_df['final_elbo'].idxmax()].to_dict(),
        'best_elbo_with_wasserstein': results_df.loc[results_df['final_elbo_with_wasserstein'].idxmax()].to_dict(),
        'lowest_loss': results_df.loc[results_df['final_loss'].idxmin()].to_dict(),
    }
    
    # Create balance score and find best
    results_df['balance_score'] = results_df['final_elbo'] - 0.1 * results_df['final_wasserstein_distance']
    best_configs['best_balance'] = results_df.loc[results_df['balance_score'].idxmax()].to_dict()
    
    return best_configs


def plot_single_experiment_results(metrics_history, losses, output_dir, exp_id):
    """Plot results for a single experiment."""
    elbo_values = [m.elbo for m in metrics_history]
    elbo_with_wasserstein = [m.elbo_with_wasserstein for m in metrics_history]
    wasserstein_distances = [m.wasserstein_distance for m in metrics_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(losses, color='purple', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # ELBO
    axes[0, 1].plot(elbo_values, color='blue', linewidth=2)
    axes[0, 1].set_title('ELBO')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('ELBO')
    axes[0, 1].grid(True)
    
    # ELBO with Wasserstein
    axes[1, 0].plot(elbo_with_wasserstein, color='red', linewidth=2)
    axes[1, 0].set_title('ELBO with Wasserstein Regularization')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('ELBO with Regularization')
    axes[1, 0].grid(True)
    
    # Wasserstein Distance
    axes[1, 1].plot(wasserstein_distances, color='green', linewidth=2)
    axes[1, 1].set_title('Wasserstein Distance')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Wasserstein Distance')
    axes[1, 1].grid(True)
    
    plt.suptitle(f'WGF-GMM Training Results - {exp_id}', fontsize=16)
    plt.tight_layout()
    plt.savefig(