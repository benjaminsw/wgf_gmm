import jax
from jax import vmap, grad
import jax.numpy as np
import equinox as eqx
from jax.lax import stop_gradient
from src.id import PID
from src.trainers.util import loss_step
from typing import Tuple, NamedTuple
from src.base import (Target,
                      PIDCarry,
                      PIDOpt,
                      PIDParameters)
from jaxtyping import PyTree
from jax.lax import map
import jax.scipy as jsp
from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pickle
from collections import defaultdict


class GMMComponent(NamedTuple):
    """Represents a single Gaussian component with mean and covariance"""
    mean: jax.Array  # Shape: (d_z,)
    cov: jax.Array   # Shape: (d_z, d_z)
    weight: float    # Scalar weight


class GMMState(NamedTuple):
    """State for GMM-based particle representation"""
    components: list[GMMComponent]  # List of Gaussian components
    n_components: int
    prev_components: list[GMMComponent] = None  # For Wasserstein regularization


class WGFGMMMetrics(NamedTuple):
    """Metrics for WGF-GMM training"""
    elbo: float
    elbo_with_wasserstein: float
    wasserstein_distance: float
    lambda_reg: float
    lr_mean: float
    lr_cov: float
    lr_weight: float


# [Include all the existing GMM functions from your original code]
def particles_to_gmm(particles: jax.Array, 
                     weights: jax.Array = None,
                     use_em: bool = True,
                     n_components: int = None) -> GMMState:
    """Convert particle representation to GMM representation."""
    n_particles, d_z = particles.shape
    
    if weights is None:
        weights = np.ones(n_particles) / n_particles
    
    if not use_em or n_components is None:
        n_components = n_particles
    
    if use_em and n_components < n_particles:
        return _fit_gmm_em(particles, weights, n_components)
    else:
        components = []
        for i in range(n_particles):
            mean = particles[i]
            cov = np.eye(d_z) * 0.1
            weight = weights[i]
            components.append(GMMComponent(mean=mean, cov=cov, weight=weight))
        
        return GMMState(components=components, n_components=n_particles)


def _fit_gmm_em(particles: jax.Array, weights: jax.Array, n_components: int, 
                max_iter: int = 50, tol: float = 1e-6) -> GMMState:
    """Fit GMM to particles using EM algorithm."""
    n_particles, d_z = particles.shape
    key = jax.random.PRNGKey(42)
    
    means = _kmeans_plus_plus_init(key, particles, n_components)
    covs = np.stack([np.eye(d_z) * 0.5 for _ in range(n_components)])
    gmm_weights = np.ones(n_components) / n_components
    
    prev_log_likelihood = -np.inf
    
    for iter_idx in range(max_iter):
        responsibilities = _e_step(particles, weights, means, covs, gmm_weights)
        means, covs, gmm_weights = _m_step(particles, weights, responsibilities)
        
        log_likelihood = _compute_log_likelihood(particles, weights, means, covs, gmm_weights)
        if np.abs(log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = log_likelihood
    
    components = []
    for i in range(n_components):
        components.append(GMMComponent(
            mean=means[i],
            cov=covs[i],
            weight=gmm_weights[i]
        ))
    
    return GMMState(components=components, n_components=n_components)


def _kmeans_plus_plus_init(key: jax.random.PRNGKey, particles: jax.Array, 
                          n_components: int) -> jax.Array:
    """K-means++ initialization for GMM means."""
    n_particles, d_z = particles.shape
    means = np.zeros((n_components, d_z))
    
    key, subkey = jax.random.split(key)
    first_idx = jax.random.randint(subkey, (), 0, n_particles)
    means = means.at[0].set(particles[first_idx])
    
    for i in range(1, n_components):
        distances = np.full(n_particles, np.inf)
        for j in range(i):
            dist_to_j = np.sum((particles - means[j]) ** 2, axis=1)
            distances = np.minimum(distances, dist_to_j)
        
        key, subkey = jax.random.split(key)
        probs = distances / np.sum(distances)
        next_idx = jax.random.categorical(subkey, np.log(probs))
        means = means.at[i].set(particles[next_idx])
    
    return means


def _e_step(particles: jax.Array, weights: jax.Array, means: jax.Array, 
           covs: jax.Array, gmm_weights: jax.Array) -> jax.Array:
    """E-step of EM algorithm."""
    n_particles, d_z = particles.shape
    n_components = means.shape[0]
    
    log_probs = np.zeros((n_particles, n_components))
    for i in range(n_components):
        diff = particles - means[i]
        cov_inv = np.linalg.inv(covs[i] + 1e-6 * np.eye(d_z))
        mahal_dist = np.sum(diff @ cov_inv * diff, axis=1)
        log_det = np.linalg.slogdet(covs[i] + 1e-6 * np.eye(d_z))[1]
        log_probs = log_probs.at[:, i].set(
            np.log(gmm_weights[i]) - 0.5 * (d_z * np.log(2 * np.pi) + log_det + mahal_dist)
        )
    
    log_sum = jsp.special.logsumexp(log_probs, axis=1, keepdims=True)
    responsibilities = np.exp(log_probs - log_sum)
    
    return responsibilities


def _m_step(particles: jax.Array, weights: jax.Array, 
           responsibilities: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """M-step of EM algorithm."""
    n_particles, d_z = particles.shape
    n_components = responsibilities.shape[1]
    
    weighted_resp = responsibilities * weights[:, None]
    nk = np.sum(weighted_resp, axis=0)
    
    means = np.zeros((n_components, d_z))
    for k in range(n_components):
        if nk[k] > 1e-8:
            means = means.at[k].set(np.sum(weighted_resp[:, k:k+1] * particles, axis=0) / nk[k])
    
    covs = np.zeros((n_components, d_z, d_z))
    for k in range(n_components):
        if nk[k] > 1e-8:
            diff = particles - means[k]
            weighted_diff = weighted_resp[:, k:k+1] * diff
            cov = (weighted_diff.T @ diff) / nk[k]
            cov = cov + 1e-6 * np.eye(d_z)
            covs = covs.at[k].set(cov)
        else:
            covs = covs.at[k].set(np.eye(d_z) * 0.1)
    
    gmm_weights = nk / np.sum(nk)
    
    return means, covs, gmm_weights


def _compute_log_likelihood(particles: jax.Array, weights: jax.Array, 
                           means: jax.Array, covs: jax.Array, 
                           gmm_weights: jax.Array) -> float:
    """Compute log-likelihood of particles under GMM."""
    n_particles, d_z = particles.shape
    n_components = means.shape[0]
    
    log_probs = np.zeros((n_particles, n_components))
    for i in range(n_components):
        diff = particles - means[i]
        cov_inv = np.linalg.inv(covs[i] + 1e-6 * np.eye(d_z))
        mahal_dist = np.sum(diff @ cov_inv * diff, axis=1)
        log_det = np.linalg.slogdet(covs[i] + 1e-6 * np.eye(d_z))[1]
        log_probs = log_probs.at[:, i].set(
            np.log(gmm_weights[i]) - 0.5 * (d_z * np.log(2 * np.pi) + log_det + mahal_dist)
        )
    
    log_sum = jsp.special.logsumexp(log_probs, axis=1)
    return np.sum(weights * log_sum)


def gmm_to_particles(gmm_state: GMMState) -> jax.Array:
    """Extract particle locations from GMM (using means)."""
    means = [comp.mean for comp in gmm_state.components]
    return np.stack(means, axis=0)


def bures_wasserstein_distance_squared(mu1: jax.Array, cov1: jax.Array,
                                     mu2: jax.Array, cov2: jax.Array) -> float:
    """Compute squared Bures-Wasserstein distance between two Gaussian distributions."""
    mean_diff = np.sum((mu1 - mu2) ** 2)
    
    try:
        cov1_sqrt = jsp.linalg.sqrtm(cov1)
        temp = cov1_sqrt @ cov2 @ cov1_sqrt
        temp_sqrt = jsp.linalg.sqrtm(temp)
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(temp_sqrt)
    except:
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.sqrt(np.trace(cov1) * np.trace(cov2))
    
    return mean_diff + cov_term


def wasserstein_distance_gmm(gmm1: GMMState, gmm2: GMMState) -> float:
    """Compute Wasserstein distance between two GMMs using optimal transport."""
    if gmm1.n_components != gmm2.n_components:
        raise ValueError("GMMs must have same number of components")
    
    n_components = gmm1.n_components
    
    distances = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            distances = distances.at[i, j].set(
                bures_wasserstein_distance_squared(
                    gmm1.components[i].mean, gmm1.components[i].cov,
                    gmm2.components[j].mean, gmm2.components[j].cov
                )
            )
    
    total_distance = 0.0
    used_j = set()
    
    for i in range(n_components):
        best_j = None
        best_dist = np.inf
        for j in range(n_components):
            if j not in used_j and distances[i, j] < best_dist:
                best_dist = distances[i, j]
                best_j = j
        
        if best_j is not None:
            weight_avg = (gmm1.components[i].weight + gmm2.components[best_j].weight) / 2
            total_distance += weight_avg * best_dist
            used_j.add(best_j)
    
    return total_distance


def riemannian_grad_mean(mean: jax.Array, euclidean_grad_mean: jax.Array) -> jax.Array:
    """Riemannian gradient for mean parameters in the Bures-Wasserstein manifold context."""
    return euclidean_grad_mean


def riemannian_grad_cov(euclidean_grad_cov: jax.Array, cov: jax.Array) -> jax.Array:
    """Riemannian gradient for covariance matrix on the Bures-Wasserstein manifold."""
    product = euclidean_grad_cov @ cov
    symmetric_product = (product + product.T) / 2
    return 4 * symmetric_product


def retraction_cov(cov: jax.Array, tangent_vector: jax.Array) -> jax.Array:
    """Retraction operator for covariance matrices on Bures-Wasserstein manifold."""
    new_cov = cov + tangent_vector
    new_cov = (new_cov + new_cov.T) / 2
    d = new_cov.shape[0]
    regularization = 1e-6 * np.eye(d)
    new_cov = new_cov + regularization
    return new_cov


def sinkhorn_weights_update(weights: jax.Array, grad_weights: jax.Array, 
                           lr: float = 0.01, reg: float = 0.1, 
                           n_iter: int = 10) -> jax.Array:
    """Update GMM weights using Sinkhorn-based Wasserstein gradient steps."""
    n = len(weights)
    grad_projected = grad_weights - np.mean(grad_weights)
    
    log_weights = np.log(weights + 1e-8)
    log_weights_new = log_weights - lr * grad_projected
    
    for _ in range(n_iter):
        weights_new = np.exp(log_weights_new - np.max(log_weights_new))
        weights_new = weights_new / np.sum(weights_new)
        log_weights_new = np.log(weights_new + 1e-8)
    
    return weights_new


def compute_elbo(key: jax.random.PRNGKey,
                 pid: PID,
                 target: Target,
                 gmm_state: GMMState,
                 y: jax.Array,
                 hyperparams: PIDParameters) -> float:
    """Compute standard ELBO without regularization."""
    key, subkey = jax.random.split(key)
    samples = sample_from_gmm(subkey, gmm_state, hyperparams.mc_n_samples)
    
    logq = vmap(pid.log_prob, (0, None))(samples, y)
    logp = vmap(target.log_prob, (0, None))(samples, y)
    elbo = np.mean(logp - logq)
    
    return elbo


def compute_elbo_with_wasserstein_regularization(key: jax.random.PRNGKey,
                                               pid: PID,
                                               target: Target,
                                               gmm_state: GMMState,
                                               y: jax.Array,
                                               hyperparams: PIDParameters,
                                               lambda_reg: float = 0.1) -> Tuple[float, float]:
    """
    Compute regularized ELBO and return both ELBO and Wasserstein distance.
    
    Returns:
        Tuple of (elbo_with_regularization, wasserstein_distance)
    """
    # Compute standard ELBO
    elbo = compute_elbo(key, pid, target, gmm_state, y, hyperparams)
    
    # Compute Wasserstein regularization if previous state exists
    wasserstein_reg = 0.0
    if gmm_state.prev_components is not None:
        prev_gmm = GMMState(
            components=gmm_state.prev_components,
            n_components=len(gmm_state.prev_components)
        )
        wasserstein_reg = wasserstein_distance_gmm(gmm_state, prev_gmm)
    
    elbo_with_regularization = elbo - lambda_reg * wasserstein_reg
    
    return elbo_with_regularization, wasserstein_reg


def sample_from_gmm(key: jax.random.PRNGKey, gmm_state: GMMState, 
                   n_samples: int) -> jax.Array:
    """Sample from a GMM."""
    weights = np.array([comp.weight for comp in gmm_state.components])
    
    key, subkey = jax.random.split(key)
    component_indices = jax.random.categorical(
        subkey, np.log(weights), shape=(n_samples,)
    )
    
    d_z = gmm_state.components[0].mean.shape[0]
    samples = np.zeros((n_samples, d_z))
    
    for i, comp_idx in enumerate(component_indices):
        comp = gmm_state.components[comp_idx]
        key, subkey = jax.random.split(key)
        sample = jax.random.multivariate_normal(
            subkey, comp.mean, comp.cov
        )
        samples = samples.at[i].set(sample)
    
    return samples


def compute_gmm_gradients(key: jax.random.PRNGKey,
                         pid: PID,
                         target: Target,
                         gmm_state: GMMState,
                         y: jax.Array,
                         hyperparams: PIDParameters,
                         lambda_reg: float = 0.1) -> Tuple[list, list, jax.Array]:
    """Compute gradients for GMM parameters."""
    def objective_fn(means, covs, weights):
        components = []
        for i in range(len(means)):
            components.append(GMMComponent(
                mean=means[i], cov=covs[i], weight=weights[i]
            ))
        temp_gmm = GMMState(
            components=components,
            n_components=len(components),
            prev_components=gmm_state.prev_components
        )
        
        elbo_with_reg, _ = compute_elbo_with_wasserstein_regularization(
            key, pid, target, temp_gmm, y, hyperparams, lambda_reg
        )
        return elbo_with_reg
    
    means = np.stack([comp.mean for comp in gmm_state.components])
    covs = np.stack([comp.cov for comp in gmm_state.components])
    weights = np.array([comp.weight for comp in gmm_state.components])
    
    grad_fn = jax.grad(objective_fn, argnums=(0, 1, 2))
    mean_grads, cov_grads, weight_grads = grad_fn(means, covs, weights)
    
    return list(mean_grads), list(cov_grads), weight_grads


def update_gmm_parameters(gmm_state: GMMState,
                         mean_grads: list,
                         cov_grads: list,
                         weight_grads: jax.Array,
                         lr_mean: float = 0.01,
                         lr_cov: float = 0.001,
                         lr_weight: float = 0.01) -> GMMState:
    """Update GMM parameters using Riemannian gradients and Sinkhorn steps."""
    new_components = []
    
    current_weights = np.array([comp.weight for comp in gmm_state.components])
    new_weights = sinkhorn_weights_update(current_weights, weight_grads, lr_weight)
    
    for i, comp in enumerate(gmm_state.components):
        new_mean = comp.mean - lr_mean * riemannian_grad_mean(comp.mean, mean_grads[i])
        
        riem_grad_cov = riemannian_grad_cov(cov_grads[i], comp.cov)
        new_cov = retraction_cov(comp.cov, -lr_cov * riem_grad_cov)
        
        new_components.append(GMMComponent(
            mean=new_mean,
            cov=new_cov,
            weight=new_weights[i]
        ))
    
    return GMMState(
        components=new_components,
        n_components=gmm_state.n_components,
        prev_components=gmm_state.components
    )


def wgf_gmm_pvi_step_with_monitoring(key: jax.random.PRNGKey,
                                   carry: PIDCarry,
                                   target: Target,
                                   y: jax.Array,
                                   optim: PIDOpt,
                                   hyperparams: PIDParameters,
                                   lambda_reg: float = 0.1,
                                   lr_mean: float = 0.01,
                                   lr_cov: float = 0.001,
                                   lr_weight: float = 0.01) -> Tuple[float, PIDCarry, WGFGMMMetrics]:
    """
    WGF-GMM step with detailed monitoring of ELBO and Wasserstein metrics.
    
    Returns:
        Tuple of (loss_value, updated_carry, metrics)
    """
    theta_key, r_key, grad_key, metrics_key = jax.random.split(key, 4)
    
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
        metrics_key, pid, target, gmm_state, y, hyperparams, lambda_reg
    )
    
    # Step 4: Compute gradients for GMM parameters
    mean_grads, cov_grads, weight_grads = compute_gmm_gradients(
        grad_key, pid, target, gmm_state, y, hyperparams, lambda_reg
    )
    
    # Step 5: Update GMM parameters using WGF
    updated_gmm_state = update_gmm_parameters(
        gmm_state, mean_grads, cov_grads, weight_grads,
        lr_mean, lr_cov, lr_weight
    )
    
    # Step 6: Convert back to particle representation for compatibility
    updated_particles = gmm_to_particles(updated_gmm_state)
    pid = eqx.tree_at(lambda tree: tree.particles, pid, updated_particles)
    
    # Create updated carry with GMM state
    updated_carry = PIDCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=carry.r_opt_state,
        r_precon_state=carry.r_precon_state
    )
    
    # Store GMM state in carry
    updated_carry.gmm_state = updated_gmm_state
    
    # Create metrics object
    metrics = WGFGMMMetrics(
        elbo=elbo,
        elbo_with_wasserstein=elbo_with_reg,
        wasserstein_distance=wasserstein_dist,
        lambda_reg=lambda_reg,
        lr_mean=lr_mean,
        lr_cov=lr_cov,
        lr_weight=lr_weight
    )
    
    return lval, updated_carry, metrics


def run_wgf_gmm_hyperparameter_search(
    target,
    initial_carry,
    optim,
    hyperparams,
    n_updates: int = 1000,
    output_dir: Path = Path("output/wgf_gmm_search"),
    lambda_reg_values: list = [0.01, 0.05, 0.1, 0.5, 1.0],
    lr_mean_values: list = [0.005, 0.01, 0.05],
    lr_cov_values: list = [0.0005, 0.001, 0.005],
    lr_weight_values: list = [0.005, 0.01, 0.02],
    seed: int = 42
):
    """
    Run comprehensive hyperparameter search for WGF-GMM.
    
    Args:
        target: Target distribution
        initial_carry: Initial carry state
        optim: Optimizer
        hyperparams: Hyperparameters
        n_updates: Number of training updates
        output_dir: Directory to save results
        lambda_reg_values: List of lambda regularization values to try
        lr_mean_values: List of learning rates for means
        lr_cov_values: List of learning rates for covariances  
        lr_weight_values: List of learning rates for weights
        seed: Random seed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    
    all_results = []
    
    for lambda_reg in lambda_reg_values:
        for lr_mean in lr_mean_values:
            for lr_cov in lr_cov_values:
                for lr_weight in lr_weight_values:
                    
                    print(f"Running: λ={lambda_reg}, lr_mean={lr_mean}, lr_cov={lr_cov}, lr_weight={lr_weight}")
                    
                    # Reset carry for each experiment
                    carry = initial_carry
                    key, experiment_key = jax.random.split(key)
                    
                    # Storage for this experiment
                    experiment_metrics = []
                    losses = []
                    
                    for update_idx in range(n_updates):
                        experiment_key, step_key = jax.random.split(experiment_key)
                        
                        # Perform WGF-GMM step with monitoring
                        lval, carry, metrics = wgf_gmm_pvi_step_with_monitoring(
                            step_key, carry, target, None, optim, hyperparams,
                            lambda_reg=lambda_reg,
                            lr_mean=lr_mean,
                            lr_cov=lr_cov,
                            lr_weight=lr_weight
                        )
                        
                        losses.append(float(lval))
                        experiment_metrics.append(metrics)
                        
                        # Print progress every 100 steps
                        if (update_idx + 1) % 100 == 0:
                            print(f"  Step {update_idx + 1}/{n_updates}, Loss: {lval:.4f}, "
                                  f"ELBO: {metrics.elbo:.4f}, W_dist: {metrics.wasserstein_distance:.4f}")
                    
                    # Create experiment identifier
                    exp_id = f"lambda{lambda_reg}_mean{lr_mean}_cov{lr_cov}_weight{lr_weight}"
                    exp_dir = output_dir / exp_id
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save metrics as pickle
                    with open(exp_dir / "metrics.pkl", "wb") as f:
                        pickle.dump(experiment_metrics, f)
                    
                    # Save losses
                    with open(exp_dir / "losses.pkl", "wb") as f:
                        pickle.dump(losses, f)
                    
                    # Plot and save ELBO metrics
                    plot_elbo_metrics(experiment_metrics, exp_dir, exp_id)
                    
                    # Plot loss curve
                    plot_loss_curve(losses, exp_dir, exp_id)
                    
                    # Store summary for comparison
                    final_metrics = experiment_metrics[-1]
                    all_results.append({
                        'lambda_reg': lambda_reg,
                        'lr_mean': lr_mean,
                        'lr_cov': lr_cov,
                        'lr_weight': lr_weight,
                        'final_elbo': final_metrics.elbo,
                        'final_elbo_with_wasserstein': final_metrics.elbo_with_wasserstein,
                        'final_wasserstein_distance': final_metrics.wasserstein_distance,
                        'final_loss': losses[-1],
                        'exp_id': exp_id
                    })
    
    # Save comprehensive results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "hyperparameter_search_results.csv", index=False)
    
    # Create summary plots
    create_summary_plots(results_df, output_dir)
    
    # Print best configurations
    print_best_configurations(results_df)
    
    return results_df


def plot_elbo_metrics(metrics_list, save_dir, exp_id):
    """Plot ELBO and Wasserstein metrics over time."""
    elbo_values = [m.elbo for m in metrics_list]
    elbo_with_wasserstein = [m.elbo_with_wasserstein for m in metrics_list]
    wasserstein_distances = [m.wasserstein_distance for m in metrics_list]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ELBO plot
    axes[0].plot(elbo_values, label='ELBO', color='blue')
    axes[0].set_title(f'ELBO - {exp_id}')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('ELBO')
    axes[0].grid(True)
    axes[0].legend()
    
    # ELBO with Wasserstein regularization plot
    axes[1].plot(elbo_with_wasserstein, label='ELBO + Wasserstein', color='red')
    axes[1].set_title(f'ELBO with Wasserstein Regularization - {exp_id}')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('ELBO with Regularization')
    axes[1].grid(True)
    axes[1].legend()
    
    # Wasserstein distance plot
    axes[2].plot(wasserstein_distances, label='Wasserstein Distance', color='green')
    axes[2].set_title(f'Wasserstein Distance - {exp_id}')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Wasserstein Distance')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / f"elbo_metrics_{exp_id}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / f"elbo_metrics_{exp_id}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curve(losses, save_dir, exp_id):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, color='purple', linewidth=2)
    plt.title(f'Training Loss - {exp_id}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_dir / f"loss_curve_{exp_id}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / f"loss_curve_{exp_id}.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_plots(results_df, save_dir):
    """Create summary plots comparing different hyperparameter configurations."""
    
    # 1. ELBO comparison by lambda_reg
    plt.figure(figsize=(12, 8))
    lambda_groups = results_df.groupby('lambda_reg')
    
    for lambda_val, group in lambda_groups:
        plt.scatter(group['lr_mean'], group['final_elbo'], 
                   label=f'λ={lambda_val}', alpha=0.7, s=60)
    
    plt.xlabel('Learning Rate (Mean)')
    plt.ylabel('Final ELBO')
    plt.title('Final ELBO vs Learning Rate (Mean) by Lambda Regularization')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "elbo_vs_lr_mean_by_lambda.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / "elbo_vs_lr_mean_by_lambda.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap of ELBO by lr_mean and lr_cov
    pivot_elbo = results_df.pivot_table(
        values='final_elbo', 
        index='lr_cov', 
        columns='lr_mean', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(pivot_elbo, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Final ELBO: lr_cov vs lr_mean (averaged over other params)')
    plt.xlabel('Learning Rate (Mean)')
    plt.ylabel('Learning Rate (Covariance)')
    plt.savefig(save_dir / "elbo_heatmap_lr_mean_cov.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / "elbo_heatmap_lr_mean_cov.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Wasserstein distance vs Lambda regularization
    plt.figure(figsize=(10, 6))
    
    for lr_mean in results_df['lr_mean'].unique():
        subset = results_df[results_df['lr_mean'] == lr_mean]
        plt.plot(subset['lambda_reg'], subset['final_wasserstein_distance'], 
                marker='o', label=f'lr_mean={lr_mean}', linewidth=2)
    
    plt.xlabel('Lambda Regularization')
    plt.ylabel('Final Wasserstein Distance')
    plt.title('Wasserstein Distance vs Lambda Regularization')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "wasserstein_vs_lambda.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / "wasserstein_vs_lambda.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Loss vs ELBO scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(results_df['final_loss'], results_df['final_elbo'], 
                         c=results_df['lambda_reg'], cmap='plasma', alpha=0.7, s=60)
    plt.colorbar(scatter, label='Lambda Regularization')
    plt.xlabel('Final Loss')
    plt.ylabel('Final ELBO')
    plt.title('Final ELBO vs Final Loss (colored by Lambda)')
    plt.grid(True)
    plt.savefig(save_dir / "elbo_vs_loss_by_lambda.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / "elbo_vs_loss_by_lambda.png", dpi=300, bbox_inches='tight')
    plt.close()


def print_best_configurations(results_df):
    """Print the best configurations based on different metrics."""
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)
    
    # Best ELBO
    best_elbo = results_df.loc[results_df['final_elbo'].idxmax()]
    print(f"\nBest ELBO: {best_elbo['final_elbo']:.4f}")
    print(f"Configuration: λ={best_elbo['lambda_reg']}, lr_mean={best_elbo['lr_mean']}, "
          f"lr_cov={best_elbo['lr_cov']}, lr_weight={best_elbo['lr_weight']}")
    
    # Best ELBO with Wasserstein
    best_elbo_w = results_df.loc[results_df['final_elbo_with_wasserstein'].idxmax()]
    print(f"\nBest ELBO with Wasserstein: {best_elbo_w['final_elbo_with_wasserstein']:.4f}")
    print(f"Configuration: λ={best_elbo_w['lambda_reg']}, lr_mean={best_elbo_w['lr_mean']}, "
          f"lr_cov={best_elbo_w['lr_cov']}, lr_weight={best_elbo_w['lr_weight']}")
    
    # Lowest final loss
    best_loss = results_df.loc[results_df['final_loss'].idxmin()]
    print(f"\nLowest Final Loss: {best_loss['final_loss']:.4f}")
    print(f"Configuration: λ={best_loss['lambda_reg']}, lr_mean={best_loss['lr_mean']}, "
          f"lr_cov={best_loss['lr_cov']}, lr_weight={best_loss['lr_weight']}")
    
    # Best balance (high ELBO, low Wasserstein distance)
    results_df['balance_score'] = results_df['final_elbo'] - 0.1 * results_df['final_wasserstein_distance']
    best_balance = results_df.loc[results_df['balance_score'].idxmax()]
    print(f"\nBest Balance (ELBO - 0.1*Wasserstein): {best_balance['balance_score']:.4f}")
    print(f"Configuration: λ={best_balance['lambda_reg']}, lr_mean={best_balance['lr_mean']}, "
          f"lr_cov={best_balance['lr_cov']}, lr_weight={best_balance['lr_weight']}")
    
    print("\n" + "="*70)


def create_wgf_gmm_config_file(output_dir: Path, best_configs: dict):
    """Create configuration files with the best hyperparameters found."""
    config_dir = output_dir / "best_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    for config_name, config_data in best_configs.items():
        config_content = f"""default_parameters: &default_parameters
    d_z: 2
    kernel: 'norm_fixed_var_w_skip'
    n_hidden: 512

default_theta_lr: &default_theta_lr
    lr: 1e-4

experiment:
    n_reruns: 10
    n_updates: 15000
    name: 'wgf_gmm_{config_name}'
    compute_metrics: False
    use_jit: True

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
        lambda_reg: {config_data['lambda_reg']}
        lr_mean: {config_data['lr_mean']}
        lr_cov: {config_data['lr_cov']}
        lr_weight: {config_data['lr_weight']}
    extra_alg:
"""
        
        with open(config_dir / f"wgf_gmm_{config_name}.yaml", "w") as f:
            f.write(config_content)
    
    print(f"Best configuration files saved to: {config_dir}")


# Example usage function
def run_example_hyperparameter_search():
    """Example of how to run the hyperparameter search."""
    from src.problems.toy import Banana, Multimodal, XShape
    from src.utils import make_step_and_carry, config_to_parameters
    from src.base import Parameters, ModelParameters, ThetaOptParameters, ROptParameters, PIDParameters
    
    # Example target
    target = Banana()
    
    # Create parameters (you can adapt this based on your existing config)
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
    
    # Initialize step and carry
    key = jax.random.PRNGKey(42)
    step, carry = make_step_and_carry(key, parameters, target)
    
    # Run hyperparameter search
    results_df = run_wgf_gmm_hyperparameter_search(
        target=target,
        initial_carry=carry,
        optim=None,  # This would be created in make_step_and_carry
        hyperparams=parameters.extra_alg_parameters,
        n_updates=500,  # Reduced for example
        output_dir=Path("output/wgf_gmm_hyperparam_search"),
        lambda_reg_values=[0.01, 0.1, 0.5],
        lr_mean_values=[0.01, 0.05],
        lr_cov_values=[0.001, 0.005],
        lr_weight_values=[0.01, 0.02],
        seed=42
    )
    
    return results_df


if __name__ == "__main__":
    # Run the example
    results = run_example_hyperparameter_search()
    print("Hyperparameter search completed!")
    print(f"Results shape: {results.shape}")
    print("\nTop 5 configurations by ELBO:")
    print(results.nlargest(5, 'final_elbo')[['lambda_reg', 'lr_mean', 'lr_cov', 'lr_weight', 'final_elbo']])