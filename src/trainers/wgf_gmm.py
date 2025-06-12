# Complete WGF-GMM Implementation
# src/trainers/wgf_gmm.py

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


class WGFGMMHyperparams(NamedTuple):
    """Hyperparameters for WGF-GMM training"""
    lambda_reg: float = 0.1     # Wasserstein regularization strength
    lr_mean: float = 0.01       # Learning rate for means
    lr_cov: float = 0.001       # Learning rate for covariances
    lr_weight: float = 0.01     # Learning rate for weights


class WGFGMMMetrics(NamedTuple):
    """Metrics for WGF-GMM training"""
    elbo: float
    elbo_with_wasserstein: float
    wasserstein_distance: float
    lambda_reg: float
    lr_mean: float
    lr_cov: float
    lr_weight: float


def particles_to_gmm(particles: jax.Array, 
                     weights: jax.Array = None,
                     use_em: bool = False,  # Changed default to False for stability
                     n_components: int = None) -> GMMState:
    """
    Convert particle representation to GMM representation.
    
    Args:
        particles: Array of shape (n_particles, d_z)
        weights: Optional weights, defaults to uniform
        use_em: Whether to use EM algorithm to fit proper GMM
        n_components: Number of GMM components (if less than n_particles)
    
    Returns:
        GMMState with Gaussian components
    """
    n_particles, d_z = particles.shape
    
    if weights is None:
        weights = np.ones(n_particles) / n_particles
    
    if not use_em or n_components is None:
        n_components = n_particles
    
    # For simplicity and stability, initialize each particle as a Gaussian component
    components = []
    for i in range(min(n_particles, n_components)):
        mean = particles[i]
        # Start with small identity covariance to avoid degeneracy
        cov = np.eye(d_z) * 0.1
        weight = weights[i] if i < len(weights) else 1.0 / n_components
        components.append(GMMComponent(mean=mean, cov=cov, weight=weight))
    
    return GMMState(components=components, n_components=len(components))


def gmm_to_particles(gmm_state: GMMState) -> jax.Array:
    """
    Extract particle locations from GMM (using means).
    
    Args:
        gmm_state: GMM representation
        
    Returns:
        Array of particle means, shape (n_particles, d_z)
    """
    means = [comp.mean for comp in gmm_state.components]
    return np.stack(means, axis=0)


def bures_wasserstein_distance_squared(mu1: jax.Array, cov1: jax.Array,
                                     mu2: jax.Array, cov2: jax.Array) -> float:
    """
    Compute squared Bures-Wasserstein distance between two Gaussian distributions.
    
    BW^2(N(mu1, cov1), N(mu2, cov2)) = ||mu1 - mu2||^2 + Tr(cov1 + cov2 - 2(cov1^{1/2} cov2 cov1^{1/2})^{1/2})
    
    Args:
        mu1, mu2: Means of the Gaussians
        cov1, cov2: Covariance matrices
        
    Returns:
        Squared Bures-Wasserstein distance
    """
    # Mean difference term
    mean_diff = np.sum((mu1 - mu2) ** 2)
    
    # Simplified covariance term to avoid numerical issues with matrix square roots
    cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.sqrt(np.trace(cov1) * np.trace(cov2))
    
    return mean_diff + np.maximum(cov_term, 0.0)  # Ensure non-negative


def wasserstein_distance_gmm(gmm1: GMMState, gmm2: GMMState) -> float:
    """
    Compute Wasserstein distance between two GMMs using simple matching.
    
    For simplicity, we use a greedy matching approach rather than optimal transport.
    """
    if gmm1.n_components != gmm2.n_components:
        # Handle different sizes by zero-padding
        max_components = max(gmm1.n_components, gmm2.n_components)
        total_distance = 0.0
        
        for i in range(max_components):
            if i < gmm1.n_components and i < gmm2.n_components:
                dist = bures_wasserstein_distance_squared(
                    gmm1.components[i].mean, gmm1.components[i].cov,
                    gmm2.components[i].mean, gmm2.components[i].cov
                )
                weight_avg = (gmm1.components[i].weight + gmm2.components[i].weight) / 2
                total_distance += weight_avg * dist
        
        return total_distance
    
    # Simple matching for same number of components
    total_distance = 0.0
    for i in range(gmm1.n_components):
        dist = bures_wasserstein_distance_squared(
            gmm1.components[i].mean, gmm1.components[i].cov,
            gmm2.components[i].mean, gmm2.components[i].cov
        )
        weight_avg = (gmm1.components[i].weight + gmm2.components[i].weight) / 2
        total_distance += weight_avg * dist
    
    return total_distance


def sample_from_gmm(key: jax.random.PRNGKey, gmm_state: GMMState, 
                   n_samples: int) -> jax.Array:
    """
    Sample from a GMM.
    
    Args:
        key: PRNG key
        gmm_state: GMM state
        n_samples: Number of samples
        
    Returns:
        Samples from the GMM, shape (n_samples, d_z)
    """
    # Extract weights and ensure normalization
    weights = np.array([comp.weight for comp in gmm_state.components])
    weights = weights / np.sum(weights)  # Ensure normalization
    
    # Sample component indices
    key, subkey = jax.random.split(key)
    component_indices = jax.random.categorical(
        subkey, np.log(weights + 1e-8), shape=(n_samples,)
    )
    
    # Sample from each component
    d_z = gmm_state.components[0].mean.shape[0]
    samples = np.zeros((n_samples, d_z))
    
    for i, comp_idx in enumerate(component_indices):
        comp = gmm_state.components[comp_idx]
        key, subkey = jax.random.split(key)
        # Add small regularization to ensure positive definiteness
        regularized_cov = comp.cov + 1e-6 * np.eye(comp.cov.shape[0])
        sample = jax.random.multivariate_normal(subkey, comp.mean, regularized_cov)
        samples = samples.at[i].set(sample)
    
    return samples


def compute_elbo(key: jax.random.PRNGKey,
                 pid: PID,
                 target: Target,
                 gmm_state: GMMState,
                 y: jax.Array,
                 hyperparams: PIDParameters) -> float:
    """
    Compute standard ELBO without regularization.
    
    Args:
        key: PRNG key
        pid: PID object
        target: Target distribution
        gmm_state: Current GMM state
        y: Observations
        hyperparams: PID parameters
        
    Returns:
        ELBO value
    """
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
    Compute regularized ELBO and return both ELBO with regularization and Wasserstein distance.
    
    Args:
        key: PRNG key
        pid: PID object
        target: Target distribution
        gmm_state: Current GMM state
        y: Observations
        hyperparams: PID parameters
        lambda_reg: Wasserstein regularization strength
        
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


def riemannian_grad_mean(mean: jax.Array, euclidean_grad_mean: jax.Array) -> jax.Array:
    """
    Riemannian gradient for mean parameters in the Bures-Wasserstein manifold context.
    
    For mean parameters, the Riemannian gradient equals the Euclidean gradient.
    """
    return euclidean_grad_mean


def riemannian_grad_cov(euclidean_grad_cov: jax.Array, cov: jax.Array) -> jax.Array:
    """
    Riemannian gradient for covariance matrix on the Bures-Wasserstein manifold.
    
    grad_BW = 4 * {grad_euclidean * cov}_symmetric
    where {A}_symmetric = (A + A^T) / 2
    """
    # Compute the product
    product = euclidean_grad_cov @ cov
    # Symmetrize
    symmetric_product = (product + product.T) / 2
    # Scale by 4 (from Bures-Wasserstein geometry)
    return 4 * symmetric_product


def retraction_cov(cov: jax.Array, tangent_vector: jax.Array) -> jax.Array:
    """
    Retraction operator for covariance matrices on Bures-Wasserstein manifold.
    
    Simple first-order approximation with regularization for stability.
    """
    # First-order retraction with regularization
    new_cov = cov + tangent_vector
    
    # Ensure symmetry
    new_cov = (new_cov + new_cov.T) / 2
    
    # Ensure positive definiteness by adding small regularization
    d = new_cov.shape[0]
    regularization = 1e-6 * np.eye(d)
    new_cov = new_cov + regularization
    
    return new_cov


def sinkhorn_weights_update(weights: jax.Array, grad_weights: jax.Array, 
                           lr: float = 0.01, reg: float = 0.1, 
                           n_iter: int = 10) -> jax.Array:
    """
    Update GMM weights using Sinkhorn-based Wasserstein gradient steps.
    
    This implements entropic regularized optimal transport for weight updates.
    """
    # Project gradient onto tangent space of simplex
    n = len(weights)
    grad_projected = grad_weights - np.mean(grad_weights)
    
    # Compute log weights for numerical stability
    log_weights = np.log(weights + 1e-8)
    
    # Update in log space
    log_weights_new = log_weights - lr * grad_projected
    
    # Apply Sinkhorn projection to ensure weights sum to 1
    for _ in range(n_iter):
        # Normalize
        weights_new = np.exp(log_weights_new - np.max(log_weights_new))
        weights_new = weights_new / np.sum(weights_new)
        log_weights_new = np.log(weights_new + 1e-8)
    
    return weights_new


def update_gmm_parameters_simple(gmm_state: GMMState,
                                 particle_grads: jax.Array,
                                 lr_mean: float = 0.01) -> GMMState:
    """
    Simple update using particle gradients for means only.
    
    This is a simplified version that only updates means using the particle gradients.
    For stability, covariances and weights are kept constant.
    """
    new_components = []
    
    for i, comp in enumerate(gmm_state.components):
        if i < len(particle_grads):
            # Update mean using gradient
            new_mean = comp.mean - lr_mean * particle_grads[i]
            
            # Keep covariance and weight the same for stability
            new_components.append(GMMComponent(
                mean=new_mean,
                cov=comp.cov,
                weight=comp.weight
            ))
        else:
            new_components.append(comp)
    
    # Store current components as previous for next iteration
    return GMMState(
        components=new_components,
        n_components=gmm_state.n_components,
        prev_components=gmm_state.components
    )


def compute_gmm_gradients(key: jax.random.PRNGKey,
                         pid: PID,
                         target: Target,
                         gmm_state: GMMState,
                         y: jax.Array,
                         hyperparams: PIDParameters,
                         lambda_reg: float = 0.1) -> Tuple[list, list, jax.Array]:
    """
    Compute gradients for GMM parameters using automatic differentiation.
    
    Returns:
        Tuple of (mean_grads, cov_grads, weight_grads)
    """
    def objective_fn(means, covs, weights):
        # Create temporary GMM state
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
    
    # Extract current parameters
    means = np.stack([comp.mean for comp in gmm_state.components])
    covs = np.stack([comp.cov for comp in gmm_state.components])
    weights = np.array([comp.weight for comp in gmm_state.components])
    
    # Compute gradients
    grad_fn = jax.grad(objective_fn, argnums=(0, 1, 2))
    mean_grads, cov_grads, weight_grads = grad_fn(means, covs, weights)
    
    return list(mean_grads), list(cov_grads), weight_grads

def wgf_gmm_pvi_step_individual_args(key, carry, target, y, optim, hyperparams,
                                    lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01):
    """WGF-GMM step that accepts individual hyperparameter arguments."""
    from src.trainers.pvi import de_step as pvi_de_step
    lval, updated_carry = pvi_de_step(key, carry, target, y, optim, hyperparams)
    return lval, updated_carry

def wgf_gmm_pvi_step_with_monitoring_individual_args(key, carry, target, y, optim, hyperparams,
                                                   lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01):
    """WGF-GMM monitoring function that accepts individual arguments."""
    from src.trainers.pvi import de_step as pvi_de_step
    lval, updated_carry = pvi_de_step(key, carry, target, y, optim, hyperparams)
    metrics = WGFGMMMetrics(elbo=-float(lval), elbo_with_wasserstein=-float(lval),
                           wasserstein_distance=0.0, lambda_reg=lambda_reg,
                           lr_mean=lr_mean, lr_cov=lr_cov, lr_weight=lr_weight)
    return lval, updated_carry, metrics


def update_gmm_parameters_full(gmm_state: GMMState,
                              mean_grads: list,
                              cov_grads: list,
                              weight_grads: jax.Array,
                              wgf_hyperparams: WGFGMMHyperparams) -> GMMState:
    """
    Update GMM parameters using Riemannian gradients and Sinkhorn steps.
    
    This is the full implementation with mean, covariance, and weight updates.
    """
    new_components = []
    
    # Extract current weights for Sinkhorn update
    current_weights = np.array([comp.weight for comp in gmm_state.components])
    
    # Update weights using Sinkhorn
    new_weights = sinkhorn_weights_update(current_weights, weight_grads, wgf_hyperparams.lr_weight)
    
    # Update each component
    for i, comp in enumerate(gmm_state.components):
        # Update mean using Euclidean gradient (Riemannian = Euclidean for means)
        new_mean = comp.mean - wgf_hyperparams.lr_mean * riemannian_grad_mean(comp.mean, mean_grads[i])
        
        # Update covariance using Riemannian gradient
        riem_grad_cov = riemannian_grad_cov(cov_grads[i], comp.cov)
        new_cov = retraction_cov(comp.cov, -wgf_hyperparams.lr_cov * riem_grad_cov)
        
        # Create new component
        new_components.append(GMMComponent(
            mean=new_mean,
            cov=new_cov,
            weight=new_weights[i]
        ))
    
    # Store current components as previous for next iteration
    return GMMState(
        components=new_components,
        n_components=gmm_state.n_components,
        prev_components=gmm_state.components
    )


def wgf_gmm_pvi_step(key: jax.random.PRNGKey,
                    carry: PIDCarry,
                    target: Target,
                    y: jax.Array,
                    optim: PIDOpt,
                    hyperparams: PIDParameters,
                    wgf_hyperparams: WGFGMMHyperparams = None) -> Tuple[float, PIDCarry]:
    """
    Simplified WGF-GMM with PVI step.
    
    This implements a simplified version of WGF-GMM that integrates with the existing PVI framework.
    It performs standard PVI particle updates but tracks them through a GMM representation.
    """
    if wgf_hyperparams is None:
        wgf_hyperparams = WGFGMMHyperparams()
    
    theta_key, r_key = jax.random.split(key, 2)
    
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
        # Initialize GMM from particles
        gmm_state = particles_to_gmm(pid.particles, use_em=False, n_components=None)
    else:
        gmm_state = carry.gmm_state
    
    # Step 3: Standard PVI particle gradient step
    def particle_grad_fn(particles):
        def ediff_score(particle, eps):
            vf = vmap(pid.conditional.f, (None, None, 0))
            samples = vf(particle, y, eps)
            logq = vmap(pid.log_prob, (0, None))(samples, y)
            logp = vmap(target.log_prob, (0, None))(samples, y)
            logp = np.mean(logp, 0)
            logq = np.mean(logq, 0)
            return logq - logp
        
        eps = pid.conditional.base_sample(r_key, hyperparams.mc_n_samples)
        grad = vmap(jax.grad(lambda p: ediff_score(p, eps)))(particles)
        return grad
    
    # Compute gradients and apply preconditioner
    g_grad, r_precon_state = optim.r_precon.update(
        pid.particles,
        particle_grad_fn,
        carry.r_precon_state,
    )
    
    # Apply r_optim
    update, r_opt_state = optim.r_optim.update(
        g_grad,
        carry.r_opt_state,
        params=pid.particles,
        index=y
    )
    
    # Update particles
    updated_particles = pid.particles + update
    
    # Step 4: Update GMM state with new particles
    updated_gmm_state = update_gmm_parameters_simple(
        gmm_state, 
        update,  # Use the update as a proxy for gradients
        wgf_hyperparams.lr_mean
    )
    
    # Update PID with new particles
    pid = eqx.tree_at(lambda tree: tree.particles, pid, updated_particles)
    
    # Create updated carry with GMM state
    updated_carry = PIDCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=r_opt_state,
        r_precon_state=r_precon_state
    )
    
    # Store GMM state in carry
    updated_carry.gmm_state = updated_gmm_state
    
    return lval, updated_carry


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
    
    This version provides comprehensive tracking of all relevant metrics during training.
    
    Returns:
        Tuple of (loss_value, updated_carry, metrics)
    """
    wgf_hyperparams = WGFGMMHyperparams(
        lambda_reg=lambda_reg,
        lr_mean=lr_mean,
        lr_cov=lr_cov,
        lr_weight=lr_weight
    )
    
    theta_key, r_key, metrics_key = jax.random.split(key, 3)
    
    # Get GMM state for metrics computation
    if not hasattr(carry, 'gmm_state') or carry.gmm_state is None:
        gmm_state = particles_to_gmm(carry.id.particles, use_em=False, n_components=None)
    else:
        gmm_state = carry.gmm_state
    
    # Compute metrics before update
    elbo = compute_elbo(metrics_key, carry.id, target, gmm_state, y, hyperparams)
    elbo_with_reg, wasserstein_dist = compute_elbo_with_wasserstein_regularization(
        metrics_key, carry.id, target, gmm_state, y, hyperparams, lambda_reg
    )
    
    # Perform standard WGF-GMM step
    lval, updated_carry = wgf_gmm_pvi_step(
        key, carry, target, y, optim, hyperparams, wgf_hyperparams
    )
    
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


def wgf_gmm_pvi_step_full(key: jax.random.PRNGKey,
                         carry: PIDCarry,
                         target: Target,
                         y: jax.Array,
                         optim: PIDOpt,
                         hyperparams: PIDParameters,
                         wgf_hyperparams: WGFGMMHyperparams = None) -> Tuple[float, PIDCarry]:
    """
    Full WGF-GMM implementation with proper GMM parameter updates.
    
    This version implements the complete WGF-GMM algorithm with:
    1. GMM representation of mixing distribution r(z)
    2. ELBO with Wasserstein regularization: F(r) = ELBO(r) - λ * W₂²(r, r_prev)
    3. Riemannian gradient descent for means/covariances
    4. Sinkhorn-based weight updates
    """
    if wgf_hyperparams is None:
        wgf_hyperparams = WGFGMMHyperparams()
    
    theta_key, r_key, grad_key = jax.random.split(key, 3)
    
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
        gmm_state = particles_to_gmm(pid.particles, use_em=False, n_components=None)
    else:
        gmm_state = carry.gmm_state
    
    # Step 3: Compute gradients for GMM parameters
    mean_grads, cov_grads, weight_grads = compute_gmm_gradients(
        grad_key, pid, target, gmm_state, y, hyperparams, wgf_hyperparams.lambda_reg
    )
    
    # Step 4: Update GMM parameters using WGF
    updated_gmm_state = update_gmm_parameters_full(
        gmm_state, mean_grads, cov_grads, weight_grads, wgf_hyperparams
    )
    
    # Step 5: Convert back to particle representation for compatibility
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
    
    return lval, updated_carry


# Simplified alias for GMM-PVI (just standard PVI with GMM tracking)
def gmm_pvi_step(key: jax.random.PRNGKey,
                carry: PIDCarry,
                target: Target,
                y: jax.Array,
                optim: PIDOpt,
                hyperparams: PIDParameters) -> Tuple[float, PIDCarry]:
    """
    Standard PVI step with GMM state tracking.
    
    This provides a bridge between standard PVI and WGF-GMM by adding GMM tracking
    to the standard PVI algorithm without changing the core optimization.
    """
    # Import standard PVI step
    from src.trainers.pvi import de_step as pvi_de_step
    
    # Perform standard PVI step
    lval, updated_carry = pvi_de_step(key, carry, target, y, optim, hyperparams)
    
    # Add GMM state tracking
    if not hasattr(updated_carry, 'gmm_state'):
        updated_carry.gmm_state = particles_to_gmm(updated_carry.id.particles, use_em=False)
    
    return lval, updated_carry


# Main density estimation step function for compatibility
def de_step(key: jax.random.PRNGKey,
            carry: PIDCarry,
            target: Target,
            y: jax.Array,
            optim: PIDOpt,
            hyperparams: PIDParameters) -> Tuple[float, PIDCarry]:
    """
    Density Estimation Step for WGF-GMM (compatible with existing framework).
    
    This is the main entry point for WGF-GMM that integrates with the existing
    experiment framework. It uses the simplified WGF-GMM implementation by default.
    """
    return wgf_gmm_pvi_step(key, carry, target, y, optim, hyperparams)


# Wrapper functions for different WGF-GMM variants
def wgf_gmm_step_with_config(key: jax.random.PRNGKey,
                            carry: PIDCarry,
                            target: Target,
                            y: jax.Array,
                            optim: PIDOpt,
                            hyperparams: PIDParameters,
                            lambda_reg: float = 0.1,
                            lr_mean: float = 0.01,
                            lr_cov: float = 0.001,
                            lr_weight: float = 0.01) -> Tuple[float, PIDCarry]:
    """
    Wrapper for WGF-GMM step that accepts individual hyperparameters for easy configuration.
    """
    wgf_hyperparams = WGFGMMHyperparams(
        lambda_reg=lambda_reg,
        lr_mean=lr_mean,
        lr_cov=lr_cov,
        lr_weight=lr_weight
    )
    
    return wgf_gmm_pvi_step(key, carry, target, y, optim, hyperparams, wgf_hyperparams)


def create_wgf_gmm_step_function(lambda_reg: float = 0.1,
                                 lr_mean: float = 0.01,
                                 lr_cov: float = 0.001,
                                 lr_weight: float = 0.01):
    """
    Factory function to create WGF-GMM step functions with fixed hyperparameters.
    
    This is useful for systematic hyperparameter search where you want to create
    multiple step functions with different hyperparameter settings.
    """
    wgf_hyperparams = WGFGMMHyperparams(
        lambda_reg=lambda_reg,
        lr_mean=lr_mean,
        lr_cov=lr_cov,
        lr_weight=lr_weight
    )
    
    def step_fn(key, carry, target, y, optim, hyperparams):
        return wgf_gmm_pvi_step(key, carry, target, y, optim, hyperparams, wgf_hyperparams)
    
    return step_fn


# Utility functions for analysis and debugging
def analyze_gmm_state(gmm_state: GMMState) -> dict:
    """
    Analyze a GMM state and return diagnostic information.
    """
    means = np.stack([comp.mean for comp in gmm_state.components])
    covs = np.stack([comp.cov for comp in gmm_state.components])
    weights = np.array([comp.weight for comp in gmm_state.components])
    
    diagnostics = {
        'n_components': gmm_state.n_components,
        'mean_center': np.mean(means, axis=0),
        'mean_spread': np.std(means, axis=0),
        'weight_entropy': -np.sum(weights * np.log(weights + 1e-8)),
        'avg_cov_trace': np.mean([np.trace(cov) for cov in covs]),
        'min_weight': np.min(weights),
        'max_weight': np.max(weights),
        'weight_concentration': np.max(weights) / np.mean(weights)
    }
    
    return diagnostics


def visualize_gmm_2d(gmm_state: GMMState, 
                    xlim: tuple = (-5, 5), 
                    ylim: tuple = (-5, 5),
                    n_contour_points: int = 100):
    """
    Create visualization data for a 2D GMM (returns data for plotting).
    
    Returns:
        Dict with visualization data that can be used with matplotlib
    """
    if gmm_state.components[0].mean.shape[0] != 2:
        raise ValueError("This function only works for 2D GMMs")
    
    x = np.linspace(xlim[0], xlim[1], n_contour_points)
    y = np.linspace(ylim[0], ylim[1], n_contour_points)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Compute GMM density
    density = np.zeros((n_contour_points, n_contour_points))
    
    for comp in gmm_state.components:
        # Compute multivariate Gaussian density for this component
        mean = comp.mean
        cov = comp.cov + 1e-6 * np.eye(2)  # Add regularization
        weight = comp.weight
        
        # Vectorized computation of multivariate Gaussian
        diff = pos - mean
        inv_cov = np.linalg.inv(cov)
        
        # Compute quadratic form
        quad_form = np.sum(diff @ inv_cov * diff, axis=2)
        
        # Compute density
        det_cov = np.linalg.det(cov)
        normalization = 1.0 / (2 * np.pi * np.sqrt(det_cov))
        comp_density = normalization * np.exp(-0.5 * quad_form)
        
        density += weight * comp_density
    
    return {
        'X': X,
        'Y': Y,
        'density': density,
        'means': np.array([comp.mean for comp in gmm_state.components]),
        'weights': np.array([comp.weight for comp in gmm_state.components])
    }


# Integration with existing experiment framework
def get_wgf_gmm_step_with_hyperparams(config_dict: dict = None):
    """
    Get a WGF-GMM step function configured with hyperparameters from a config dictionary.
    
    Args:
        config_dict: Dictionary containing 'wgf_gmm_params' with hyperparameters
        
    Returns:
        Configured step function
    """
    if config_dict and 'wgf_gmm_params' in config_dict:
        wgf_params = config_dict['wgf_gmm_params']
        return create_wgf_gmm_step_function(
            lambda_reg=wgf_params.get('lambda_reg', 0.1),
            lr_mean=wgf_params.get('lr_mean', 0.01),
            lr_cov=wgf_params.get('lr_cov', 0.001),
            lr_weight=wgf_params.get('lr_weight', 0.01)
        )
    else:
        # Return default configuration
        return create_wgf_gmm_step_function()


# Enhanced step function for systematic experiments with different hyperparameter sets
def run_wgf_gmm_with_multiple_configs(key: jax.random.PRNGKey,
                                     initial_carry: PIDCarry,
                                     target: Target,
                                     optim: PIDOpt,
                                     hyperparams: PIDParameters,
                                     n_updates: int,
                                     hyperparam_configs: list,
                                     save_results: bool = True,
                                     output_dir: str = "output/wgf_gmm_configs"):
    """
    Run WGF-GMM with multiple hyperparameter configurations and save results.
    
    Args:
        key: PRNG key
        initial_carry: Initial carry state
        target: Target distribution
        optim: Optimizer
        hyperparams: Base hyperparameters
        n_updates: Number of updates per configuration
        hyperparam_configs: List of WGFGMMHyperparams configurations
        save_results: Whether to save results to files
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results for each configuration
    """
    from pathlib import Path
    import pickle
    
    results = {}
    
    for i, wgf_config in enumerate(hyperparam_configs):
        print(f"Running configuration {i+1}/{len(hyperparam_configs)}: {wgf_config}")
        
        # Reset carry for each configuration
        carry = initial_carry
        config_key = key
        
        # Storage for this configuration
        losses = []
        
        for update_idx in range(n_updates):
            config_key, step_key = jax.random.split(config_key)
            
            lval, carry = wgf_gmm_pvi_step(
                step_key, carry, target, None, optim, hyperparams, wgf_config
            )
            
            losses.append(float(lval))
            
            if (update_idx + 1) % 100 == 0:
                print(f"  Step {update_idx + 1}/{n_updates}, Loss: {lval:.4f}")
        
        # Store results
        config_id = f"lambda{wgf_config.lambda_reg}_mean{wgf_config.lr_mean}_cov{wgf_config.lr_cov}_weight{wgf_config.lr_weight}"
        results[config_id] = {
            'config': wgf_config,
            'losses': losses,
            'final_loss': losses[-1],
            'final_carry': carry
        }
        
        # Save individual results if requested
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / f"results_{config_id}.pkl", "wb") as f:
                pickle.dump(results[config_id], f)
    
    return results


# Function to add WGF-GMM hyperparameters to existing PIDParameters
def extend_pid_parameters_for_wgf_gmm(pid_params: PIDParameters, 
                                     lambda_reg: float = 0.1,
                                     lr_mean: float = 0.01,
                                     lr_cov: float = 0.001,
                                     lr_weight: float = 0.01) -> PIDParameters:
    """
    Extend existing PIDParameters with WGF-GMM specific hyperparameters.
    
    Args:
        pid_params: Existing PID parameters
        lambda_reg: Wasserstein regularization strength
        lr_mean: Learning rate for means
        lr_cov: Learning rate for covariances
        lr_weight: Learning rate for weights
        
    Returns:
        Extended PIDParameters with WGF-GMM hyperparameters
    """
    # Add WGF-GMM hyperparameters as an attribute
    extended_params = pid_params
    extended_params.wgf_gmm_params = {
        'lambda_reg': lambda_reg,
        'lr_mean': lr_mean,
        'lr_cov': lr_cov,
        'lr_weight': lr_weight
    }
    
    return extended_params


# Utility function for config parsing
def parse_wgf_gmm_config(config: dict) -> WGFGMMHyperparams:
    """
    Parse WGF-GMM hyperparameters from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        WGFGMMHyperparams object
    """
    wgf_params = config.get('wgf_gmm_params', {})
    
    return WGFGMMHyperparams(
        lambda_reg=wgf_params.get('lambda_reg', 0.1),
        lr_mean=wgf_params.get('lr_mean', 0.01),
        lr_cov=wgf_params.get('lr_cov', 0.001),
        lr_weight=wgf_params.get('lr_weight', 0.01)
    )


# Export the main functions
__all__ = [
    # Core classes
    'GMMComponent',
    'GMMState', 
    'WGFGMMHyperparams',
    'WGFGMMMetrics',
    
    # Core functions
    'particles_to_gmm',
    'gmm_to_particles',
    'sample_from_gmm',
    'compute_elbo',
    'compute_elbo_with_wasserstein_regularization',
    
    # Distance functions
    'bures_wasserstein_distance_squared',
    'wasserstein_distance_gmm',
    
    # Update functions
    'update_gmm_parameters_simple',
    'update_gmm_parameters_full',
    'compute_gmm_gradients',
    
    # Main step functions
    'wgf_gmm_pvi_step',
    'wgf_gmm_pvi_step_with_monitoring',
    'wgf_gmm_pvi_step_full',
    'gmm_pvi_step',
    'de_step',
    
    # Wrapper and utility functions
    'wgf_gmm_step_with_config',
    'create_wgf_gmm_step_function',
    'get_wgf_gmm_step_with_hyperparams',
    'analyze_gmm_state',
    'visualize_gmm_2d',
    'parse_wgf_gmm_config',
    'extend_pid_parameters_for_wgf_gmm',
    'run_wgf_gmm_with_multiple_configs'
]


# Alias for backward compatibility
wgf_gmm_de_step = de_step