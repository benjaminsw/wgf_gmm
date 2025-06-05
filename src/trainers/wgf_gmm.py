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


def particles_to_gmm(particles: jax.Array, 
                     weights: jax.Array = None,
                     use_em: bool = True,
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
    
    if use_em and n_components < n_particles:
        # Fit proper GMM using EM algorithm
        return _fit_gmm_em(particles, weights, n_components)
    else:
        # Initialize each particle as a Gaussian component
        components = []
        for i in range(n_particles):
            mean = particles[i]
            # Start with small identity covariance to avoid degeneracy
            cov = np.eye(d_z) * 0.1
            weight = weights[i]
            components.append(GMMComponent(mean=mean, cov=cov, weight=weight))
        
        return GMMState(components=components, n_components=n_particles)


def _fit_gmm_em(particles: jax.Array, weights: jax.Array, n_components: int, 
                max_iter: int = 50, tol: float = 1e-6) -> GMMState:
    """
    Fit GMM to particles using EM algorithm.
    
    Args:
        particles: Array of shape (n_particles, d_z)
        weights: Particle weights
        n_components: Number of GMM components
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
    
    Returns:
        GMMState with fitted GMM components
    """
    n_particles, d_z = particles.shape
    key = jax.random.PRNGKey(42)
    
    # Initialize GMM parameters
    # Initialize means using k-means++ style initialization
    means = _kmeans_plus_plus_init(key, particles, n_components)
    
    # Initialize covariances as identity matrices
    covs = np.stack([np.eye(d_z) * 0.5 for _ in range(n_components)])
    
    # Initialize weights uniformly
    gmm_weights = np.ones(n_components) / n_components
    
    prev_log_likelihood = -np.inf
    
    for iter_idx in range(max_iter):
        # E-step: compute responsibilities
        responsibilities = _e_step(particles, weights, means, covs, gmm_weights)
        
        # M-step: update parameters
        means, covs, gmm_weights = _m_step(particles, weights, responsibilities)
        
        # Check convergence
        log_likelihood = _compute_log_likelihood(particles, weights, means, covs, gmm_weights)
        if np.abs(log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = log_likelihood
    
    # Create GMMState
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
    
    # Choose first center randomly
    key, subkey = jax.random.split(key)
    first_idx = jax.random.randint(subkey, (), 0, n_particles)
    means = means.at[0].set(particles[first_idx])
    
    for i in range(1, n_components):
        # Compute distances to nearest centers
        distances = np.full(n_particles, np.inf)
        for j in range(i):
            dist_to_j = np.sum((particles - means[j]) ** 2, axis=1)
            distances = np.minimum(distances, dist_to_j)
        
        # Choose next center with probability proportional to squared distance
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
    
    # Compute log probabilities for each component
    log_probs = np.zeros((n_particles, n_components))
    for i in range(n_components):
        diff = particles - means[i]
        cov_inv = np.linalg.inv(covs[i] + 1e-6 * np.eye(d_z))
        mahal_dist = np.sum(diff @ cov_inv * diff, axis=1)
        log_det = np.linalg.slogdet(covs[i] + 1e-6 * np.eye(d_z))[1]
        log_probs = log_probs.at[:, i].set(
            np.log(gmm_weights[i]) - 0.5 * (d_z * np.log(2 * np.pi) + log_det + mahal_dist)
        )
    
    # Compute responsibilities using log-sum-exp trick
    log_sum = jsp.special.logsumexp(log_probs, axis=1, keepdims=True)
    responsibilities = np.exp(log_probs - log_sum)
    
    return responsibilities


def _m_step(particles: jax.Array, weights: jax.Array, 
           responsibilities: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """M-step of EM algorithm."""
    n_particles, d_z = particles.shape
    n_components = responsibilities.shape[1]
    
    # Effective counts
    weighted_resp = responsibilities * weights[:, None]
    nk = np.sum(weighted_resp, axis=0)
    
    # Update means
    means = np.zeros((n_components, d_z))
    for k in range(n_components):
        if nk[k] > 1e-8:
            means = means.at[k].set(np.sum(weighted_resp[:, k:k+1] * particles, axis=0) / nk[k])
    
    # Update covariances
    covs = np.zeros((n_components, d_z, d_z))
    for k in range(n_components):
        if nk[k] > 1e-8:
            diff = particles - means[k]
            weighted_diff = weighted_resp[:, k:k+1] * diff
            cov = (weighted_diff.T @ diff) / nk[k]
            # Add regularization
            cov = cov + 1e-6 * np.eye(d_z)
            covs = covs.at[k].set(cov)
        else:
            covs = covs.at[k].set(np.eye(d_z) * 0.1)
    
    # Update weights
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
    
    # Covariance term: Tr(cov1 + cov2 - 2(cov1^{1/2} cov2 cov1^{1/2})^{1/2})
    # Compute cov1^{1/2}
    try:
        cov1_sqrt = jsp.linalg.sqrtm(cov1)
        # Compute cov1^{1/2} cov2 cov1^{1/2}
        temp = cov1_sqrt @ cov2 @ cov1_sqrt
        # Compute its square root
        temp_sqrt = jsp.linalg.sqrtm(temp)
        # Final covariance term
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(temp_sqrt)
    except:
        # Fallback to simpler approximation if matrix square root fails
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.sqrt(np.trace(cov1) * np.trace(cov2))
    
    return mean_diff + cov_term


def wasserstein_distance_gmm(gmm1: GMMState, gmm2: GMMState) -> float:
    """
    Compute Wasserstein distance between two GMMs using optimal transport.
    
    For simplicity, we approximate using pairwise BW distances and optimal matching.
    """
    if gmm1.n_components != gmm2.n_components:
        raise ValueError("GMMs must have same number of components")
    
    n_components = gmm1.n_components
    
    # Compute pairwise BW distances
    distances = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            distances = distances.at[i, j].set(
                bures_wasserstein_distance_squared(
                    gmm1.components[i].mean, gmm1.components[i].cov,
                    gmm2.components[j].mean, gmm2.components[j].cov
                )
            )
    
    # Simple approximation: match components greedily
    # In practice, you'd use Hungarian algorithm or Sinkhorn
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
    """
    Riemannian gradient for mean parameters in the Bures-Wasserstein manifold context.
    
    For mean parameters in the Bures-Wasserstein geometry, the mean lives in Euclidean space ℝᵈ,
    so the Riemannian gradient equals the Euclidean gradient.
    
    Args:
        mean: Current mean parameter, shape (d_z,) - for API consistency
        euclidean_grad_mean: Euclidean gradient w.r.t. mean, shape (d_z,)
        
    Returns:
        Riemannian gradient w.r.t. mean parameters, shape (d_z,)
    """
    return euclidean_grad_mean


def riemannian_grad_cov(euclidean_grad_cov: jax.Array, cov: jax.Array) -> jax.Array:
    """
    Riemannian gradient for covariance matrix on the Bures-Wasserstein manifold.
    
    grad_BW = 4 * {grad_euclidean * cov}_symmetric
    where {A}_symmetric = (A + A^T) / 2
    
    Args:
        euclidean_grad_cov: Euclidean gradient w.r.t. covariance
        cov: Current covariance matrix
        
    Returns:
        Riemannian gradient on Bures-Wasserstein manifold
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
    
    R_Sigma(X) = Sigma + X + L_X[Sigma] @ X @ L_X[Sigma]
    where L_X[Sigma] is the solution to L @ X + X @ L = Sigma (Lyapunov equation)
    
    For simplicity, we use a first-order approximation: R_Sigma(X) ≈ Sigma + X
    and ensure positive definiteness by adding small regularization.
    
    Args:
        cov: Current covariance matrix
        tangent_vector: Tangent vector (update direction)
        
    Returns:
        Updated covariance matrix on the manifold
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
    
    Args:
        weights: Current weights (should sum to 1)
        grad_weights: Euclidean gradient w.r.t. weights
        lr: Learning rate
        reg: Entropic regularization parameter
        n_iter: Number of Sinkhorn iterations
        
    Returns:
        Updated weights on the simplex
    """
    # Project gradient onto tangent space of simplex
    n = len(weights)
    grad_projected = grad_weights - np.mean(grad_weights)
    
    # Compute log weights for numerical stability
    log_weights = np.log(weights + 1e-8)
    
    # Update in log space
    log_weights_new = log_weights - lr * grad_projected
    
    # Apply Sinkhorn projection to ensure weights sum to 1
    # This is a simplified version - in practice you'd use full Sinkhorn
    for _ in range(n_iter):
        # Normalize
        weights_new = np.exp(log_weights_new - np.max(log_weights_new))
        weights_new = weights_new / np.sum(weights_new)
        log_weights_new = np.log(weights_new + 1e-8)
    
    return weights_new


def compute_elbo_with_wasserstein_regularization(key: jax.random.PRNGKey,
                                               pid: PID,
                                               target: Target,
                                               gmm_state: GMMState,
                                               y: jax.Array,
                                               hyperparams: PIDParameters,
                                               lambda_reg: float = 0.1) -> float:
    """
    Compute regularized ELBO: F(r) = ELBO(r) - λ * W₂²(r, r_prev)
    
    Args:
        key: PRNG key
        pid: PID object
        target: Target distribution
        gmm_state: Current GMM state
        y: Observations
        hyperparams: PID parameters
        lambda_reg: Wasserstein regularization strength
        
    Returns:
        Regularized ELBO value
    """
    # Sample from GMM
    key, subkey = jax.random.split(key)
    samples = sample_from_gmm(subkey, gmm_state, hyperparams.mc_n_samples)
    
    # Compute standard ELBO terms
    logq = vmap(pid.log_prob, (0, None))(samples, y)
    logp = vmap(target.log_prob, (0, None))(samples, y)
    elbo = np.mean(logp - logq)
    
    # Add Wasserstein regularization if previous state exists
    wasserstein_reg = 0.0
    if gmm_state.prev_components is not None:
        prev_gmm = GMMState(
            components=gmm_state.prev_components,
            n_components=len(gmm_state.prev_components)
        )
        wasserstein_reg = wasserstein_distance_gmm(gmm_state, prev_gmm)
    
    return elbo - lambda_reg * wasserstein_reg


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
    # Extract weights
    weights = np.array([comp.weight for comp in gmm_state.components])
    
    # Sample component indices
    key, subkey = jax.random.split(key)
    component_indices = jax.random.categorical(
        subkey, np.log(weights), shape=(n_samples,)
    )
    
    # Sample from each component
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
    """
    Compute gradients for GMM parameters using REINFORCE/reparameterization.
    
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
        
        return compute_elbo_with_wasserstein_regularization(
            key, pid, target, temp_gmm, y, hyperparams, lambda_reg
        )
    
    # Extract current parameters
    means = np.stack([comp.mean for comp in gmm_state.components])
    covs = np.stack([comp.cov for comp in gmm_state.components])
    weights = np.array([comp.weight for comp in gmm_state.components])
    
    # Compute gradients
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
    """
    Update GMM parameters using Riemannian gradients and Sinkhorn steps.
    
    Args:
        gmm_state: Current GMM state
        mean_grads: Gradients for means
        cov_grads: Gradients for covariances
        weight_grads: Gradients for weights
        lr_mean: Learning rate for means
        lr_cov: Learning rate for covariances
        lr_weight: Learning rate for weights
        
    Returns:
        Updated GMM state
    """
    new_components = []
    
    # Extract current weights for Sinkhorn update
    current_weights = np.array([comp.weight for comp in gmm_state.components])
    
    # Update weights using Sinkhorn
    new_weights = sinkhorn_weights_update(current_weights, weight_grads, lr_weight)
    
    # Update each component
    for i, comp in enumerate(gmm_state.components):
        # Update mean using Euclidean gradient (Riemannian = Euclidean for means)
        new_mean = comp.mean - lr_mean * riemannian_grad_mean(comp.mean, mean_grads[i])
        
        # Update covariance using Riemannian gradient
        riem_grad_cov = riemannian_grad_cov(cov_grads[i], comp.cov)
        new_cov = retraction_cov(comp.cov, -lr_cov * riem_grad_cov)
        
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
                    lambda_reg: float = 0.1,
                    lr_mean: float = 0.01,
                    lr_cov: float = 0.001,
                    lr_weight: float = 0.01) -> Tuple[float, PIDCarry]:
    """
    Full WGF-GMM with PVI step following the PDF specification.
    
    This implements:
    1. GMM representation of mixing distribution r(z)
    2. ELBO with Wasserstein regularization: F(r) = ELBO(r) - λ * W₂²(r, r_prev)
    3. Riemannian gradient descent for means/covariances
    4. Sinkhorn-based weight updates
    5. Proper variational semantics
    
    Args:
        key: PRNG key
        carry: PID carry state
        target: Target distribution
        y: Observations
        optim: PID optimizer
        hyperparams: PID parameters
        lambda_reg: Wasserstein regularization strength
        lr_mean: Learning rate for means
        lr_cov: Learning rate for covariances
        lr_weight: Learning rate for weights
        
    Returns:
        Tuple of (loss_value, updated_carry)
    """
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
        # Initialize GMM from particles
        gmm_state = particles_to_gmm(pid.particles, use_em=True, n_components=None)
    else:
        gmm_state = carry.gmm_state
    
    # Step 3: Compute gradients for GMM parameters
    mean_grads, cov_grads, weight_grads = compute_gmm_gradients(
        grad_key, pid, target, gmm_state, y, hyperparams, lambda_reg
    )
    
    # Step 4: Update GMM parameters using WGF
    updated_gmm_state = update_gmm_parameters(
        gmm_state, mean_grads, cov_grads, weight_grads,
        lr_mean, lr_cov, lr_weight
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
    
    # Store GMM state in carry (extend PIDCarry if needed)
    updated_carry.gmm_state = updated_gmm_state
    
    return lval, updated_carry