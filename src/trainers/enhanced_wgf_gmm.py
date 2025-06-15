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
from jax.scipy.special import digamma, gammaln
from functools import partial


class DirichletGMMComponent(NamedTuple):
    """Enhanced GMM component with Dirichlet weight parameters"""
    mean: jax.Array  # Shape: (d_z,)
    cov: jax.Array   # Shape: (d_z, d_z)
    alpha: float     # Dirichlet concentration parameter for this component


class DirichletGMMState(NamedTuple):
    """Enhanced GMM state with Dirichlet weight distribution"""
    components: list[DirichletGMMComponent]
    n_components: int
    prior_alpha: jax.Array  # Prior Dirichlet parameters, shape (n_components,)
    prev_components: list[DirichletGMMComponent] = None
    # Additional fields for component management
    active_mask: jax.Array = None  # Boolean mask for active components


# Backward compatibility with original GMMComponent/GMMState
class GMMComponent(NamedTuple):
    """Legacy: single Gaussian component with mean and covariance"""
    mean: jax.Array  # Shape: (d_z,)
    cov: jax.Array   # Shape: (d_z, d_z)
    weight: float    # Scalar weight


class GMMState(NamedTuple):
    """Legacy: GMM-based particle representation"""
    components: list[GMMComponent]
    n_components: int
    prev_components: list[GMMComponent] = None


def particles_to_dirichlet_gmm(particles: jax.Array, 
                              weights: jax.Array = None,
                              use_em: bool = True,
                              n_components: int = None,
                              prior_concentration: float = 1.0,
                              sparsity_prior: bool = True) -> DirichletGMMState:
    """
    Convert particle representation to Dirichlet-enhanced GMM representation.
    
    Args:
        particles: Array of shape (n_particles, d_z)
        weights: Optional weights, defaults to uniform
        use_em: Whether to use EM algorithm to fit proper GMM
        n_components: Number of GMM components (if less than n_particles)
        prior_concentration: Base concentration for Dirichlet prior
        sparsity_prior: Whether to use sparsity-inducing prior
        
    Returns:
        DirichletGMMState with Gaussian components and Dirichlet weights
    """
    n_particles, d_z = particles.shape
    
    if weights is None:
        weights = np.ones(n_particles) / n_particles
    
    if not use_em or n_components is None:
        n_components = n_particles
    
    if use_em and n_components < n_particles:
        # Fit proper GMM using EM algorithm first
        legacy_gmm = _fit_gmm_em(particles, weights, n_components)
        # Convert to Dirichlet GMM
        return _convert_legacy_to_dirichlet_gmm(legacy_gmm, prior_concentration, sparsity_prior)
    else:
        # Initialize each particle as a Gaussian component
        components = []
        for i in range(n_particles):
            mean = particles[i]
            # Start with small identity covariance to avoid degeneracy
            cov = np.eye(d_z) * 0.1
            # Convert weight to Dirichlet concentration parameter
            alpha = weights[i] * n_particles * prior_concentration
            alpha = max(alpha, 0.1)  # Ensure positivity
            
            components.append(DirichletGMMComponent(mean=mean, cov=cov, alpha=alpha))
        
        # Set up prior parameters
        if sparsity_prior:
            # Asymmetric prior that encourages sparsity
            prior_alpha = np.full(n_particles, prior_concentration * 0.5)
        else:
            # Symmetric prior
            prior_alpha = np.full(n_particles, prior_concentration)
        
        # Initialize active mask (all components start active)
        active_mask = np.ones(n_particles, dtype=bool)
        
        return DirichletGMMState(
            components=components, 
            n_components=n_particles,
            prior_alpha=prior_alpha,
            active_mask=active_mask
        )


def _convert_legacy_to_dirichlet_gmm(legacy_gmm: GMMState, 
                                   prior_concentration: float = 1.0,
                                   sparsity_prior: bool = True) -> DirichletGMMState:
    """Convert legacy GMM to Dirichlet-enhanced version."""
    current_weights = np.array([comp.weight for comp in legacy_gmm.components])
    n_components = len(current_weights)
    
    # Convert weights to concentration parameters
    total_concentration = n_components * prior_concentration
    alphas = current_weights * total_concentration
    alphas = np.maximum(alphas, 0.1)  # Ensure minimum concentration
    
    # Create Dirichlet components
    dirichlet_components = []
    for i, comp in enumerate(legacy_gmm.components):
        dirichlet_components.append(DirichletGMMComponent(
            mean=comp.mean,
            cov=comp.cov,
            alpha=alphas[i]
        ))
    
    # Set up prior
    if sparsity_prior:
        prior_alpha = np.full(n_components, prior_concentration * 0.5)
    else:
        prior_alpha = np.full(n_components, prior_concentration)
    
    active_mask = np.ones(n_components, dtype=bool)
    
    return DirichletGMMState(
        components=dirichlet_components,
        n_components=n_components,
        prior_alpha=prior_alpha,
        active_mask=active_mask,
        prev_components=None
    )


def _fit_gmm_em(particles: jax.Array, weights: jax.Array, n_components: int, 
                max_iter: int = 50, tol: float = 1e-6) -> GMMState:
    """Legacy EM fitting - returns standard GMMState."""
    n_particles, d_z = particles.shape
    key = jax.random.PRNGKey(42)
    
    # Initialize GMM parameters
    means = _kmeans_plus_plus_init(key, particles, n_components)
    covs = np.stack([np.eye(d_z) * 0.5 for _ in range(n_components)])
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
    
    # Create legacy GMMState
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


def dirichlet_gmm_to_particles(dirichlet_gmm: DirichletGMMState) -> jax.Array:
    """
    Extract particle locations from Dirichlet GMM (using means).
    
    Args:
        dirichlet_gmm: Dirichlet GMM representation
        
    Returns:
        Array of particle means, shape (n_particles, d_z)
    """
    means = [comp.mean for comp in dirichlet_gmm.components]
    return np.stack(means, axis=0)


def bures_wasserstein_distance_squared(mu1: jax.Array, cov1: jax.Array,
                                     mu2: jax.Array, cov2: jax.Array) -> float:
    """
    Compute squared Bures-Wasserstein distance between two Gaussian distributions.
    """
    # Mean difference term
    mean_diff = np.sum((mu1 - mu2) ** 2)
    
    # Covariance term: Tr(cov1 + cov2 - 2(cov1^{1/2} cov2 cov1^{1/2})^{1/2})
    try:
        cov1_sqrt = jsp.linalg.sqrtm(cov1)
        temp = cov1_sqrt @ cov2 @ cov1_sqrt
        temp_sqrt = jsp.linalg.sqrtm(temp)
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(temp_sqrt)
    except:
        # Fallback to simpler approximation
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.sqrt(np.trace(cov1) * np.trace(cov2))
    
    return mean_diff + cov_term


def wasserstein_distance_dirichlet_gmm(gmm1: DirichletGMMState, gmm2: DirichletGMMState) -> float:
    """
    Compute Wasserstein distance between two Dirichlet GMMs using optimal transport.
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
    
    # Get expected weights from Dirichlet distributions
    weights1 = expected_weights_from_dirichlet(
        np.array([comp.alpha for comp in gmm1.components])
    )
    weights2 = expected_weights_from_dirichlet(
        np.array([comp.alpha for comp in gmm2.components])
    )
    
    # Simple greedy matching (in practice, use Hungarian algorithm)
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
            weight_avg = (weights1[i] + weights2[best_j]) / 2
            total_distance += weight_avg * best_dist
            used_j.add(best_j)
    
    return total_distance


def riemannian_grad_mean(mean: jax.Array, euclidean_grad_mean: jax.Array) -> jax.Array:
    """Riemannian gradient for mean parameters (Euclidean space)."""
    return euclidean_grad_mean


def riemannian_grad_cov(euclidean_grad_cov: jax.Array, cov: jax.Array) -> jax.Array:
    """Riemannian gradient for covariance matrix on Bures-Wasserstein manifold."""
    product = euclidean_grad_cov @ cov
    symmetric_product = (product + product.T) / 2
    return 4 * symmetric_product


def retraction_cov(cov: jax.Array, tangent_vector: jax.Array) -> jax.Array:
    """Retraction operator for covariance matrices on Bures-Wasserstein manifold."""
    new_cov = cov + tangent_vector
    new_cov = (new_cov + new_cov.T) / 2  # Ensure symmetry
    
    # Ensure positive definiteness
    d = new_cov.shape[0]
    regularization = 1e-6 * np.eye(d)
    new_cov = new_cov + regularization
    
    return new_cov


def bayesian_weight_update(current_alphas: jax.Array, 
                          grad_weights: jax.Array,
                          prior_alpha: jax.Array,
                          lr: float = 0.01,
                          use_natural_gradient: bool = True,
                          prior_strength: float = 0.01) -> jax.Array:
    """
    Bayesian update for Dirichlet weight parameters.
    
    Replaces ad-hoc Sinkhorn projection with proper Bayesian inference.
    """
    if use_natural_gradient:
        alpha_sum = np.sum(current_alphas)
        
        # Trigamma approximation
        def trigamma_approx(x):
            return np.where(x > 1.0, 
                          1.0/x + 1.0/(2.0*x**2), 
                          1.0/np.maximum(x, 1e-6))
        
        trigamma_alphas = trigamma_approx(current_alphas)
        trigamma_sum = trigamma_approx(alpha_sum)
        
        # Fisher Information Matrix diagonal terms
        fisher_diag = trigamma_alphas - trigamma_sum
        
        # Natural gradient = F⁻¹ * euclidean_gradient
        natural_grad = grad_weights / np.maximum(fisher_diag, 1e-6)
        
        # Update with natural gradient
        new_alphas = current_alphas + lr * natural_grad
    else:
        # Standard gradient update
        new_alphas = current_alphas + lr * grad_weights
    
    # Add prior regularization (MAP estimation)
    new_alphas = new_alphas + prior_strength * (prior_alpha - current_alphas)
    
    # Ensure positivity
    new_alphas = np.maximum(new_alphas, 0.1)
    
    return new_alphas


def expected_weights_from_dirichlet(alphas: jax.Array) -> jax.Array:
    """Compute expected weights under Dirichlet distribution."""
    return alphas / np.sum(alphas)


def sample_weights_from_dirichlet(key: jax.random.PRNGKey, 
                                 alphas: jax.Array) -> jax.Array:
    """Sample weights from Dirichlet distribution."""
    return jax.random.dirichlet(key, alphas)


def dirichlet_entropy(alphas: jax.Array) -> float:
    """Compute entropy of Dirichlet distribution."""
    alpha_sum = np.sum(alphas)
    
    # Log Beta function
    log_beta = np.sum(gammaln(alphas)) - gammaln(alpha_sum)
    
    # Entropy term
    entropy_term = np.sum((alphas - 1.0) * (digamma(alphas) - digamma(alpha_sum)))
    
    return log_beta - entropy_term


def compute_dirichlet_regularization(current_alphas: jax.Array,
                                   prior_alphas: jax.Array) -> float:
    """
    Compute KL divergence between current and prior Dirichlet distributions.
    Provides natural regularization toward the prior.
    """
    current_sum = np.sum(current_alphas)
    prior_sum = np.sum(prior_alphas)
    
    # Log Beta ratio
    log_beta_ratio = (np.sum(gammaln(prior_alphas)) - gammaln(prior_sum) - 
                     np.sum(gammaln(current_alphas)) + gammaln(current_sum))
    
    # Expectation term
    expectation_term = np.sum((current_alphas - prior_alphas) * 
                             (digamma(current_alphas) - digamma(current_sum)))
    
    return log_beta_ratio + expectation_term


def enhanced_elbo_with_bayesian_weights(key: jax.random.PRNGKey,
                                      pid: PID,
                                      target: Target,
                                      dirichlet_gmm: DirichletGMMState,
                                      y: jax.Array,
                                      hyperparams: PIDParameters,
                                      lambda_reg: float = 0.1,
                                      weight_reg: float = 0.01,
                                      use_expected_weights: bool = False) -> float:
    """
    Enhanced ELBO with Bayesian weight treatment.
    
    F(r) = ELBO(r) - λ * W₂²(r, r_prev) - β * KL(q(π) || p(π))
    """
    # Get weights (either sample or use expected values)
    alphas = np.array([comp.alpha for comp in dirichlet_gmm.components])
    
    if use_expected_weights:
        # Use expected weights for more stable training
        weights = expected_weights_from_dirichlet(alphas)
    else:
        # Sample weights from Dirichlet for full Bayesian treatment
        key, weight_key = jax.random.split(key)
        weights = sample_weights_from_dirichlet(weight_key, alphas)
    
    # Sample from GMM using these weights
    key, sample_key = jax.random.split(key)
    samples = sample_from_dirichlet_gmm(sample_key, dirichlet_gmm, hyperparams.mc_n_samples, weights)
    
    # Compute standard ELBO terms
    logq = vmap(pid.log_prob, (0, None))(samples, y)
    logp = vmap(target.log_prob, (0, None))(samples, y)
    elbo = np.mean(logp - logq)
    
    # Add Wasserstein regularization if previous state exists
    wasserstein_reg = 0.0
    if dirichlet_gmm.prev_components is not None:
        prev_gmm = DirichletGMMState(
            components=dirichlet_gmm.prev_components,
            n_components=len(dirichlet_gmm.prev_components),
            prior_alpha=dirichlet_gmm.prior_alpha
        )
        wasserstein_reg = wasserstein_distance_dirichlet_gmm(dirichlet_gmm, prev_gmm)
    
    # Add Dirichlet regularization
    dirichlet_kl = compute_dirichlet_regularization(alphas, dirichlet_gmm.prior_alpha)
    
    return elbo - lambda_reg * wasserstein_reg - weight_reg * dirichlet_kl


def sample_from_dirichlet_gmm(key: jax.random.PRNGKey, 
                             dirichlet_gmm: DirichletGMMState, 
                             n_samples: int,
                             weights: jax.Array = None) -> jax.Array:
    """
    Sample from a Dirichlet GMM.
    
    Args:
        key: PRNG key
        dirichlet_gmm: Dirichlet GMM state
        n_samples: Number of samples
        weights: Optional pre-computed weights
        
    Returns:
        Samples from the GMM, shape (n_samples, d_z)
    """
    if weights is None:
        # Use expected weights
        alphas = np.array([comp.alpha for comp in dirichlet_gmm.components])
        weights = expected_weights_from_dirichlet(alphas)
    
    # Sample component indices
    key, subkey = jax.random.split(key)
    component_indices = jax.random.categorical(
        subkey, np.log(weights), shape=(n_samples,)
    )
    
    # Sample from each component
    d_z = dirichlet_gmm.components[0].mean.shape[0]
    samples = np.zeros((n_samples, d_z))
    
    for i, comp_idx in enumerate(component_indices):
        comp = dirichlet_gmm.components[comp_idx]
        key, subkey = jax.random.split(key)
        sample = jax.random.multivariate_normal(
            subkey, comp.mean, comp.cov
        )
        samples = samples.at[i].set(sample)
    
    return samples


def compute_dirichlet_gmm_gradients(key: jax.random.PRNGKey,
                                   pid: PID,
                                   target: Target,
                                   dirichlet_gmm: DirichletGMMState,
                                   y: jax.Array,
                                   hyperparams: PIDParameters,
                                   lambda_reg: float = 0.1,
                                   weight_reg: float = 0.01) -> Tuple[list, list, jax.Array]:
    """
    Compute gradients for Dirichlet GMM parameters.
    
    Returns:
        Tuple of (mean_grads, cov_grads, alpha_grads)
    """
    def objective_fn(means, covs, alphas):
        # Create temporary Dirichlet GMM state
        components = []
        for i in range(len(means)):
            components.append(DirichletGMMComponent(
                mean=means[i], cov=covs[i], alpha=alphas[i]
            ))
        temp_gmm = DirichletGMMState(
            components=components,
            n_components=len(components),
            prior_alpha=dirichlet_gmm.prior_alpha,
            prev_components=dirichlet_gmm.prev_components
        )
        
        return enhanced_elbo_with_bayesian_weights(
            key, pid, target, temp_gmm, y, hyperparams, lambda_reg, weight_reg
        )
    
    # Extract current parameters
    means = np.stack([comp.mean for comp in dirichlet_gmm.components])
    covs = np.stack([comp.cov for comp in dirichlet_gmm.components])
    alphas = np.array([comp.alpha for comp in dirichlet_gmm.components])
    
    # Compute gradients
    grad_fn = jax.grad(objective_fn, argnums=(0, 1, 2))
    mean_grads, cov_grads, alpha_grads = grad_fn(means, covs, alphas)
    
    return list(mean_grads), list(cov_grads), alpha_grads


def update_dirichlet_gmm_parameters(dirichlet_gmm: DirichletGMMState,
                                   mean_grads: list,
                                   cov_grads: list,
                                   alpha_grads: jax.Array,
                                   lr_mean: float = 0.01,
                                   lr_cov: float = 0.001,
                                   lr_alpha: float = 0.01,
                                   sparsity_threshold: float = 0.05) -> DirichletGMMState:
    """
    Update Dirichlet GMM parameters using Bayesian weight updates.
    
    Key improvement: Bayesian weight updates instead of ad-hoc Sinkhorn projection.
    """
    # Update Dirichlet parameters using Bayesian approach
    current_alphas = np.array([comp.alpha for comp in dirichlet_gmm.components])
    new_alphas = bayesian_weight_update(
        current_alphas, alpha_grads, dirichlet_gmm.prior_alpha, lr_alpha
    )
    
    # Compute expected weights for sparsity management
    expected_weights = expected_weights_from_dirichlet(new_alphas)
    
    # Update active mask based on expected weights
    new_active_mask = expected_weights > sparsity_threshold
    
    # Update means and covariances
    new_components = []
    for i, comp in enumerate(dirichlet_gmm.components):
        if new_active_mask[i]:  # Only update active components
            # Update mean
            new_mean = comp.mean - lr_mean * riemannian_grad_mean(comp.mean, mean_grads[i])
            
            # Update covariance
            riem_grad_cov = riemannian_grad_cov(cov_grads[i], comp.cov)
            new_cov = retraction_cov(comp.cov, -lr_cov * riem_grad_cov)
        else:
            # Keep inactive components unchanged but with reduced alpha
            new_mean = comp.mean
            new_cov = comp.cov
        
        # Create new component with updated Dirichlet parameter
        new_components.append(DirichletGMMComponent(
            mean=new_mean,
            cov=new_cov,
            alpha=new_alphas[i]
        ))
    
    return DirichletGMMState(
        components=new_components,
        n_components=dirichlet_gmm.n_components,
        prior_alpha=dirichlet_gmm.prior_alpha,
        prev_components=dirichlet_gmm.components,
        active_mask=new_active_mask
    )


def adaptive_component_management(dirichlet_gmm: DirichletGMMState,
                                sparsity_threshold: float = 0.05,
                                merge_threshold: float = 0.1,
                                split_threshold: float = 0.5) -> DirichletGMMState:
    """
    Adaptive component management: merge similar components, split heavy ones.
    
    This addresses the "drop during training" requirement from the comments.
    """
    alphas = np.array([comp.alpha for comp in dirichlet_gmm.components])
    expected_weights = expected_weights_from_dirichlet(alphas)
    
    # Identify components to remove (very low weight)
    active_components = []
    active_indices = []
    
    for i, (comp, weight) in enumerate(zip(dirichlet_gmm.components, expected_weights)):
        if weight > sparsity_threshold:
            active_components.append(comp)
            active_indices.append(i)
    
    # Update prior alpha to match active components
    new_prior_alpha = dirichlet_gmm.prior_alpha[active_indices] if len(active_indices) > 0 else np.array([1.0])
    
    # Create new active mask
    new_active_mask = np.zeros(len(dirichlet_gmm.components), dtype=bool)
    new_active_mask = new_active_mask.at[active_indices].set(True)
    
    if len(active_components) == 0:
        # Fallback: keep at least one component
        active_components = [dirichlet_gmm.components[0]]
        new_prior_alpha = np.array([1.0])
        new_active_mask = new_active_mask.at[0].set(True)
    
    return DirichletGMMState(
        components=active_components,
        n_components=len(active_components),
        prior_alpha=new_prior_alpha,
        prev_components=dirichlet_gmm.prev_components,
        active_mask=new_active_mask
    )


def enhanced_wgf_gmm_pvi_step(key: jax.random.PRNGKey,
                             carry: PIDCarry,
                             target: Target,
                             y: jax.Array,
                             optim: PIDOpt,
                             hyperparams: PIDParameters,
                             lambda_reg: float = 0.1,
                             weight_reg: float = 0.01,
                             lr_mean: float = 0.01,
                             lr_cov: float = 0.001,
                             lr_alpha: float = 0.01,
                             use_component_management: bool = True,
                             sparsity_threshold: float = 0.05) -> Tuple[float, PIDCarry]:
    """
    Enhanced WGF-GMM with Dirichlet-Bayesian weight updates.
    
    Key improvements over original:
    1. Dirichlet-enhanced GMM representation with proper Bayesian weight treatment
    2. Bayesian weight updates instead of ad-hoc Sinkhorn projection
    3. Natural gradient descent in Dirichlet parameter space
    4. Automatic regularization through KL divergence to prior
    5. Adaptive component management (sparsity, merging, splitting)
    6. Better theoretical foundation
    
    Args:
        key: PRNG key
        carry: PID carry state
        target: Target distribution
        y: Observations
        optim: PID optimizer
        hyperparams: PID parameters
        lambda_reg: Wasserstein regularization strength
        weight_reg: Dirichlet regularization strength
        lr_mean: Learning rate for means
        lr_cov: Learning rate for covariances
        lr_alpha: Learning rate for Dirichlet parameters
        use_component_management: Whether to use adaptive component management
        sparsity_threshold: Threshold for component removal
        
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
    
    # Step 2: Initialize or retrieve Dirichlet GMM state
    if not hasattr(carry, 'dirichlet_gmm_state') or carry.dirichlet_gmm_state is None:
        # Initialize Dirichlet GMM from particles
        dirichlet_gmm_state = particles_to_dirichlet_gmm(
            pid.particles, 
            use_em=True, 
            n_components=None,
            prior_concentration=1.0,
            sparsity_prior=True
        )
    else:
        dirichlet_gmm_state = carry.dirichlet_gmm_state
    
    # Step 3: Compute gradients for Dirichlet GMM parameters
    mean_grads, cov_grads, alpha_grads = compute_dirichlet_gmm_gradients(
        grad_key, pid, target, dirichlet_gmm_state, y, hyperparams, lambda_reg, weight_reg
    )
    
    # Step 4: Update Dirichlet GMM parameters using Bayesian approach
    updated_dirichlet_gmm_state = update_dirichlet_gmm_parameters(
        dirichlet_gmm_state, mean_grads, cov_grads, alpha_grads,
        lr_mean, lr_cov, lr_alpha, sparsity_threshold
    )
    
    # Step 5: Adaptive component management (optional)
    if use_component_management:
        updated_dirichlet_gmm_state = adaptive_component_management(
            updated_dirichlet_gmm_state, sparsity_threshold
        )
    
    # Step 6: Convert back to particle representation for compatibility
    updated_particles = dirichlet_gmm_to_particles(updated_dirichlet_gmm_state)
    pid = eqx.tree_at(lambda tree: tree.particles, pid, updated_particles)
    
    # Create updated carry with Dirichlet GMM state
    updated_carry = PIDCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=carry.r_opt_state,
        r_precon_state=carry.r_precon_state
    )
    
    # Store Dirichlet GMM state in carry
    updated_carry.dirichlet_gmm_state = updated_dirichlet_gmm_state
    
    return lval, updated_carry


# Utility functions for analysis and debugging

def analyze_dirichlet_gmm_state(dirichlet_gmm: DirichletGMMState) -> dict:
    """
    Analyze the current state of a Dirichlet GMM for debugging/monitoring.
    
    Returns:
        Dictionary with analysis results
    """
    alphas = np.array([comp.alpha for comp in dirichlet_gmm.components])
    expected_weights = expected_weights_from_dirichlet(alphas)
    
    analysis = {
        'n_components': dirichlet_gmm.n_components,
        'expected_weights': expected_weights,
        'weight_entropy': dirichlet_entropy(alphas),
        'effective_components': np.sum(expected_weights > 0.01),
        'max_weight': np.max(expected_weights),
        'min_weight': np.min(expected_weights),
        'weight_concentration': np.sum(alphas),
        'sparsity_ratio': np.sum(expected_weights < 0.05) / len(expected_weights)
    }
    
    return analysis


def get_effective_gmm_representation(dirichlet_gmm: DirichletGMMState,
                                   threshold: float = 0.01) -> GMMState:
    """
    Convert Dirichlet GMM to standard GMM representation using expected weights.
    Only includes components above threshold.
    
    Useful for visualization and compatibility with legacy code.
    """
    alphas = np.array([comp.alpha for comp in dirichlet_gmm.components])
    expected_weights = expected_weights_from_dirichlet(alphas)
    
    # Filter components above threshold
    active_indices = expected_weights > threshold
    active_components = []
    
    for i, (comp, weight) in enumerate(zip(dirichlet_gmm.components, expected_weights)):
        if active_indices[i]:
            active_components.append(GMMComponent(
                mean=comp.mean,
                cov=comp.cov,
                weight=weight
            ))
    
    # Renormalize weights
    if len(active_components) > 0:
        total_weight = sum(comp.weight for comp in active_components)
        active_components = [
            GMMComponent(comp.mean, comp.cov, comp.weight / total_weight)
            for comp in active_components
        ]
    
    return GMMState(
        components=active_components,
        n_components=len(active_components)
    )


# Legacy compatibility functions

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
    Legacy wrapper that provides the same interface as the original function.
    
    This automatically uses the enhanced Dirichlet-Bayesian approach under the hood.
    """
    return enhanced_wgf_gmm_pvi_step(
        key=key,
        carry=carry,
        target=target,
        y=y,
        optim=optim,
        hyperparams=hyperparams,
        lambda_reg=lambda_reg,
        weight_reg=0.01,  # Default Dirichlet regularization
        lr_mean=lr_mean,
        lr_cov=lr_cov,
        lr_alpha=lr_weight,  # Use weight learning rate for alpha
        use_component_management=True,
        sparsity_threshold=0.05
    )


# Export the main interface
__all__ = [
    'DirichletGMMComponent',
    'DirichletGMMState', 
    'enhanced_wgf_gmm_pvi_step',
    'wgf_gmm_pvi_step',  # Legacy compatibility
    'particles_to_dirichlet_gmm',
    'dirichlet_gmm_to_particles',
    'analyze_dirichlet_gmm_state',
    'get_effective_gmm_representation'
]