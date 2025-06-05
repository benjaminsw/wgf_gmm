import jax
from jax import vmap
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
import jax.scipy as jsp
from functools import partial


class GMMComponent(NamedTuple):
    """Represents a single Gaussian component with mean and covariance"""
    mean: jax.Array  # Shape: (d_z,)
    cov: jax.Array   # Shape: (d_z, d_z)
    weight: float    # Scalar weight


class GMMState(NamedTuple):
    """State for GMM-based particle representation"""
    components: jax.Array  # Array instead of list for JAX compatibility
    n_components: int
    prev_components: jax.Array = None  # For Wasserstein regularization


def particles_to_gmm_simple(particles: jax.Array, 
                           weights: jax.Array = None) -> GMMState:
    """
    Convert particle representation to GMM representation.
    Simplified version that avoids complex EM fitting.
    
    Args:
        particles: Array of shape (n_particles, d_z)
        weights: Optional weights, defaults to uniform
    
    Returns:
        GMMState with Gaussian components
    """
    n_particles, d_z = particles.shape
    
    if weights is None:
        weights = np.ones(n_particles) / n_particles
    
    # Initialize each particle as a Gaussian component with small covariance
    means = particles  # Shape: (n_particles, d_z)
    covs = np.tile(np.eye(d_z) * 0.1, (n_particles, 1, 1))  # Shape: (n_particles, d_z, d_z)
    
    # Pack into arrays for JAX compatibility
    components = np.stack([means, covs.reshape(n_particles, -1), weights], axis=-1)
    
    return GMMState(components=components, n_components=n_particles)


def gmm_to_particles(gmm_state: GMMState) -> jax.Array:
    """
    Extract particle locations from GMM (using means).
    
    Args:
        gmm_state: GMM representation
        
    Returns:
        Array of particle means, shape (n_particles, d_z)
    """
    # Extract means from the packed components array
    means = gmm_state.components[:, 0, :]  # Assuming first element is means
    return means


def compute_elbo_with_wasserstein_regularization(key: jax.random.PRNGKey,
                                               pid: PID,
                                               target: Target,
                                               gmm_state: GMMState,
                                               y: jax.Array,
                                               hyperparams: PIDParameters,
                                               lambda_reg: float = 0.1) -> float:
    """
    Compute regularized ELBO: F(r) = ELBO(r) - λ * W₂²(r, r_prev)
    Simplified version without complex Wasserstein distance computation.
    """
    # Sample from current particles (treating GMM as particle approximation)
    particles = gmm_to_particles(gmm_state)
    key, subkey = jax.random.split(key)
    
    # Use particles directly to compute ELBO
    samples = pid.sample(subkey, hyperparams.mc_n_samples, None)
    
    # Compute standard ELBO terms
    logq = vmap(pid.log_prob, (0, None))(samples, y)
    logp = vmap(target.log_prob, (0, None))(samples, y)
    elbo = np.mean(logp - logq)
    
    # Add simple regularization based on particle spread
    wasserstein_reg = 0.0
    if gmm_state.prev_components is not None:
        prev_particles = gmm_to_particles(GMMState(
            components=gmm_state.prev_components,
            n_components=gmm_state.n_components
        ))
        # Simple L2 distance as proxy for Wasserstein distance
        wasserstein_reg = np.mean(np.sum((particles - prev_particles) ** 2, axis=1))
    
    return elbo - lambda_reg * wasserstein_reg


def update_gmm_simple(gmm_state: GMMState,
                     particle_updates: jax.Array) -> GMMState:
    """
    Update GMM parameters using simple particle updates.
    
    Args:
        gmm_state: Current GMM state
        particle_updates: Updates for particle positions
        
    Returns:
        Updated GMM state
    """
    # Extract current means
    current_means = gmm_to_particles(gmm_state)
    
    # Apply updates to means
    new_means = current_means + particle_updates
    
    # Keep covariances and weights the same for simplicity
    d_z = current_means.shape[1]
    n_particles = current_means.shape[0]
    weights = np.ones(n_particles) / n_particles
    covs = np.tile(np.eye(d_z) * 0.1, (n_particles, 1, 1))
    
    # Pack into new components array
    new_components = np.stack([new_means, covs.reshape(n_particles, -1), weights], axis=-1)
    
    # Store current components as previous for next iteration
    return GMMState(
        components=new_components,
        n_components=gmm_state.n_components,
        prev_components=gmm_state.components
    )


def wgf_gmm_pvi_step(key: jax.random.PRNGKey,
                    carry,
                    target: Target,
                    y: jax.Array,
                    optim: PIDOpt,
                    hyperparams: PIDParameters,
                    lambda_reg: float = 0.1,
                    lr_mean: float = 0.01,
                    lr_cov: float = 0.001,
                    lr_weight: float = 0.01) -> Tuple[float, any]:
    """
    Simplified WGF-GMM with PVI step.
    
    This implements a simplified version that:
    1. Converts particles to GMM representation
    2. Computes ELBO with simple Wasserstein regularization
    3. Updates particles using standard PVI gradients
    4. Converts back to particle representation
    """
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
    if carry.gmm_state is None:
        # Initialize GMM from particles
        gmm_state = particles_to_gmm_simple(pid.particles)
    else:
        gmm_state = carry.gmm_state
    
    # Step 3: Compute particle gradients with WGF-inspired updates
    def particle_grad_with_wgf(particles):
        """Compute gradients for particles with WGF regularization."""
        
        # Standard PVI gradient
        def ediff_score(particle, eps):
            vf = vmap(pid.conditional.f, (None, None, 0))
            samples = vf(particle, y, eps)
            logq = vmap(pid.log_prob, (0, None))(samples, y)
            logp = vmap(target.log_prob, (0, None))(samples, y)
            logp_mean = np.mean(logp, 0)
            logq_mean = np.mean(logq, 0)
            return logq_mean - logp_mean
        
        eps = pid.conditional.base_sample(r_key, hyperparams.mc_n_samples)
        grad_standard = vmap(jax.grad(lambda p: ediff_score(p, eps)))(particles)
        
        # Add WGF-inspired regularization
        if gmm_state.prev_components is not None:
            prev_particles = gmm_to_particles(GMMState(
                components=gmm_state.prev_components,
                n_components=gmm_state.n_components
            ))
            grad_wgf = lambda_reg * (particles - prev_particles)
        else:
            grad_wgf = np.zeros_like(particles)
        
        return grad_standard + grad_wgf
    
    # Apply preconditioner and optimizer
    g_grad, r_precon_state = optim.r_precon.update(
        pid.particles,
        particle_grad_with_wgf,
        carry.r_precon_state,
    )
    
    update, r_opt_state = optim.r_optim.update(
        g_grad,
        carry.r_opt_state,
        params=pid.particles,
        index=y
    )
    
    # Step 4: Update particles and GMM state
    updated_particles = pid.particles + update
    pid = eqx.tree_at(lambda tree: tree.particles, pid, updated_particles)
    
    # Update GMM state
    updated_gmm_state = update_gmm_simple(gmm_state, update)
    
    # Create updated carry with GMM state
    # We need to handle the gmm_state attribute
    class UpdatedCarry:
        def __init__(self, id, theta_opt_state, r_opt_state, r_precon_state, gmm_state):
            self.id = id
            self.theta_opt_state = theta_opt_state
            self.r_opt_state = r_opt_state
            self.r_precon_state = r_precon_state
            self.gmm_state = gmm_state
    
    updated_carry = UpdatedCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=r_opt_state,
        r_precon_state=r_precon_state,
        gmm_state=updated_gmm_state
    )
    
    return lval, updated_carry