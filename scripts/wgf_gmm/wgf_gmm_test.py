#!/usr/bin/env python3
"""
Simple WGF-GMM Test Script
Tests the basic functionality of the WGF-GMM implementation.
"""

import jax
import jax.numpy as np
from src.problems.toy import Banana
from src.utils import make_step_and_carry
from src.base import Parameters, ModelParameters, ThetaOptParameters, ROptParameters, PIDParameters

def test_wgf_gmm_basic():
    """Test basic WGF-GMM functionality."""
    print("Testing basic WGF-GMM functionality...")
    
    # Create a simple problem
    target = Banana()
    key = jax.random.PRNGKey(42)
    
    # Create parameters for WGF-GMM
    parameters = Parameters(
        algorithm='wgf_gmm',
        model_parameters=ModelParameters(
            d_z=2,
            use_particles=True,
            n_particles=10,  # Very small for testing
            kernel='norm_fixed_var_w_skip',
            n_hidden=64     # Small for testing
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
        extra_alg_parameters=PIDParameters(mc_n_samples=25)  # Small for testing
    )
    
    # Initialize
    init_key, train_key = jax.random.split(key)
    
    try:
        step, carry = make_step_and_carry(init_key, parameters, target)
        print("✓ Successfully created step function and carry")
    except Exception as e:
        print(f"✗ Failed to create step function: {e}")
        return False
    
    # Test a few steps
    print("Running 5 training steps...")
    
    for i in range(5):
        train_key, step_key = jax.random.split(train_key)
        
        try:
            lval, carry = step(step_key, carry, target, None)
            print(f"  Step {i+1}: Loss = {lval:.4f}")
            
            # Check for NaN values
            if np.isnan(lval):
                print(f"✗ Loss became NaN at step {i+1}")
                return False
                
        except Exception as e:
            print(f"✗ Error at step {i+1}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("✓ WGF-GMM basic test completed successfully!")
    return True


def test_import_wgf_gmm():
    """Test importing WGF-GMM functions."""
    print("Testing WGF-GMM imports...")
    
    try:
        from src.trainers.wgf_gmm import (
            wgf_gmm_pvi_step,
            wgf_gmm_pvi_step_with_monitoring,
            particles_to_gmm,
            gmm_to_particles,
            compute_elbo,
            WGFGMMMetrics,
            WGFGMMHyperparams
        )
        print("✓ Successfully imported WGF-GMM functions")
        return True
    except ImportError as e:
        print(f"✗ Failed to import WGF-GMM functions: {e}")
        return False


def test_wgf_gmm_monitoring():
    """Test WGF-GMM with monitoring."""
    print("Testing WGF-GMM with monitoring...")
    
    try:
        from src.trainers.wgf_gmm import wgf_gmm_pvi_step_with_monitoring
    except ImportError as e:
        print(f"✗ Cannot import monitoring function: {e}")
        return False
    
    # Create a simple problem
    target = Banana()
    key = jax.random.PRNGKey(42)
    
    parameters = Parameters(
        algorithm='pvi',  # Use regular PVI to get basic setup
        model_parameters=ModelParameters(
            d_z=2,
            use_particles=True,
            n_particles=10,
            kernel='norm_fixed_var_w_skip',
            n_hidden=64
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
        extra_alg_parameters=PIDParameters(mc_n_samples=25)
    )
    
    # Initialize
    init_key, train_key = jax.random.split(key)
    step, carry = make_step_and_carry(init_key, parameters, target)
    
    # Extract optimizer
    optim = None
    if hasattr(step, 'keywords') and 'optim' in step.keywords:
        optim = step.keywords['optim']
    elif hasattr(step, 'func') and hasattr(step.func, 'keywords'):
        optim = step.func.keywords.get('optim')
    
    if optim is None:
        print("✗ Could not extract optimizer")
        return False
    
    # Test monitoring function
    try:
        lval, carry, metrics = wgf_gmm_pvi_step_with_monitoring(
            train_key, carry, target, None, optim, 
            parameters.extra_alg_parameters,
            lambda_reg=0.1, lr_mean=0.01, lr_cov=0.001, lr_weight=0.01
        )
        
        print(f"✓ Monitoring step completed: Loss = {lval:.4f}")
        print(f"  ELBO = {metrics.elbo:.4f}")
        print(f"  Wasserstein distance = {metrics.wasserstein_distance:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Monitoring step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_particles_to_gmm():
    """Test particle-to-GMM conversion."""
    print("Testing particle-to-GMM conversion...")
    
    try:
        from src.trainers.wgf_gmm import particles_to_gmm, gmm_to_particles
    except ImportError as e:
        print(f"✗ Cannot import conversion functions: {e}")
        return False
    
    # Create some test particles
    key = jax.random.PRNGKey(42)
    particles = jax.random.normal(key, (5, 2))  # 5 particles in 2D
    
    try:
        # Convert to GMM
        gmm_state = particles_to_gmm(particles, use_em=False)
        print(f"✓ Converted {len(particles)} particles to GMM with {gmm_state.n_components} components")
        
        # Convert back to particles
        recovered_particles = gmm_to_particles(gmm_state)
        print(f"✓ Converted GMM back to {len(recovered_particles)} particles")
        
        # Check shapes
        if recovered_particles.shape == particles.shape:
            print("✓ Particle shapes match")
        else:
            print(f"✗ Shape mismatch: {recovered_particles.shape} vs {particles.shape}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("WGF-GMM Test Suite")
    print("=" * 40)
    
    tests = [
        test_import_wgf_gmm,
        test_particles_to_gmm,
        test_wgf_gmm_basic,
        test_wgf_gmm_monitoring,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        print(f"\n{test_func.__name__}:")
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! WGF-GMM implementation is working.")
        return True
    else:
        print("✗ Some tests failed. Check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)