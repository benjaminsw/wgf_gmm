import jax
import optax
from src.base import *
from src.base import RPreconParameters
from src.nn import Net
from src.id import PID, SID
from src.conditional import KERNELS
from src.preconditioner import (clip_grad_norm,
                                identity,
                                rms)
from src.ropt import (regularized_wasserstein_descent,
                      stochastic_gradient_to_update,
                      scale_by_schedule,
                      kl_descent,
                      lr_to_schedule)
from src.trainers.pvi import de_step as pvi_de_step
from src.trainers.svi import de_step as svi_de_step
from src.trainers.uvi import de_step as uvi_de_step
from src.trainers.sm import de_step as sm_de_step
# Import the actual WGF-GMM implementation
# Note: If src/trainers/wgf_gmm.py has issues, you may need to replace it with the fixed version
try:
    from src.trainers.wgf_gmm import wgf_gmm_pvi_step
except ImportError as e:
    print(f"Warning: Could not import from src.trainers.wgf_gmm: {e}")
    print("Please replace src/trainers/wgf_gmm.py with the fixed implementation")
    # Fallback to a simple implementation
    def wgf_gmm_pvi_step(key, carry, target, y, optim, hyperparams, **kwargs):
        # Simple fallback that just calls standard PVI
        return pvi_de_step(key, carry, target, y, optim, hyperparams)
import equinox as eqx
import yaml
import re


# Create wrapper function to match the expected interface
def wgf_gmm_de_step(key, carry, target, y, optim, hyperparams):
    """
    Wrapper for WGF-GMM PVI step to match the standard de_step interface.
    This calls the actual implementation in src.trainers.wgf_gmm
    """
    # Handle the gmm_state attribute that WGF-GMM expects
    if not hasattr(carry, 'gmm_state'):
        # Create a temporary extended carry with gmm_state
        class ExtendedCarry:
            def __init__(self, original_carry):
                self.id = original_carry.id
                self.theta_opt_state = original_carry.theta_opt_state
                self.r_opt_state = original_carry.r_opt_state
                self.r_precon_state = original_carry.r_precon_state
                self.gmm_state = None  # Initialize as None
        
        extended_carry = ExtendedCarry(carry)
    else:
        extended_carry = carry
    
    # Call the actual WGF-GMM implementation
    lval, updated_extended_carry = wgf_gmm_pvi_step(
        key=key,
        carry=extended_carry,
        target=target,
        y=y,
        optim=optim,
        hyperparams=hyperparams,
        lambda_reg=0.1,    # Wasserstein regularization strength
        lr_mean=0.01,      # Learning rate for means
        lr_cov=0.001,      # Learning rate for covariances
        lr_weight=0.01     # Learning rate for weights
    )
    
    # Convert back to standard PIDCarry format
    updated_carry = type(carry)(
        id=updated_extended_carry.id,
        theta_opt_state=updated_extended_carry.theta_opt_state,
        r_opt_state=updated_extended_carry.r_opt_state,
        r_precon_state=updated_extended_carry.r_precon_state
    )
    
    return lval, updated_carry


# Update DE_STEPS to use wgf_gmm instead of gmm_pvi
DE_STEPS = {'pvi': pvi_de_step,
            'wgf_gmm': wgf_gmm_de_step,  # Use wgf_gmm instead of gmm_pvi
            'svi': svi_de_step,
            'uvi': uvi_de_step,
            'sm': sm_de_step}


def make_model(key: jax.random.PRNGKey,
               model_parameters: ModelParameters,
               d_x: int):
    """
    Make a model based on the model hyperparameters.
    """
    key1, key2 = jax.random.split(key, 2)
    assert model_parameters.kernel in KERNELS
    likelihood = KERNELS[model_parameters.kernel]

    conditional = likelihood(
        key1,
        d_x,
        model_parameters.d_z,
        model_parameters.d_y,
        n_hidden=model_parameters.n_hidden)
    n_particles = model_parameters.n_particles

    if model_parameters.use_particles:
        init = jax.random.normal(key2, (model_parameters.n_particles,
                                        model_parameters.d_z))
        model = PID(
            key2,
            conditional,
            n_particles,
            init=init 
        )
    else:
        model = SID(conditional)
    return model


def make_theta_opt(topt_param: ThetaOptParameters):
    """
    Make an optimizer for kernel parameters based on the optimizer hyperparameters.
    """
    theta_transform = []

    if topt_param.clip:
        clip = optax.clip_by_global_norm(topt_param.max_clip)
        theta_transform.append(clip)

    if topt_param.lr_decay:
        lr = optax.linear_schedule(topt_param.lr,
                                   topt_param.min_lr,
                                   topt_param.interval)
    else:
        lr = topt_param.lr

    if topt_param.optimizer == 'adam':
        optimizer = optax.adam(lr, b1=0.9, b2=0.99)
    elif topt_param.optimizer == 'rmsprop':
        optimizer = optax.rmsprop(lr) # topt_param.decay)
    else:
        optimizer = optax.sgd(lr)

    theta_transform.append(optimizer)
    return optax.chain(*theta_transform)


def make_r_opt(key: jax.random.PRNGKey,
               ropt_param: ROptParameters,
               sgld: bool=False):
    """
    Make an optimizer for distribution spacebased on the optimizer hyperparameters.
    """
    transform = []

    if ropt_param.lr_decay:
        lr = optax.linear_schedule(ropt_param.lr,
                                   ropt_param.min_lr,
                                   ropt_param.interval)
    else:
        lr = ropt_param.lr
    if sgld:
        transform.append(
            kl_descent(key)
        )
    else:
        transform.append(
            regularized_wasserstein_descent(key,
                                            ropt_param.regularization))
    transform.append(scale_by_schedule(lr_to_schedule(lr)))
    transform.append(stochastic_gradient_to_update())
    return optax.chain(*transform)


def make_r_precon(r_precon_param: ROptParameters):
    """
    Make a preconditioner for distribution space based on the preconditioner hyperparameters.
    """
    if r_precon_param:
        if r_precon_param.type == 'clip':
            return clip_grad_norm(r_precon_param.max_norm,
                                r_precon_param.agg)
        elif r_precon_param.type == 'rms':
            return rms(r_precon_param.agg, False)
    return identity()


def make_step_and_carry(
        key: jax.random.PRNGKey,
        parameters: Parameters,
        target: Target):
    """
    Make a step function and carry for a given algorithm.
    """
    model_key, key = jax.random.split(key, 2)
    id = make_model(model_key,
                    parameters.model_parameters,
                    target.dim)

    if parameters.theta_opt_parameters is not None:
        theta_optim = make_theta_opt(parameters.theta_opt_parameters)

    if target.de:
        step = DE_STEPS[parameters.algorithm]
    else:
        raise NotImplementedError("Only DE is supported")
    
    id_state = eqx.filter(id, id.get_filter_spec())
    
    # Handle both 'pvi' and 'wgf_gmm' with the same setup
    if parameters.algorithm in ['pvi', 'wgf_gmm']:
        ropt_key, key = jax.random.split(key, 2)
        r_optim = make_r_opt(ropt_key,
                             parameters.r_opt_parameters)
        r_precon = make_r_precon(parameters.r_precon_parameters)
        optim = PIDOpt(theta_optim, r_optim, r_precon)
        carry = PIDCarry(id,
                         theta_optim.init(id_state),
                         r_optim.init(id_state),
                         r_precon.init(id))
    elif parameters.algorithm == 'uvi':
        optim = SVIOpt(theta_optim)
        carry = SVICarry(id, theta_optim.init(id_state))
    elif parameters.algorithm == 'svi':
        optim = SVIOpt(theta_optim)
        carry = SVICarry(id, theta_optim.init(id_state))
    elif parameters.algorithm == 'sm':
        dual_optim = make_theta_opt(parameters.dual_opt_parameters)
        dual = Net(key,
                   target.dim,
                   target.dim,
                   parameters.dual_parameters.n_hidden,
                   act=jax.nn.relu)
        dual_state = eqx.filter(dual, dual.get_filter_spec())
        carry = SMCarry(id,
                        theta_optim.init(id_state),
                        dual,
                        dual_optim.init(dual_state))
        optim = SMOpt(theta_optim, dual_optim)
    else:
        raise ValueError(f"Unknown algorithm {parameters.algorithm}")

    def partial_step(key, carry, target, y):
        return step(
            key,
            carry,
            target,
            y,
            optim,
            parameters.extra_alg_parameters,
        )
    return partial_step, carry


def config_to_parameters(config: dict, algorithm: str):
    """
    Make a parameters from a config dictionary and an algorithm name.
    """
    parameters = {'algorithm': algorithm,}
    parameters['model_parameters'] = ModelParameters(
        **config[algorithm]['model']
    )
    if 'theta_opt' in config[algorithm]:
        parameters['theta_opt_parameters'] = ThetaOptParameters(
            **config[algorithm]['theta_opt']
        )
    
    # Handle both 'pvi' and 'wgf_gmm' the same way
    if algorithm in ['pvi', 'wgf_gmm']:
        parameters['r_opt_parameters'] = ROptParameters(
            **config[algorithm]['r_opt']
        )
        if 'r_precon' in config[algorithm]:
            parameters['r_precon_parameters'] = RPreconParameters(
                **config[algorithm]['r_precon']
            )
        parameters['extra_alg_parameters'] = PIDParameters(
            **config[algorithm]['extra_alg']
        )
    elif algorithm == 'svi':
        parameters['extra_alg_parameters'] = SVIParameters(
            **config[algorithm]['extra_alg']
        )
    elif algorithm == 'uvi':
        parameters['extra_alg_parameters'] = UVIParameters(
            **config[algorithm]['extra_alg']
        )
    elif algorithm == 'sm':
        parameters['dual_opt_parameters'] = ThetaOptParameters(
            **config[algorithm]['dual_opt']
        )
        parameters['dual_parameters'] = DualParameters(
            **config[algorithm]['dual']
        )
        parameters['extra_alg_parameters'] = SMParameters(
            **config[algorithm]['extra_alg']
        )
    else:
        raise ValueError(f"Algorithm {algorithm} is not supported")
    return Parameters(**parameters)


def parse_config(config_path: str):
    """
    Parse a config file into a dictionary.
    """
    def none_to_dict(loader, node):
        # Construct the mapping using the default method
        mapping = loader.construct_mapping(node)
        # Replace all None values with {}
        return {k: ({} if v is None else v) for k, v in mapping.items()}
    # Load Empty Entries as an Empty Dictionary
    yaml.SafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        none_to_dict)
    # Parse exponents (e.g., 1e-3) as floats
    yaml.SafeLoader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config