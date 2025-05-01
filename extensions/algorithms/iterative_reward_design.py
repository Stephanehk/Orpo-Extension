import os
from typing import Callable, Optional

from sacred import Experiment
from extensions.reward_modeling.reward_wrapper import RewardWrapper
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv

from extensions.algorithms.train_policy import create_env as original_create_env, ex

# Create a new experiment for iterative reward design
iterative_ex = Experiment("iterative_reward_design", save_git_info=False)

def create_custom_env(config, reward_wrapper_class: Optional[Callable] = None):
    """
    Creates an environment with a configurable reward wrapper.
    
    Args:
        config: The environment configuration
        reward_wrapper_class: Optional custom reward wrapper class to use instead of RewardWrapper
    """
    base_env = PandemicPolicyGymEnv(config)
    wrapper_class = reward_wrapper_class if reward_wrapper_class is not None else RewardWrapper
    return wrapper_class(base_env, reward_model=config.get("reward_model", "default"))

# @iterative_ex.config
# def config():
#     # Default configuration
#     env_to_run = "tomato"
#     experiment_parts = [env_to_run]
#     reward_wrapper_class = None  # Use default RewardWrapper if None

@iterative_ex.automain
def main(env_to_run, experiment_parts, reward_wrapper_class, _log):
    """
    Main function that runs the training with a configurable reward wrapper.
    """
    # Override the create_env function in train_policy with our custom version
    global create_env
    create_env = lambda config: create_custom_env(config, reward_wrapper_class)
    
    # Run the original experiment
    ex.run(
        config_updates={
            "env_to_run": env_to_run,
            "experiment_parts": experiment_parts,
        }
    ) 

