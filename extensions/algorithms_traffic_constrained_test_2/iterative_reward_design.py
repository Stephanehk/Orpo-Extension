
import os

from sacred import Experiment
from occupancy_measures.agents.orpo import ORPO, ORPOPolicy
from extensions.utils.random_action_policy_wrapper import RandomActionPolicy
from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info
from occupancy_measures.agents.orpo import ORPO
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import make_multi_agent
from flow.utils.registry import make_create_env

from extensions.reward_modeling.reward_wrapper import RewardWrapper,RewardModel
from occupancy_measures.experiments.traffic_experiments import create_traffic_config
from extensions.algorithms_traffic_constrained_test_2.dup_configs import dup_configs

from extensions.algorithms_traffic_constrained_test_2.train_policy import ex
import extensions.algorithms_traffic_constrained_test_2.train_policy
import extensions.algorithms_traffic_constrained_test_2.unique_id_state as unique_id_state
import time

import warnings
warnings.filterwarnings("ignore")


iterative_ex = Experiment("algorithms_traffic_constrained_test_2", save_git_info=False)

def replace_default_policy(policy_id, policy):
    from extensions.utils.random_action_policy_wrapper import RandomActionPolicy
    if policy_id == "default_policy" or policy_id == "safe_policy0":
        return RandomActionPolicy(policy)  # assumes this wraps an existing policy
    return policy

@iterative_ex.config
def config():
    # Default configuration matching your typical train_policy arguments
    env_to_run = "pandemic"
    level=4
    reward_fun = "proxy"
    exp_algo = "ORPO"
    om_divergence_coeffs_1 = ["0.06"]
    checkpoint_to_load_policies = None
    checkpoint_to_load_current_policy = None
    seed = 0
    experiment_tag = "state"
    om_divergence_type = ["kl"]
    num_rollout_workers = 2
    num_gpus = 1
    experiment_parts = [env_to_run]
    reward_wrapper_class = None  # Use default RewardWrapper if None
    num_training_iters_1 = 260,

# @iterative_ex.automain
@iterative_ex.automain

def main(
    env_to_run,
    level,
    reward_fun,
    exp_algo,
    om_divergence_coeffs_1,
    checkpoint_to_load_policies,
    checkpoint_to_load_current_policy,
    seed,
    experiment_tag,
    om_divergence_type,
    num_rollout_workers,
    num_gpus,
    experiment_parts,
    reward_wrapper_class,
    num_training_iters_1,
    _log
):
    """
    Main function that runs the training with a configurable reward wrapper.
    """

    # unique_id_state.state["unique_id"] = f"{reward_fun}_{seed}_{int(time.time())}"
    # random_checkpoint_path = checkpoint_to_load_current_policy
    #------------------------------------
    # flow_params,reward_specification,reward_fun,reward_scale= dup_configs(experiment_parts=experiment_parts)
    # create_env, env_name = make_create_env(
    #     params=flow_params,
    #     reward_specification=reward_specification,
    #     reward_fun=reward_fun,
    #     reward_scale=reward_scale,
    # )
    # print (f"env_name: {env_name}")
    # register_env(env_name, make_multi_agent(create_env))

    # # random_checkpoint_path = checkpoint_to_load_current_policy
    # # algorithm = Algorithm.from_checkpoint(checkpoint_to_load_current_policy)
    # checkpoint_info = get_checkpoint_info(checkpoint_to_load_current_policy)  # This returns a dict

    # state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    # # Extract the config and override GPU settings
    # config = state["config"].copy()
    # config["input_"]=checkpoint_to_load_current_policy
    # config["env"] = env_name

    # # print(vars(config))
    # algorithm = ORPO(config=config)
    # # Load the checkpoint
    # algorithm.restore(checkpoint_to_load_current_policy)
    
    # # policy_ids = list(algorithm.workers.local_worker().policy_map.keys())
    # # print(policy_ids)
    # policy = algorithm.get_policy("safe_policy0")
    # wrapped_policy = RandomActionPolicy(policy, random_prob=0.25)

    # algorithm.workers.local_worker().policy_map["default_policy"] = wrapped_policy
    # for remote_worker in algorithm.workers.remote_workers():
    #     remote_worker.foreach_policy.remote(replace_default_policy)
    # # Save the wrapped policy to a new checkpoint
    # random_checkpoint_path = checkpoint_to_load_current_policy + "_random"
    # algorithm.save(random_checkpoint_path)
    #------------------------------------
    reference_result = ex.run(
        config_updates={
            "env_to_run": env_to_run,
            "level": level,
            "reward_fun": reward_fun,
            "exp_algo": exp_algo,
            "om_divergence_coeffs": om_divergence_coeffs_1,
            "checkpoint_to_load_policies": checkpoint_to_load_policies,
            "checkpoint_to_load_current_policy": checkpoint_to_load_current_policy,
            "seed": seed,
            "experiment_tag": experiment_tag,
            "om_divergence_type": om_divergence_type,
            "num_rollout_workers": num_rollout_workers,
            "num_gpus": num_gpus,
            "experiment_parts": experiment_parts,
            "num_training_iters": num_training_iters_1,      
        }
    )
