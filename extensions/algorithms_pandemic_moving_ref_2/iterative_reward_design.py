
import os

from sacred import Experiment
from occupancy_measures.agents.orpo import ORPO, ORPOPolicy


from extensions.reward_modeling.reward_wrapper_reg_extra import RewardWrapper,RewardModel

from extensions.algorithms_pandemic_moving_ref_2.train_policy import ex
import extensions.algorithms_pandemic_moving_ref_2.train_policy
import extensions.algorithms_pandemic_moving_ref_2.unique_id_state as unique_id_state
import time

import warnings
warnings.filterwarnings("ignore")

iterative_ex = Experiment("iterative_reward_design_pandemic_sas_moving_ref_2", save_git_info=False)


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

    #all these args must be manual set per environment (annoying but we can't init gym env here) 
    if "pandemic" in env_to_run:
        reward_model = RewardModel(
            obs_dim=2*24*13, # Assuming the observation space is a 1D array of size 24*13
            action_dim=3,
            sequence_lens=193,
            discrete_actions = True,
            env_name="pandemic_sas",
            unique_id=unique_id_state.state["unique_id"],
            n_epochs=200,
            lr=0.0001
        )    
    elif "tomato" in env_to_run:
        reward_model = RewardModel(
            obs_dim=2*36, # Assuming the observation space is a 1D array of size 24*13
            action_dim=4,
            sequence_lens=100,
            discrete_actions = True,
            env_name="tomato",
            unique_id=unique_id_state.state["unique_id"],
            n_epochs=100
        )
    else:
        raise ValueError("Unsupported environment type")
    reward_model.zero_model_params()
    reward_model.save_params()
    
    for i in range(10):
        print ("(iterative_reward_design.py) UNIQUE ID (WHICH SHOULD BE THE SAME FOR ALL ITERATIONS):")
        print (unique_id_state.state["unique_id"])
        print ("======================")
        print ("checkpoint_to_load_current_policy", checkpoint_to_load_current_policy)
        print ("checkpoint_to_load_policies", checkpoint_to_load_policies)
        print ("======================")
        
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
        eval_batch_reference = reference_result.result[2]

        checkpoint_to_load_policies = ["/next/u/stephhk/orpo/"+reference_result.result[1]]
        # checkpoint_to_load_current_policy = "/next/u/stephhk/orpo/"+reference_result.result[1]

        print (checkpoint_to_load_policies)
        print (checkpoint_to_load_current_policy)

        print ("======================")
        print ("saving over opt checkpoint to:", over_opt_result.result[1])
        print ("saving reference checkpoint to:", reference_result.result[1])
        print ("======================")

        time.sleep(5)

        