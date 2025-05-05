import pandemic_simulator as ps
import torch
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.environment.pandemic_env import PandemicPolicyGymEnv
from pandemic_simulator.environment.reward import (
    RewardFunctionFactory,
    RewardFunctionType,
    SumReward,
)
from pandemic_simulator.environment.simulator_opts import PandemicSimOpts
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.utils.typing import AlgorithmConfigDict
from ray.tune.registry import register_env

from occupancy_measures.models.model_with_discriminator import (
    ModelWithDiscriminatorConfig,
)

from occupancy_measures.envs.pandemic_callbacks import PandemicCallbacks

def make_cfg(delta_hi, delta_lo, num_persons=500):
    location_configs = [
        ps.env.LocationConfig(ps.env.Home, num=150),
        ps.env.LocationConfig(
            ps.env.GroceryStore,
            num=2,
            num_assignees=5,
            state_opts=dict(visitor_capacity=30),
        ),
        ps.env.LocationConfig(
            ps.env.Office, num=2, num_assignees=150, state_opts=dict(visitor_capacity=0)
        ),
        ps.env.LocationConfig(
            ps.env.School, num=10, num_assignees=2, state_opts=dict(visitor_capacity=30)
        ),
        ps.env.LocationConfig(
            ps.env.Hospital,
            num=1,
            num_assignees=15,
            state_opts=dict(patient_capacity=5),
        ),
        ps.env.LocationConfig(
            ps.env.RetailStore,
            num=2,
            num_assignees=5,
            state_opts=dict(visitor_capacity=30),
        ),
        ps.env.LocationConfig(
            ps.env.HairSalon,
            num=2,
            num_assignees=3,
            state_opts=dict(visitor_capacity=5),
        ),
        ps.env.LocationConfig(
            ps.env.Restaurant,
            num=1,
            num_assignees=6,
            state_opts=dict(visitor_capacity=30),
        ),
        ps.env.LocationConfig(
            ps.env.Bar, num=1, num_assignees=3, state_opts=dict(visitor_capacity=30)
        ),
    ]

    return ps.env.PandemicSimConfig(
        num_persons=num_persons,
        location_configs=location_configs,
        person_routine_assignment=ps.sh.DefaultPersonRoutineAssignment(),
        delta_start_lo=delta_lo,
        delta_start_hi=delta_hi,
    )

def make_reg():
    return ps.sh.austin_regulations

#note this must match the configs in pandemic_experiments.py
def get_pandemic_env_gt_rew():
    # Environment
    horizon = 192
    num_persons = 500

    delta_start_lo = 95
    delta_start_hi = 105
    # (INFECTION_SUMMARY_ABSOLUTE, POLITICAL, LOWER_STAGE, SMOOTH_STAGE_CHANGES)
    true_weights = [10, 10, 0.1, 0.01]  # true reward weights
    proxy_weights = [10, 0, 0.1, 0.01]  # proxy reward weights

    sim_config = make_cfg(delta_start_hi, delta_start_lo, num_persons)
    regulations = make_reg()
    done_fn = ps.env.DoneFunctionFactory.default(
        ps.env.DoneFunctionType.TIME_LIMIT, horizon=horizon
    )

    # proxy_reward_fn = SumReward(
    #     reward_fns=[
    #         RewardFunctionFactory.default(
    #             RewardFunctionType.INFECTION_SUMMARY_ABSOLUTE,
    #             summary_type=InfectionSummary.CRITICAL,
    #         ),
    #         RewardFunctionFactory.default(
    #             RewardFunctionType.POLITICAL,
    #             summary_type=InfectionSummary.CRITICAL,
    #         ),
    #         RewardFunctionFactory.default(
    #             RewardFunctionType.LOWER_STAGE, num_stages=len(regulations)
    #         ),
    #         RewardFunctionFactory.default(
    #             RewardFunctionType.SMOOTH_STAGE_CHANGES,
    #             num_stages=len(regulations),
    #         ),
    #     ],
    #     weights=proxy_weights,
    # )

    true_reward_fn = SumReward(
        reward_fns=[
            RewardFunctionFactory.default(
                RewardFunctionType.INFECTION_SUMMARY_ABSOLUTE,
                summary_type=InfectionSummary.CRITICAL,
            ),
            RewardFunctionFactory.default(
                RewardFunctionType.POLITICAL,
                summary_type=InfectionSummary.CRITICAL,
            ),
            RewardFunctionFactory.default(
                RewardFunctionType.LOWER_STAGE, num_stages=len(regulations)
            ),
            RewardFunctionFactory.default(
                RewardFunctionType.SMOOTH_STAGE_CHANGES,
                num_stages=len(regulations),
            ),
        ],
        weights=true_weights,
    )

    sim_opt = PandemicSimOpts(
        spontaneous_testing_rate=0.3
    )  # testing rate set based on contact tracing experiment from original code

    env_name = "pandemic_env_multiagent_gt_rew_only"
    obs_history_size = 3
    num_days_in_obs = 8
    use_safe_policy_actions = False
    reward_fun = "true"
    safe_policy = "S0-4-0"
    env_config = {
        "sim_config": sim_config,
        "sim_opts": sim_opt,
        "pandemic_regulations": regulations,
        "done_fn": done_fn,
        "reward_fun": reward_fun,
        "true_reward_fun": true_reward_fn,
        "proxy_reward_fun": true_reward_fn,
        "constrain": True,
        "four_start": False,
        "obs_history_size": obs_history_size,
        "num_days_in_obs": num_days_in_obs,
        "use_safe_policy_actions": use_safe_policy_actions,
        "safe_policy": safe_policy,
        "horizon": horizon,
    }
    # env = PandemicPolicyGymEnv(env_config)
    return env_config, env_name