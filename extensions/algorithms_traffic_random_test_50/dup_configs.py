from flow.flow_cfg.get_experiment import get_exp
from flow.utils.rllib import FlowParamsEncoder

import json
from occupancy_measures.envs.traffic_callbacks import TrafficCallbacks

def dup_configs(experiment_parts=[]):
    exp_tag = "singleagent_merge_bus"  # horizon might need to be updated for bottleneck to 1040
    experiment_parts.append(exp_tag)
    scenario = get_exp(exp_tag)
    flow_params = scenario.flow_params
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4
    )
    exp_algo = "PPO"
    reward_fun = "proxy"
    assert reward_fun in ["true", "proxy"]
    callbacks = TrafficCallbacks
    use_safe_policy_actions = False
    # Rewards and weights
    proxy_rewards = ["vel", "accel", "headway"]
    proxy_weights = [1, 1, 0.1]
    true_rewards = ["commute", "accel", "headway"]
    true_weights = [1, 1, 0.1]
    true_reward_specification = [
        (r, float(w)) for r, w in zip(true_rewards, true_weights)
    ]
    proxy_reward_specification = [
        (r, float(w)) for r, w in zip(proxy_rewards, proxy_weights)
    ]
    reward_specification = {
        "true": true_reward_specification,
        "proxy": proxy_reward_specification,
    }
    reward_scale = 0.0001
    horizon = flow_params["env"].horizon
    flow_params["env"].horizon = horizon
    return flow_params,reward_specification, reward_fun,reward_scale