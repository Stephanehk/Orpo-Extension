import torch
import torch.nn as nn
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType

class CustomPolicy(TorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        
        # Initialize the base network
        self.net = TorchFC(
            observation_space,
            action_space,
            action_space.n,  # num_outputs
            config,
            name="net"
        )
        
        # Initialize reward model based on config
        self.reward_model_type = config.get("reward_model", "custom")
        
        if self.reward_model_type == "custom":
            self.reward_net = nn.Sequential(
                nn.Linear(observation_space.shape[0] + action_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        else:  # pandemic reward model
            # Extract observation components
            self.critical_infection_idx = 0
            self.stage_idx = 1
            self.prev_stage_idx = 2
            
            # Reward weights
            self.infection_weight = 10.0
            self.political_weight = 0.0
            self.stage_weight = 0.1
            self.smooth_weight = 0.01

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs
    ):
        # Use the base network to compute actions
        actions, state_out, extra_fetches = super().compute_actions(
            obs_batch,
            state_batches,
            prev_action_batch,
            prev_reward_batch,
            info_batch,
            episodes,
            explore,
            timestep,
            **kwargs
        )
        
        # Compute custom rewards
        if self.reward_model_type == "custom":
            # Concatenate observations and actions
            obs_actions = torch.cat([
                torch.from_numpy(obs_batch).float(),
                torch.from_numpy(actions).float()
            ], dim=1)
            rewards = self.reward_net(obs_actions).squeeze(-1).detach().numpy()
        else:  # pandemic reward model
            obs = torch.from_numpy(obs_batch).float()
            critical_infections = obs[:, self.critical_infection_idx]
            current_stage = obs[:, self.stage_idx]
            prev_stage = obs[:, self.prev_stage_idx]
            
            infection_reward = -self.infection_weight * critical_infections
            political_reward = torch.zeros_like(infection_reward)
            stage_reward = -self.stage_weight * current_stage
            smooth_reward = -self.smooth_weight * torch.abs(current_stage - prev_stage)
            
            rewards = (infection_reward + political_reward + stage_reward + smooth_reward).detach().numpy()
        
        # Add rewards to extra_fetches
        extra_fetches["custom_rewards"] = rewards
        
        return actions, state_out, extra_fetches

    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        actions_normalized=True,
        in_training=False,
    ):
        return super().compute_log_likelihoods(
            actions,
            obs_batch,
            state_batches,
            prev_action_batch,
            prev_reward_batch,
            actions_normalized,
            in_training,
        )

    def value_function(self):
        return self.net.value_function() 