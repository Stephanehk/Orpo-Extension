import torch
import torch.nn as nn
from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np

class RewardWrapper(Wrapper):
    def __init__(self, env, reward_model="custom"):
        super().__init__(env)
        self.reward_model = reward_model
        
        if reward_model == "custom":
            # Initialize custom reward network
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            self.reward_net = nn.Sequential(
                nn.Linear(obs_dim + action_dim, 256),
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

    def step(self, action):
        # Get the original step result
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Compute custom reward
        if self.reward_model == "custom":
            # Convert to tensors
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            action_tensor = torch.from_numpy(action).float().unsqueeze(0)
            
            # Concatenate and compute reward
            obs_action = torch.cat([obs_tensor, action_tensor], dim=1)
            reward = self.reward_net(obs_action).squeeze().item()
        else:  # pandemic reward model
            # Convert to tensor
            # obs_tensor = torch.from_numpy(obs).float()
            
            # # Extract components
            # critical_infections = obs_tensor[self.critical_infection_idx]
            # current_stage = obs_tensor[self.stage_idx]
            # prev_stage = obs_tensor[self.prev_stage_idx]
            
            # # Compute reward components
            # infection_reward = -self.infection_weight * critical_infections
            # political_reward = torch.zeros_like(infection_reward)
            # stage_reward = -self.stage_weight * current_stage
            # smooth_reward = -self.smooth_weight * torch.abs(current_stage - prev_stage)
            
            # Combine rewards
            # reward = (infection_reward + political_reward + stage_reward + smooth_reward).item()
            reward = original_reward
        
        # Store original reward in info for reference
        info["original_reward"] = original_reward
        
        return obs, reward, terminated, truncated, info 