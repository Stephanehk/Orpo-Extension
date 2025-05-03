import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np

from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
#create a Pytorch neural network for the reward model:
class RewardModel(nn.Module):
    def __init__(self, obs_dim, action_dim,sequence_lens, discrete_actions, lr=0.001,n_epochs=100):
        super(RewardModel, self).__init__()
        self.sequence_lens = sequence_lens
        self.action_dim= action_dim
        self.discrete_actions = discrete_actions
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.initialize_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.train()

        #initialize Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.n_epochs = n_epochs

        self.seen_traj1s= []
        self.seen_traj2s= []
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def load_params(self):
        #load in state dict
        self.load_state_dict(torch.load("active_models/reward_model.pth"))
        self.train()
    
    def initialize_model(self):
        for param in self.parameters():
            nn.init.constant_(param, 0.0)
    
    def _create_sample_batch(self, rewards, actions, obs, reward_for_pref, proxy_rewards, index):
        return {
            SampleBatch.REWARDS: rewards[index],
            SampleBatch.ACTIONS: actions[index],
            SampleBatch.OBS: obs[index],
            "reward_for_pref": reward_for_pref[index],
            "proxy_rewards": proxy_rewards[index],
        }

    def _get_concatenated_obs_action(self, obs, actions):
        if self.discrete_actions:
            encoded_actions = F.one_hot(actions.long(), self.action_dim)
            net_input = torch.cat([obs, encoded_actions], dim=1)
        else:
            net_input = torch.cat([obs, actions], dim=1)
        net_input = net_input.to(torch.float32)
        return net_input

    def _calculate_discounted_sum_and_diffs(self, traj1_rews, traj2_rews,gamma=0.99):
        discounts = gamma ** torch.arange(
            len(traj1_rews), device=traj1_rews.device
        )
        rewards_diff = (discounts * (traj2_rews - traj1_rews)).sum(axis=0)
        # return torch.clip(
        #     rewards_diff, -self.config["rew_clip"], self.config["rew_clip"]
        # )
        return rewards_diff

    def _calculate_true_reward_comparisons(self, traj1, traj2):
        traj1_true_rewards = traj1["reward_for_pref"]
        traj2_true_rewards = traj2["reward_for_pref"]
        rewards_diff = self._calculate_discounted_sum_and_diffs(
            traj1_true_rewards, traj2_true_rewards
        )
        # softmax probability that traj 1 would be chosen over traj 2 based on the true reward
        probs = torch.tensor(1 / (1 + torch.exp(rewards_diff)))
        return (torch.rand(probs.size(), device=probs.device) < probs).float()

    def _calculate_boltzmann_pred_probs(self, traj1, traj2):
       
        obs = traj1["obs"].flatten(1)
        net_input = self._get_concatenated_obs_action(obs, traj1["actions"]).to(self.device)
        net_input.requires_grad = True

        traj1_preds = self.forward(net_input).flatten()#TODO: need to figure out how to add initial reward values to these estimates
        traj2_preds = self.forward(net_input).flatten()

        #add original proxy reward to the predicted reward
        traj1_preds += torch.tensor(traj1["proxy_rewards"].flatten()).to(self.device)
        traj2_preds += torch.tensor(traj2["proxy_rewards"].flatten()).to(self.device)

        preds_diff = self._calculate_discounted_sum_and_diffs(traj1_preds, traj2_preds)
        softmax_probs = 1 / (1 + preds_diff.exp())
       
        return softmax_probs

    def get_batch_sequences(self, train_batch,batch_seq_lens):
        actions = train_batch[SampleBatch.ACTIONS]
        actions_numpy = convert_to_numpy(actions)
        actions = torch.from_numpy(actions_numpy)

        rewards_sequences = add_time_dimension(
            train_batch[SampleBatch.REWARDS],
            seq_lens=batch_seq_lens,
            framework="torch",
            time_major=False,
        )
        obs_sequences = add_time_dimension(
            train_batch[SampleBatch.OBS],
            seq_lens=batch_seq_lens,
            framework="torch",
            time_major=False,
        )
        acs_sequences = add_time_dimension(
            actions,
            seq_lens=batch_seq_lens,
            framework="torch",
            time_major=False,
        )
        
        reward_sequences_for_prefs = add_time_dimension(
                [i["true_rew"] if "true_rew" in i else 0 for idx, i in enumerate(train_batch["infos"])],
                seq_lens=batch_seq_lens,
                framework="torch",
                time_major=False,
            )

        proxy_reward_seq = add_time_dimension(
                [i["original_reward"] if "original_reward" in i else 0 for idx, i in enumerate(train_batch["infos"])],
                seq_lens=batch_seq_lens,
                framework="torch",
                time_major=False,
            )
        #the first element of these sequences is blank (i.e., initial obs, but same as the obs at the first time-step), so we need to remove it
       
        rewards_sequences = rewards_sequences[:,1:]
        acs_sequences = acs_sequences[:,1:]
        obs_sequences = obs_sequences[:,1:]
        reward_sequences_for_prefs = reward_sequences_for_prefs[:,1:]
        proxy_reward_seq = proxy_reward_seq[:,1:]

        return rewards_sequences,acs_sequences,obs_sequences, reward_sequences_for_prefs, proxy_reward_seq
    
    def update_params(self,train_batch1, train_batch2):
        #seq lens:seq_lens: A 1D tensor of sequence lengths, denoting the non-padded length in timesteps of each rollout in the batch.
        # print (train_batch[SampleBatch.ACTIONS].shape)
        #I think seq_lens = [193, 193,...num traj] but gotta see the size first
        assert len(train_batch1) == len(train_batch2)#doesn't necesserily have to be the case, but cleans up implementation
        batch_seq_lens = [193 for _ in range(int(len(train_batch1)/193))]
        batch_seq_lens = torch.tensor(batch_seq_lens)
        num_sequences = len(batch_seq_lens)
       
        rewards_sequences1,acs_sequences1,obs_sequences1, reward_sequences_for_prefs1, proxy_reward_seq1 = self.get_batch_sequences(train_batch1,batch_seq_lens)
        rewards_sequences2,acs_sequences2,obs_sequences2, reward_sequences_for_prefs2, proxy_reward_seq2 = self.get_batch_sequences(train_batch2,batch_seq_lens)
        
        for _ in range(self.n_epochs):
            reward_model_loss = 0
            trajectory_pairs = [(i, j) for i in range(num_sequences) for j in range(num_sequences)]

            for indices_pair in trajectory_pairs:
                traj1 = self._create_sample_batch(
                    rewards_sequences1,
                    acs_sequences1,
                    obs_sequences1,
                    reward_sequences_for_prefs1,
                    proxy_reward_seq1,
                    indices_pair[0],
                )
                traj2 = self._create_sample_batch(
                    rewards_sequences2,
                    acs_sequences2,
                    obs_sequences2,
                    reward_sequences_for_prefs2,
                    proxy_reward_seq2,
                    indices_pair[1],
                )

                predicted_reward_probs = self._calculate_boltzmann_pred_probs(traj1, traj2).to(self.device)
                true_reward_label = self._calculate_true_reward_comparisons(traj1, traj2).to(self.device)

                reward_model_loss += torch.nn.functional.binary_cross_entropy(
                    predicted_reward_probs, true_reward_label
                )
           
            #update model params 
        
            self.optimizer.zero_grad()
            reward_model_loss.backward()
            self.optimizer.step()
            print("Reward model loss:", reward_model_loss.item())
            
        #save model state dict
        torch.save(self.state_dict(), "active_models/reward_model.pth")
        assert False

class RewardWrapper(Wrapper):
    def __init__(self, env, reward_model="custom"):
        super().__init__(env)
        self.reward_model = reward_model
        # print ("RewardWrapper initialized with reward loading_id:", loading_id)
        
        if reward_model == "custom":
            #load in reward model from disk
            self.reward_net = RewardModel(
                obs_dim=24*13, # Assuming the observation space is a 1D array of size 24*13
                action_dim=3,
                sequence_lens=192,
                discrete_actions = True, 
            )
            self.reward_net.load_params()
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
            reward = self.reward_net(obs_action).squeeze().item() + original_reward
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

