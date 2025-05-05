import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np

from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def push(self, traj1, traj2, true_label):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = {
            'traj1': traj1,
            'traj2': traj2,
            'true_label': true_label
        }
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[idx] for idx in indices]

    def __len__(self):
        return len(self.buffer)

#create a Pytorch neural network for the reward model:
class RewardModel(nn.Module):
    def __init__(self, obs_dim, action_dim, sequence_lens, discrete_actions, lr=0.001, n_epochs=100, unique_id=None):
        super(RewardModel, self).__init__()
        self.sequence_lens = sequence_lens
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.initialize_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.train()
        self.unique_id = unique_id

        print ("Create rm with unique_id:", self.unique_id)

        #initialize Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.n_epochs = n_epochs
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()

        if self.unique_id is None:
            raise ValueError("unique_id must be set to save parameters")
        torch.save(self.state_dict(), f"active_models/reward_model_{self.unique_id}.pth")

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # print ("forward output:")
        # print (x)
        return x

    def load_params(self):
        if self.unique_id is None:
            raise ValueError("unique_id must be set to load parameters")
        #load in state dict
        self.load_state_dict(torch.load(f"active_models/reward_model_{self.unique_id}.pth"))
        self.train()
    
    def initialize_model(self):
        # Initialize all layers with Xavier/Glorot initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Zero out the last layer
        last_layer = list(self.modules())[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.zeros_(last_layer.weight)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)
            
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
        probs = 1 / (1 + torch.exp(rewards_diff))
        return (torch.rand(probs.size(), device=probs.device) < probs).float()

    def _calculate_boltzmann_pred_probs(self, traj1, traj2):
       
        net_input1 = self._get_concatenated_obs_action(traj1["obs"].flatten(1).to(self.device), traj1["actions"].to(self.device))
        net_input2 = self._get_concatenated_obs_action(traj2["obs"].flatten(1).to(self.device), traj2["actions"].to(self.device))

        traj1_preds = self.forward(net_input1).flatten()#TODO: need to figure out how to add initial reward values to these estimates
        traj2_preds = self.forward(net_input2).flatten()

        #add original proxy reward to the predicted reward
        traj1_preds += traj1["proxy_rewards"].flatten().to(self.device)
        traj2_preds += traj2["proxy_rewards"].flatten().to(self.device)

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
    
    def update_params(self, train_batch1, train_batch2, iteration):
        # Re-initialize model weights
        self.initialize_model()
        self.train()
        
        #seq lens:seq_lens: A 1D tensor of sequence lengths, denoting the non-padded length in timesteps of each rollout in the batch.
      
        assert len(train_batch1) == len(train_batch2)#doesn't necesserily have to be the case, but cleans up implementation
        batch_seq_lens = [self.sequence_lens for _ in range(int(len(train_batch1)/self.sequence_lens))]
        batch_seq_lens = torch.tensor(batch_seq_lens)
        num_sequences = len(batch_seq_lens)
        print ("num_sequences:", num_sequences)
       
        rewards_sequences1,acs_sequences1,obs_sequences1, reward_sequences_for_prefs1, proxy_reward_seq1 = self.get_batch_sequences(train_batch1,batch_seq_lens)
        rewards_sequences2,acs_sequences2,obs_sequences2, reward_sequences_for_prefs2, proxy_reward_seq2 = self.get_batch_sequences(train_batch2,batch_seq_lens)
       
        trajectory_pairs = [(i, j) for i in range(num_sequences-1) for j in range(num_sequences-1)]
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

            true_reward_label = self._calculate_true_reward_comparisons(traj1, traj2).to(self.device)
            self.replay_buffer.push(traj1, traj2, true_reward_label)
        
        # Then train on the entire replay buffer
        for _ in range(self.n_epochs):
            reward_model_loss = 0
            for item in self.replay_buffer.buffer:
                if item is None:
                    continue
                traj1 = item['traj1']
                traj2 = item['traj2']
                true_label = item['true_label']
                
                predicted_reward_probs = self._calculate_boltzmann_pred_probs(traj1, traj2).to(self.device)
                reward_model_loss += torch.nn.functional.binary_cross_entropy(
                    predicted_reward_probs, true_label
                )
            
            self.optimizer.zero_grad()
            reward_model_loss.backward()
            print (reward_model_loss)
            self.optimizer.step()
  
        #save model state dict with unique ID
        if self.unique_id is None:
            raise ValueError("unique_id must be set to save parameters")
        torch.save(self.state_dict(), f"active_models/reward_model_{self.unique_id}.pth")
        #save reward model loss
        with open(f"active_models/reward_model_loss_{self.unique_id}.txt", "a") as f:
            f.write(f"Iteration {iteration}: {reward_model_loss.item()}\n")
        #save replay buffer
        with open(f"active_models/replay_buffer_{self.unique_id}.pkl", "wb") as f:
            torch.save(self.replay_buffer, f)

class RewardWrapper(Wrapper):
    def __init__(self, env, reward_model="custom", unique_id=None):
        super().__init__(env)
        self.reward_model = reward_model
        
        if reward_model == "custom":
            #load in reward model from disk
            self.reward_net = RewardModel(
                obs_dim=24*13, # Assuming the observation space is a 1D array of size 24*13
                action_dim=3,
                sequence_lens=193,
                discrete_actions = True,
                unique_id=unique_id
            )
            self.reward_net.load_params()
            self.reward_net.eval()
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
    
    def _get_concatenated_obs_action(self, obs, actions):
        if self.reward_net.discrete_actions:
            encoded_actions = F.one_hot(actions.long(), self.reward_net.action_dim)
            net_input = torch.cat([obs, encoded_actions], dim=1)
        else:
            net_input = torch.cat([obs, actions], dim=1)
        net_input = net_input.to(torch.float32)
        return net_input

    def step(self, action):
        # Get the original step result
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Compute custom reward
        if self.reward_model == "custom":
            # Convert to tensors
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            action_tensor = torch.tensor([action]).float()
            # Concatenate and compute reward
            net_input = self._get_concatenated_obs_action(obs_tensor.flatten(1), action_tensor).to(self.reward_net.device)
            reward = self.reward_net(net_input).squeeze().item() + original_reward
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

