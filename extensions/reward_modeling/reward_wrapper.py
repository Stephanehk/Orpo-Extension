import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR,CosineAnnealingLR
import time

from gymnasium import Wrapper
from gymnasium.spaces import Box
import numpy as np
import os
import datetime


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
    def __init__(self, obs_dim, action_dim, sequence_lens, discrete_actions, env_name,lr=0.001,n_epochs=250, unique_id=None,n_prefs_per_update=None):
        super(RewardModel, self).__init__()
        self.sequence_lens = sequence_lens
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.env_name = env_name
        if env_name == "tomato":
            self.fc1 = nn.Linear(obs_dim + action_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 512)
            self.fc4 = nn.Linear(512, 512)
            self.fc5 = nn.Linear(512, 1)
        elif env_name == "pandemic":
            self.fc1 = nn.Linear(obs_dim + action_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 1)
        # self.initialize_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.train()
        self.unique_id = unique_id
        self.n_prefs_per_update=n_prefs_per_update


        print ("Create rm with unique_id:", self.unique_id)

        #initialize Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay=1e-5)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # decay_factor = (final_lr / lr) ** (1 / n_epochs)  # decay factor per step
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: decay_factor ** step)
        self.n_epochs = n_epochs
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()

        if self.unique_id is None:
            raise ValueError("unique_id must be set to save parameters")
        #check to see if file exists
        # try:
        #     with open(f"active_models/reward_model_{self.unique_id}.pth", "rb") as f:
        #         pass
        # except FileNotFoundError:
        # torch.save(self.state_dict(), f"active_models/reward_model_{self.unique_id}.pth")
        #delay for 1 second to ensure the file is created
        time.sleep(1)
    def save_params(self):
        torch.save(self.state_dict(), f"active_models/reward_model_{self.unique_id}.pth")

    def forward(self, x):
        if self.env_name  == "tomato":
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.fc5(x)
        elif self.env_name  == "pandemic":
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
        # print ("forward output:")
        # print (x)
        return x

    def load_params(self, map_to_cpu=False):
        if self.unique_id is None:
            raise ValueError("unique_id must be set to load parameters")
        #load in state dict
        if map_to_cpu:
            self.load_state_dict(torch.load(f"active_models/reward_model_{self.unique_id}.pth", map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(f"active_models/reward_model_{self.unique_id}.pth"))
        self.train()


    def get_fp(self):
        if self.unique_id is None:
            raise ValueError("unique_id must be set to load parameters")
        #load in state dict
        return f"active_models/reward_model_{self.unique_id}.pth"
    
    def zero_model_params(self):
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

    def reinitialize_model(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                layer.reset_parameters()
            elif isinstance(layer, nn.LayerNorm):
                layer.reset_parameters()
            elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d):
                layer.reset_running_stats()
                layer.reset_parameters()

    def initialize_model(self, output_layer_name=None, scale=1e-3):
        """
        Initializes the model weights such that the output predictions are very small.
        
        Args:
            model (nn.Module): The PyTorch model to initialize.
            output_layer_name (str, optional): Name of the final output layer. If None, the last linear layer is assumed.
            scale (float): The scale factor for the weights to ensure small outputs.
        """
        return
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Identify and re-initialize the output layer
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and (output_layer_name is None or name == output_layer_name):
                with torch.no_grad():
                    module.weight *= scale
                    if module.bias is not None:
                        module.bias.zero_()
                if output_layer_name is not None:
                    break  # stop if we found the named output layer
            
    def _create_sample_batch(self, rewards, actions, obs, new_obs, reward_for_pref, proxy_rewards, index):
        return {
            SampleBatch.REWARDS: rewards[index],
            SampleBatch.ACTIONS: actions[index],
            SampleBatch.OBS: obs[index],
            "new_obs": new_obs[index],
            "reward_for_pref": reward_for_pref[index],
            "proxy_rewards": proxy_rewards[index],
        }

    def _get_concatenated_obs_action(self, obs, new_obs, actions):
        if self.discrete_actions:
            encoded_actions = F.one_hot(actions.long(), self.action_dim)
            net_input = torch.cat([obs,new_obs, encoded_actions], dim=1)
        else:
            net_input = torch.cat([obs,new_obs, actions], dim=1)
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
       
        net_input1 = self._get_concatenated_obs_action(traj1["obs"].flatten(1).to(self.device),traj1["new_obs"].flatten(1).to(self.device), traj1["actions"].to(self.device))
        net_input2 = self._get_concatenated_obs_action(traj2["obs"].flatten(1).to(self.device),traj2["new_obs"].flatten(1).to(self.device), traj2["actions"].to(self.device))

        traj1_preds = self.forward(net_input1).flatten()#TODO: need to figure out how to add initial reward values to these estimates
        traj2_preds = self.forward(net_input2).flatten()

        # print ("pred reward add ons:", (traj1_preds, traj2_preds))

        #add original proxy reward to the predicted reward
        combined_traj1_preds = traj1_preds + traj1["proxy_rewards"].flatten().to(self.device)
        combined_traj2_preds = traj2_preds + traj2["proxy_rewards"].flatten().to(self.device)

        preds_diff = self._calculate_discounted_sum_and_diffs(combined_traj1_preds, combined_traj2_preds)
        softmax_probs = 1 / (1 + preds_diff.exp())
       
        return softmax_probs, traj1_preds.cpu().detach().numpy()

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
        new_obs_sequences = add_time_dimension(
            train_batch["new_obs"],
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
        #misnaming error for the tomato env
        if sum([i["true_rew"] if "true_rew" in i else 0 for idx, i in enumerate(train_batch["infos"])]) == 0:
            reward_sequences_for_prefs = add_time_dimension(
                [i["true_reward"] if "true_reward" in i else 0 for idx, i in enumerate(train_batch["infos"])],
                seq_lens=batch_seq_lens,
                framework="torch",
                time_major=False,
            )
        #original_reward
        proxy_reward_seq = add_time_dimension(
                [i["original_reward"] if "original_reward" in i else 0 for idx, i in enumerate(train_batch["infos"])],
                seq_lens=batch_seq_lens,
                framework="torch",
                time_major=False,
            )

        modified_reward_seq = add_time_dimension(
                [i["modified_reward"] if "modified_reward" in i else 0 for idx, i in enumerate(train_batch["infos"])],
                seq_lens=batch_seq_lens,
                framework="torch",
                time_major=False,
            )
        
        print ("Proxy reward seq:")
        print (proxy_reward_seq)
        print ("Reward seq for prefs:")
        print (modified_reward_seq)
        print ("==========================")
        #the first element of these sequences is blank (i.e., initial obs, but same as the obs at the first time-step), so we need to remove it
       
        rewards_sequences = rewards_sequences[:,1:]
        acs_sequences = acs_sequences[:,1:]
        obs_sequences = obs_sequences[:,1:]
        new_obs_sequences = new_obs_sequences[:,1:]
        reward_sequences_for_prefs = reward_sequences_for_prefs[:,1:]
        proxy_reward_seq = proxy_reward_seq[:,1:]
        #lets just double check we actually found the right rewards
        # assert sum(reward_sequences_for_prefs) != 0
        # assert sum(proxy_reward_seq) != 0

        return rewards_sequences,acs_sequences,obs_sequences,new_obs_sequences, reward_sequences_for_prefs, proxy_reward_seq
    
    def add2replay(self, train_batch1, train_batch2):
        assert len(train_batch1) == len(train_batch2)#doesn't necesserily have to be the case, but cleans up implementation
        batch_seq_lens = [self.sequence_lens for _ in range(int(len(train_batch1)/self.sequence_lens))]
        batch_seq_lens = torch.tensor(batch_seq_lens)
        num_sequences = len(batch_seq_lens)
        print ("num_sequences:", num_sequences)

        # print (len(train_batch1))
        # print (train_batch1["obs"])
        # print (train_batch1["actions"].shape)
        # print (train_batch1["terminateds"])
        # print ("\n")
        # print (train_batch1["truncateds"])
        
        # bools = train_batch1["terminateds"]
        # debug = [j - i - 1 for i, j in zip(
        #         [i for i, x in enumerate(bools) if x],
        #         [j for j, x in enumerate(bools) if x][1:]
        #     )]
        # print ("debug:", debug)

        rewards_sequences1,acs_sequences1,obs_sequences1,new_obs_sequences1, reward_sequences_for_prefs1, proxy_reward_seq1 = self.get_batch_sequences(train_batch1,batch_seq_lens)
        rewards_sequences2,acs_sequences2,obs_sequences2,new_obs_sequences2, reward_sequences_for_prefs2, proxy_reward_seq2 = self.get_batch_sequences(train_batch2,batch_seq_lens)
       
        trajectory_pairs = [(i, j) for i in range(num_sequences-1) for j in range(num_sequences-1)]
        # print ("# of trajectory pairs 2 add:", len(trajectory_pairs))
        # print (len(train_batch1))
        # print (len(train_batch2))
        # print ("num_sequences:", num_sequences)
        #randomly sample n_prefs_per_update pairs of trajectories
        if self.n_prefs_per_update is not None:
            selected_is = np.random.choice(list(range(len(trajectory_pairs))), size=self.n_prefs_per_update, replace=False)
            trajectory_pairs = [trajectory_pairs[i] for i in selected_is]
        for indices_pair in trajectory_pairs:
            traj1 = self._create_sample_batch(
                rewards_sequences1,
                acs_sequences1,
                obs_sequences1,
                new_obs_sequences1,
                reward_sequences_for_prefs1,
                proxy_reward_seq1,
                indices_pair[0],
            )
            traj2 = self._create_sample_batch(
                rewards_sequences2,
                acs_sequences2,
                obs_sequences2,
                new_obs_sequences2,
                reward_sequences_for_prefs2,
                proxy_reward_seq2,
                indices_pair[1],
            )
            print (proxy_reward_seq1)
            print (reward_sequences_for_prefs1)
            print ("\n")
            print (proxy_reward_seq2)
            print (reward_sequences_for_prefs2)
            print ("==============")

            true_reward_label = self._calculate_true_reward_comparisons(traj1, traj2).to(self.device)
            self.replay_buffer.push(traj1, traj2, true_reward_label)

    def update_params(self, train_batch1, train_batch2, iteration, debug_mode=False):
        # Re-initialize model weights
        # self.initialize_model()
        self.reinitialize_model()
        self.train()
        if len (self.replay_buffer) > 1000:
            self.n_epochs = 500
        if len (self.replay_buffer) > 2000:
            self.n_epochs = 1000
        self.scheduler  = CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, eta_min=1e-4)
        
        #seq lens:seq_lens: A 1D tensor of sequence lengths, denoting the non-padded length in timesteps of each rollout in the batch.
        if not debug_mode:
            self.add2replay(train_batch1, train_batch2)
        
        # Then train on the entire replay buffer
        all_losses = []
        for _ in range(self.n_epochs):
            reward_model_loss = 0
            for item in self.replay_buffer.buffer:
                if item is None:
                    continue
                traj1 = item['traj1']
                traj2 = item['traj2']
                true_label = item['true_label']
                
                predicted_reward_probs, traj_1_preds = self._calculate_boltzmann_pred_probs(traj1, traj2)
                predicted_reward_probs = predicted_reward_probs.to(self.device)
                loss = torch.nn.functional.binary_cross_entropy(
                    predicted_reward_probs, true_label
                )
                # print ("   ",loss)

                reward_model_loss += loss
            reward_model_loss /= len(self.replay_buffer.buffer)
            
            self.optimizer.zero_grad()
            reward_model_loss.backward()
            print ("reward model loss:")
            print (reward_model_loss)
            all_losses.append(reward_model_loss.item())
            self.optimizer.step()
            self.scheduler.step()
  
        #save model state dict with unique ID
        if self.unique_id is None:
            raise ValueError("unique_id must be set to save parameters")
        torch.save(self.state_dict(), f"active_models/reward_model_{self.unique_id}.pth")
        #save reward model loss
        with open(f"active_models/reward_model_loss_{self.unique_id}.txt", "a") as f:
            f.write(f"Iteration {iteration}: {reward_model_loss.item()}\n")

            
        with open(f"active_models/reward_model_all_losses_{self.unique_id}.txt", "a") as f:
            f.write(f"Iteration {iteration}: {all_losses}\n")
        with open(f"active_models/traj_1_preds_{self.unique_id}.txt", "a") as f:
            f.write(f"Iteration {iteration}: {traj_1_preds}\n")
        
        #save replay buffer
        with open(f"active_models/replay_buffer_{self.unique_id}.pkl", "wb") as f:
            torch.save(self.replay_buffer, f)

class RewardWrapper(Wrapper):
    def __init__(self, env, reward_model="custom", unique_id=None):
        super().__init__(env)
        self.reward_model = reward_model
        
        if reward_model == "custom_pandemic":
            #load in reward model from disk
            self.reward_net = RewardModel(
                obs_dim=2*24*13, # Assuming the observation space is a 1D array of size 24*13
                action_dim=3,
                sequence_lens=193,
                discrete_actions = True,
                env_name="pandemic",
                unique_id=unique_id
            )
            self.reward_net.load_params(map_to_cpu=True)
            self.reward_net.eval()
        elif reward_model == "custom_tomato":
            self.reward_net = RewardModel(
                obs_dim=2*36, # Assuming the observation space is a 1D array of size 24*13
                action_dim=4,
                sequence_lens=100,
                discrete_actions = True,
                env_name="tomato",
                unique_id=unique_id
            )
            # print ("device: ", self.reward_net.device)
            self.reward_net.load_params(map_to_cpu=True)
            self.reward_net.eval()

        self.timestamp = os.path.getmtime(self.reward_net.get_fp())

      
    def _get_concatenated_obs_action(self, obs, new_obs, actions):
        if self.reward_net.discrete_actions:
            encoded_actions = F.one_hot(actions.long(), self.reward_net.action_dim)
            net_input = torch.cat([obs,new_obs, encoded_actions], dim=1)
        else:
            net_input = torch.cat([obs,new_obs, actions], dim=1)
        net_input = net_input.to(torch.float32)
        return net_input

    def one_hot_encode(self,num, n_classes):
        res = np.zeros(n_classes)
        res[num] = 1
        return res

    def step(self, action):
        # Get the original step result
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Compute custom reward
        if "custom" in self.reward_model:
            #this is so janky <- we need to figure out a better way to do this
            # self.reward_net.load_params(map_to_cpu=True)
            if os.path.getmtime(self.reward_net.get_fp()) != self.timestamp:
                self.reward_net.load_params(map_to_cpu=True)
                self.timestamp = os.path.getmtime(self.reward_net.get_fp())
                print ("Reloading reward model parameters")
            # Convert to tensors
            #check if obs is an OrderedDict
            obs_in=obs
            las_obs_in = self.env._last_observation
            if isinstance(obs, dict):
                #26 is the number of agents in the tomatoes env
                obs_in = np.concatenate((self.one_hot_encode(obs["agent"],26), obs["tomatoes"]))
                las_obs_in = np.concatenate((self.one_hot_encode(las_obs_in["agent"],26), las_obs_in["tomatoes"]))
            else:
                #TODO: this works for pandemic env, but might not work for other envs (need to test)
                las_obs_in = self.env.obs_to_numpy(las_obs_in)
            obs_tensor = torch.from_numpy(obs_in).float().unsqueeze(0)
            last_obs_tensor = torch.from_numpy(las_obs_in).float().unsqueeze(0)
            action_tensor = torch.tensor([action]).float()
            # print (obs_tensor.shape)
            # print (last_obs_tensor.shape)
            # print (action_tensor.shape)
            # Concatenate and compute reward
            net_input = self._get_concatenated_obs_action(obs_tensor.flatten(1),last_obs_tensor.flatten(1),action_tensor).to(self.reward_net.device)
            reward = self.reward_net(net_input).squeeze().item() + original_reward
            info["modified_reward"] = reward

            print ("original_reward:", original_reward)
            print ("modified reward:", reward)
            print ("\n")
        else:
            reward = original_reward
        
        # Store original reward in info for reference
        info["original_reward"] = original_reward
        # print ("overwriting reward...")
        
        return obs, reward, terminated, truncated, info

