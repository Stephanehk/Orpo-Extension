import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_rewards_from_run(run_path):
    """
    Load true and proxy rewards from a training run.
    
    Args:
        run_path: Path to the training run directory
        
    Returns:
        tuple: (true_rewards, proxy_rewards, steps)
    """
    run_path = Path(run_path)
    result_file = run_path / "result.json"
    
    if not result_file.exists():
        raise FileNotFoundError(f"Could not find result.json in {run_path}")
    
    # with open(result_file, 'r') as f:     
    #     results = [json.loads(line) for line in f]
    results = []
    with open(result_file, 'r') as f:
        for i, line in enumerate(f, start=1):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line {i}: {e}")
    
    true_rewards = []
    proxy_rewards = []
    steps = []
    
    for result in results:
        if 'episode_reward_mean' in result:
            true_rewards.append(result.get('custom_metrics', {}).get('true_reward_mean', 0))
            proxy_rewards.append(result.get('custom_metrics', {}).get('proxy_reward_mean', 0))
            steps.append(result['training_iteration'])
    
    return np.array(true_rewards), np.array(proxy_rewards), np.array(steps)

def plot_rewards(run_path, save_path=None):
    """
    Plot true and proxy rewards from a training run.
    
    Args:
        run_path: Path to the training run directory
        save_path: Optional path to save the plot
    """
    true_rewards, proxy_rewards, steps = load_rewards_from_run(run_path)
    print (true_rewards)
    plt.figure(figsize=(10, 6))
    # plt.plot(steps, true_rewards, label='True Reward', color='blue')
    plt.plot(steps, proxy_rewards, label='Proxy Reward', color='red')
    print (len(proxy_rewards))
    
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.title('True vs Proxy Rewards During Training')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    # Path to the training run

    #bad regularization
    # run_path = "data/logs/pandemic/ORPO/proxy/model_128-128/state/weights_10.0_0.1_0.01/kl-0.06/seed_0/2025-04-29_09-51-02"
    #no regularization
    
    #-----PANDEMIC-----
    #contrained with regularization
    # run_path = "data/logs/pandemic/2025-05-05_13-36-03/"  #"data/logs/pandemic/2025-05-05_21-18-28/"
    #over-optimization (i.e., no regularization)
    # run_path = "data/logs/pandemic/ORPO/proxy/model_128-128/weights_10.0_0.1_0.01//seed_0/2025-05-05_21-29-00/" #"data/logs/pandemic/2025-05-06_06-41-21/"
    #-----TOMATO WORLD-----
    # contrained with regularization
    # run_path = "data/logs/tomato/rhard/ORPO/proxy/model_512-512-512-512/state/kl-0.8/seed_0/2025-05-06_17-56-37/"
    # over-optimization (i.e., no regularization)
    # run_path = "data/logs/tomato/rhard/ORPO/proxy/model_512-512-512-512/seed_0/2025-05-06_15-48-27"
    #2025-05-05_21-18-28/
    #-----TRAFFIC-----
    #constrained with regularization
    # run_path = "data/logs/traffic/2025-05-21_02-07-33/"
    # run_path = "data/logs/traffic/2025-05-20_16-18-05/"
    #over-optimization (i.e., no regularization)
    # run_path = "data/logs/traffic/singleagent_merge_bus/ORPO/proxy/model_512-512-512-512/state/kl-0.0/seed_0/2025-05-21_11-44-46/"
    run_path = "data/logs/traffic/2025-05-21_11-44-46/"
    #-----
    # run_path = "data/logs/tomato/rhard/ORPO/proxy/model_512-512-512-512//seed_0/2025-05-09_16-44-09/"
    # run_path = "data/logs/pandemic/2025-05-14_17-03-53"
    # run_path = "data/logs/traffic/2025-05-21_02-07-33/"
    #data/logs/traffic/2025-05-20_16-16-40/


    # Plot and save the rewards
    plot_rewards(run_path, save_path="rewards_plot.png") 