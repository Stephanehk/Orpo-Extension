#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next1
#SBATCH --time=200:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_ours_reg
#SBATCH --mem=64G

### Logging
#SBATCH --output=../run_logs/slurmjob_%j.out        # stdout log
#SBATCH --error=../run_logs/slurmjob_%j.err          # stderr log
#SBATCH --mail-type=END,FAIL,REQUEUE

# Environment setup
cd /next/u/loganmb/Orpo-Extension
source /next/u/loganmb/miniconda3/etc/profile.d/conda.sh
conda activate orpo

# Run ORPO algorithm with specified parameters
# NOTE: change the num_training_iters back to 300 for both 1 and 2
# NOTE: change back workers to 10 and cpus-per-task
python -m extensions.algorithms.iterative_reward_design \
  with env_to_run=tomato \
  level=4 \
  reward_fun=proxy \
  exp_algo=ORPO \
  om_divergence_coeffs_1=["0.8"] \
  om_divergence_coeffs_2=["0.0"] \
  'checkpoint_to_load_policies=["/next/u/stephhk/orpo/data/base_policy_checkpoints/tomato_base_policy/checkpoint_000003"]' \
  seed=0 \
  experiment_tag=state \
  'om_divergence_type=["kl"]' \
  num_rollout_workers=2 \
  num_gpus=1 \
  num_training_iters_1=2 \
  num_training_iters_2=2
