#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next6
#SBATCH --time=200:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_ours
#SBATCH --mem=64G

### Logging
#SBATCH --output=../run_logs/slurmjob_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=../run_logs/slurmjob_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=stephhk@stanford.edu # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE

./train_ours.sh
