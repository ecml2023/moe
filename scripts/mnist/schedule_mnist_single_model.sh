#!/bin/bash

# Generic options:

#SBATCH --account=bdrap03  # Run job under project <project>
#SBATCH --time=10:00:00  # Run for a max of 2 day

# Node resources:
# (choose between 1-4 gpus per node)

#SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type        
#SBATCH --nodes=1 --ntasks-per-node=1 # Resources from a single node
#SBATCH --gres=gpu:4      # 4 GPUS per node (plus 100% of node CPU and RAM per GPU)

# Run commands:

# Place other commands here

cd moe

export MNN_HOME=moe/:moe/src
echo "MNN_HOME  : $MNN_HOME"

export PYTHONPATH=$MNN_HOME
echo "PYTHONPATH: $PYTHONPATH"

conda init bash
conda deactivate
conda activate mnn

python src/mnist/mnist_single_model.py $*

echo "end of job"
