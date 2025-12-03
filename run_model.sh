#!/bin/bash

################################################################################
#
# Submit file for a batch job on Rosie.
#
# To submit your job, run 'sbatch <jobfile>'
# To view your jobs in the Slurm queue, run 'squeue -l -u <your_username>'
# To view details of a running job, run 'scontrol show jobid -d <jobid>'
# To cancel a job, run 'scancel <jobid>'
#
# See the manpages for salloc, srun, sbatch, squeue, scontrol, and scancel
# for more information or read the Slurm docs online: https://slurm.schedmd.com
#
################################################################################


# You _must_ specify the partition. Rosie's default is the 'teaching'
# partition for interactive nodes.  Another option is the 'batch' partition.
#SBATCH --partition=teaching

# The number of nodes to request
#SBATCH --nodes=1

# The number of GPUs to request
#SBATCH --gpus=1

# The number of CPUs to request per GPU
#SBATCH --cpus-per-gpu=20

# The error file to write to
#SBATCH --error='sbatcherrorfile.out'

# Kill the job if it takes longer than the specified time
# format: <days>-<hours>:<minutes>
# Increase time to 4 hours (adjust if your training needs more time)
#SBATCH --time=0-4:0


####
#
# Here's the actual job code.
# Note: You need to make sure that you execute this from the directory that
# your python file is located in OR provide an absolute path.
#
####

# Path to container
container="/data/containers/msoe-tensorflow-20.07-tf2-py3.sif"

# Working directory (submission directory)
WORKDIR="$(pwd)"

################################################################################
# Training Configuration - Modify these to customize training
################################################################################
DATA_DIR="datasets/dataset"                 # Training data directory
BATCH_SIZE="32"                             # Batch size for training
EPOCHS="2"                                  # Number of epochs
OUTPUT_DIR="/out"                           # Directory to save models/outputs
AUGMENT_DATA="false"                        # Use data augmentation (true/false)
FINE_TUNE="false"                           # Use fine tuning (true/false)

################################################################################
# Command to run inside the container with all model arguments
################################################################################
command="cd ${WORKDIR}/plants && python scripts/model.py \
    --data ${DATA_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --main_dir ${OUTPUT_DIR} \
    --augment_data ${AUGMENT_DATA} \
    --fine_tune ${FINE_TUNE}"

# Execute singularity container on the node. Bind /data and the submission directory
# so the container can access the notebook and datasets.
singularity exec --nv -B /data:/data -B ${WORKDIR}:${WORKDIR} ${container} /bin/bash -lc "${command}"
