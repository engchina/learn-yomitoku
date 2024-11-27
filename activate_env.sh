#!/bin/bash
# Source conda.sh to ensure conda command is available
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the desired conda environment
conda activate learn-yomitoku

# Start a new shell session in the activated environment
PS1="(learn-yomitoku) \u@\h:\w# " bash --noprofile --norc