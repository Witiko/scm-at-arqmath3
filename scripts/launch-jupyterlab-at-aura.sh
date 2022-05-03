#!/bin/bash
# Builds Jupyter Lab and runs it at aura.fi.muni.cz

set -e -o xtrace

GPUS=0
PORT=8889
NICENESS=10

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arqmath3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_VISIBLE_DEVICES="$GPUS"
nice -n "$NICENESS" jupyter-lab --port "$PORT"
