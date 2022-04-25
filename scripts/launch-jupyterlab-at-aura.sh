#!/bin/bash
# Builds Jupyter Lab and runs it at aura.fi.muni.cz

set -e -o xtrace

GPUS=0
PORT=8889
NICENESS=10

source ~/miniconda3/etc/profile.d/conda.sh
conda activate arqmath3
NVIDIA_VISIBLE_DEVICES="$GPUS" nice -n "$NICENESS" jupyter-lab --port "$PORT"
