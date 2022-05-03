#!/bin/bash
# Builds Jupyter Lab and runs it at aura.fi.muni.cz

set -e -o xtrace

GPUS=0
PORT=8889
NICENESS=10

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh

if ! conda activate arqmath3
then
    conda create --name arqmath3 python=3.9
    conda activate arqmath3
    conda install -c conda-forge jupyterlab ipywidgets nodejs=16.6.1
    conda install -c pytorch pytorch cudatoolkit=11.3
    rm -rf ~/miniconda3/pkgs/ ~/.cache/
fi

pip install .[all,notebook]

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_VISIBLE_DEVICES="$GPUS"
nice -n "$NICENESS" jupyter-lab --port "$PORT"
