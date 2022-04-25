#!/bin/bash
# Installs the arqmath3 conda environment at aura.fi.muni.cz

set -e -o xtrace

cleanup() {
  rm -rf ~/miniconda3/pkgs ~/.cache
}

mkdir conda-venv
ln -s "$PWD"/conda-venv ~/miniconda3/envs/arqmath3

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh

yes | conda create --name arqmath3 python=3.9
conda activate arqmath3
cleanup

yes | conda install -c pytorch pytorch cudatoolkit=11.3
cleanup

yes | conda install -c conda-forge jupyterlab ipywidgets nodejs=16.6.1
cleanup

pip install -U pip wheel setuptools
pip install .[all]
cleanup
