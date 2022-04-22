#!/bin/bash
# Builds a Docker image with SCM and Jupyter Lab and runs the image at apollo.fi.muni.cz

set -e -o xtrace

HOSTNAME=docker.apollo.fi.muni.cz
GPUS=9,10,11,12
PORT=8890
IMAGE_NAME=witiko/arqmath3:latest

DOCKER_BUILDKIT=1 docker build --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" --build-arg UNAME="$(id -u -n)" . -t "$IMAGE_NAME"
docker run --rm -it -u "$(id -u):$(id -g)" --hostname "$HOSTNAME" --gpus "$GPUS" -p "$PORT:$PORT" -v "$PWD"/..:/workdir:rw -w /workdir "$IMAGE_NAME" jupyter-lab --ip 0.0.0.0 --port "$PORT"
