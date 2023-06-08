#!/usr/bin/env bash

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

IMAGE_NAME="bionemo-notebooks"
CONT_NAME="bionemo-notebooks"

DOCKER_RUN_CMD="docker run \
    --name ${CONT_NAME} \
    --network host \
    --gpus all \
    --privileged \
    --rm -it \
    -p 8888:8888 \
    -p 8787:8787 \
    -p 8786:8786 \
    -u $(id -u):$(id -u) \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v /etc/shadow:/etc/shadow:ro \
    -e HOME=/workspace
    -w /workspace "

DOCKER_BUILD_CMD="docker build \
    --network host \
    -t ${IMAGE_NAME}:latest"


usage() {
    echo "Please open the source code to find."
}


build() {
    while [[ $# -gt 0 ]]; do
        case $1 in
    	    -n|--no-cache)
                DOCKER_BUILD_CMD="${DOCKER_BUILD_CMD} --no-cache"
                shift
                ;;
            *)
                echo "Unknown option $1"
                exit 1
                ;;
        esac
    done
    echo -e "Building ${DOCKER_FILE}..."
    $DOCKER_BUILD_CMD .
}


run() {
    set -x
    $DOCKER_RUN_CMD \
        ${IMAGE_NAME}
}

run-dev() {
    set -x
    $DOCKER_RUN_CMD \
        -v $(pwd -P)/notebooks:/workspace/notebooks-dev \
        ${IMAGE_NAME}
}

case $1 in
    build)
        "$@"
        ;;
    run-dev)
        "$@"
        ;;
    *)
        run
        exit 0
        ;;
esac
