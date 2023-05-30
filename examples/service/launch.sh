
#!/usr/bin/env bash

IMAGE_NAME="nvcr.io/nv-drug-discovery-dev/bionemo-example-public"
CONT_NAME="bionemo-example-public"

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
    -t ${IMAGE_NAME}:0.3.2 \
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
