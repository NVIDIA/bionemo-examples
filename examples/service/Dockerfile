FROM nvcr.io/nvidia/rapidsai/rapidsai-core:22.10-cuda11.5-runtime-ubuntu20.04-py3.9

ENV TERM=xterm

## Installing dependencies and tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget tree && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/
RUN pip install biopython py3Dmol

COPY notebooks /workspace/notebooks
RUN mkdir /workspace/.local && chmod 777 /workspace/.local

