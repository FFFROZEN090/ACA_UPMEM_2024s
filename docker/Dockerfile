# Use Ubuntu 20.04 as a base image
FROM ubuntu:20.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install basic dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake git wget curl \
    python3 python3-pip libelf-dev zlib1g-dev libtinfo-dev \
    pip install dpu

# Download and extract the UPMEM SDK
RUN wget http://sdk-releases.upmem.com/2024.1.0/ubuntu_20.04/upmem-2024.1.0-Linux-x86_64.tar.gz -O /tmp/upmem-sdk.tar.gz && \
    mkdir -p /opt/upperware && \
    tar -xzf /tmp/upmem-sdk.tar.gz -C /opt/upperware --strip-components=1 && \
    rm /tmp/upmem-sdk.tar.gz

# Set environment variables for UPMEM SDK
ENV UPMEM_HOME=/opt/upperware
ENV PATH=$UPMEM_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$UPMEM_HOME/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=$UPMEM_HOME/python:$PYTHONPATH