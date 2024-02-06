# Use an official NVIDIA CUDA 12.1 base image
ARG BASEIMAGE=nvcr.io/nvidia/nvhpc:23.5-devel-cuda12.1-ubuntu22.04
FROM ${BASEIMAGE} as base
ARG PLATFORM

ENV BUILD_PLATFORM=${PLATFORM}

# Install dependencies (example: compilers, and CUDA samples)
# Note: The list of packages you install might differ based on your application's requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY install/entry.sh /install/entry.sh
RUN bash /install/entry.sh

# CUDA Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
ENV C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY install/conda_env.sh /install/conda_env.sh
RUN bash /install/conda_env.sh
