# CUDA Experiments

## Overview
Experiment with CUDA API and explore NVIDIA's architecture under the hood.

## Objectives
- Learn CUDA programming fundamentals
- Understand GPU memory hierarchy
- Explore parallel execution models
- Investigate NVIDIA hardware architecture

## Topics
- CUDA kernel execution
- Thread blocks and grids
- Memory management (global, shared, local)
- Performance optimization
- GPU architecture deep-dive

## Getting started
A simple first experiment is already included.

- Build and run it with: `make && ./hello_cuda`
- The program prints your GPU name, compute capability, SM count, global memory, shared memory, cache, and an initial kernel launch.
- The source is in [hello_cuda.cu](hello_cuda.cu) and the build rules are in [Makefile](Makefile).