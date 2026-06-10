# RunAI at Tel Aviv University (TAU)

This document summarizes the specific configuration and workflows for using RunAI at TAU.

## Prerequisites
- **VPN:** Must be connected to TAU VPN when working remotely.
- **CLI:** `runai` CLI must be installed locally.
- **Authentication:** `runai login`

## Cluster and Project Configuration
- **Cluster:** `runai config cluster <cluster-name>`
- **Project:** `runai config project <project-name>`
  - Check your projects with `runai list projects`.

## Submitting Jobs

### Interactive Jobs (Development)
Used for Jupyter notebooks, SSH, or VS Code. These are not preemptible.
```bash
runai submit --name <job-name> -i <image> --interactive --gpu 1
```

### Train/Batch Jobs (Execution)
Used for long-running scripts. Preemptible by interactive jobs.
```bash
runai submit --name <job-name> -i <image> --gpu 1 -- <command>
```

## Resource Management
- **GPUs:** `--gpu <number>` (can be fractional, e.g., `0.5`).
- **CPUs:** `--cpu <number>`
- **Memory:** `--memory <size>` (e.g., `8G`).

## Data Storage
TAU uses NFS/Samba for persistent storage. Mount your home directory or data volumes:
```bash
runai submit ... -v /home/<user>:/container/path
```
Commonly, `/home/<user>` is where your code and data reside.

## Monitoring and Management
- **List Jobs:** `runai list jobs`
- **Describe Job:** `runai describe job <job-name>`
- **Logs:** `runai logs <job-name>`
- **Attach to Job:** `runai bash <job-name>`
- **Delete Job:** `runai delete job <job-name>`

## Recommended Docker Images
- PyTorch: `pytorch/pytorch:latest` or `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`
- TensorFlow: `tensorflow/tensorflow:latest-gpu`
