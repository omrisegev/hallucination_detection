---
name: tau-runai-manager
description: Manage GPU jobs on the Tel Aviv University (TAU) RunAI cluster. Use this skill to submit, monitor, and manage interactive and batch jobs, configure projects, and handle resource allocation on the cluster.
---

# TAU RunAI Manager

This skill provides workflows for interacting with the RunAI cluster at Tel Aviv University.

## Core Workflows

### 1. Project Configuration
Always ensure the cluster and project are correctly configured before submitting jobs.
- **Local CLI Setup (One-time):**
  ```bash
  runai config set auth-url https://tau-eng.run.ai/auth
  runai login --url https://tau-eng.run.ai
  ```
- **Check projects:** `runai project list`
- **Set default project:** `runai config project blaufer`

### 2. Submitting Jobs
Jobs are either **Interactive** (development) or **Batch/Train** (execution).

#### Interactive Job (e.g., Jupyter/SSH)
```bash
runai submit --name <job-name> -i <image> --interactive --gpu 1
```
- **Usage:** Use for `jupyter`, `bash` access, or real-time debugging.
- **Note:** Remember to delete these jobs when finished to free up GPUs.

#### Batch Job (Execution)
```bash
runai submit --name <job-name> -i <image> --gpu 1 -- <command>
```
- **Example:** `runai submit --name experiment-1 -i pytorch/pytorch:latest --gpu 1 -- python train.py`

### 3. Resource Allocation
- **GPU Fractions:** Use `--gpu 0.5` for lightweight tasks.
- **CPU/Memory:** Use `--cpu 4 --memory 16G` to specify non-GPU resources.

### 4. Mounting Data
Always mount the home directory or data volumes to ensure persistence and access to code.
- **Pattern:** `-v /home/<user>:/home/<user>` (matches local/remote paths).

### 5. Job Management
- **List status:** `runai list jobs`
- **Logs:** `runai logs <job-name>`
- **Delete:** `runai delete job <job-name>`

## Reference Material
For detailed commands, GPU quotas, and storage specifics, refer to [tau_runai_docs.md](references/tau_runai_docs.md).

## Recommended Images for Hallucination Detection
- **PyTorch/Inference:** `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`
- **HuggingFace:** `huggingface/transformers-pytorch-gpu`
