# Tau-Bench Experiment Execution Workflow

## Jupyter Session Request

Account

```
class_class59827694spring_2026
```

Partition

```
gaudi
```

QOS

```
class_gaudi
```

CPU Core Allocation

```
20
```

Memory Allocation (GiB)

```
40
```

GPU Resources

```
gpu:hl225:1
```

Jupyter Wall Time (your choice)

```
0-#
```

Jupyter lab version

```
latest
```

## Project Folder Structure Overview (In Sol not GitHub)

This repository is organized so that **model servers**, **Slurm job scripts**, and **experiment drivers** are clearly separated. Understanding this layout makes it easier to debug failures and onboard new contributors.

```
~/agent-project/
├── assistant-server.sh        # Launches the assistant vLLM server (OpenAI-compatible)
├── user-server.sh             # Launches the user vLLM server (OpenAI-compatible)
├── user-vllm-job.sh           # Slurm job wrapper for starting the user vLLM server
├── tau-experiment.sh          # Main Slurm job script that runs Tau-Bench
│
├── logs/                      # Slurm stdout logs for servers and experiments
├── errors/                    # Slurm stderr logs for servers and experiments
├── results/                   # Home for the json files
└── tau-bench/                 # Tau-Bench codebase

```

### Key Ideas

- **Server scripts** (`assistant-server.sh`, `user-server.sh`) are responsible only for starting vLLM instances.
- **Slurm job scripts** (`user-vllm-job.sh`, `tau-experiment.sh`) handle resource allocation and scheduling.
- **Tau-Bench itself** lives in its own subdirectory and is treated as an executable dependency.
- **Logs** are the first place to look when something fails.

This separation keeps experiments reproducible and prevents model server issues from contaminating benchmark logic.

---


## 1. Load Environment & Activate Conda/Mamba Env

Load the cluster-provided mamba module and activate the environment that contains PyTorch + vLLM support:

```bash
module load mamba/latest
source activate gaudi-pytorch-vllm
```

> ⚠️ All subsequent commands assume this environment is active.

---

## 2. Python Dependency Fixes

Some Tau-Bench and LiteLLM dependencies need to be explicitly installed at the user level:

```bash
pip install --user "litellm==1.41.0"
pip install --user "typing_extensions>=4.10.0"
pip install --user "tokenizers>=0.21,<0.22"
```
> ⚠️ You only need to run this once.

---

## 3. Start the User vLLM Server

Submit the Slurm job that launches the **user model** vLLM server:

```bash
sbatch user-vllm-job.sh
```

After submission, go into the **logs** directory and
scroll to the top of the file with this filename structure: "user_vllm_jobid.out". You should see the following output:

```
====================================
Starting USER vLLM server on node: gaudi001
User model: Qwen/Qwen3-32B
User server base: http://gaudi001:8007/v1
====================================
...
```

Verify that the user model is live at the gaudi endpoint:

```bash
curl -s http://gaudi001:8007/v1/models | jq
```

> ⚠️ IMPORTANT: If you copy and paste this command make sure the port number matches the **User server base** from the **user_vllm_jobid.out** file

You should see a JSON response listing the specs of the user model.

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-32B",
      "object": "model",
      "created": 1770442353,
      "owned_by": "vllm",
      "root": "Qwen/Qwen3-32B",
      "parent": null,
      "max_model_len": 40960,
      "permission": [
        {
          "id": "modelperm-845d9dc64768b3a7",
          "object": "model_permission",
          "created": 1770442353,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

## 4. Export User Model API Base

Set the environment variable that Tau-Bench uses to locate the **user model** endpoint:

```bash
export USER_MODEL_API_BASE="http://gaudi001:8007/v1"
```
> ⚠️ IMPORTANT: Again, if you copy and paste this command make sure the port number matches the **User server base** from the **user_vllm_jobid.out** file

---

## 5. Launch Tau-Bench Experiments

### Option A: Slurm Job Array (Recommended for Sweeps)

Run using a job array (This command does not assume that you start at 0 and end at 119. Choose any sub array below):

```bash
sbatch --array=0-119 tau-experiment.sh
```

```bash

# Copy any line below to run that experiment
# Each command ends with its task id for job array submissions

sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 1    # 0 ben (Done✅)
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 2    # 1 ben (Done✅)
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 3    # 2 ben (In progress)
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 4    # 3 ben (In progress)
sbatch tau-experiment.sh airline act Qwen/Qwen3-4B-Instruct-2507 5    # 4 ben (In progress)
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 1    # 5 
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 2    # 6 
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 3    # 7 
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 4    # 8 
sbatch tau-experiment.sh airline act Qwen/Qwen3-8B-Instruct-2507 5    # 9 
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 1   # 10
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 2   # 11
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 3   # 12
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 4   # 13
sbatch tau-experiment.sh airline act Qwen/Qwen3-14B-Instruct-2507 5   # 14
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 1   # 15
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 2   # 16
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 3   # 17
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 4   # 18
sbatch tau-experiment.sh airline act Qwen/Qwen3-32B-Instruct-2507 5   # 19
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 1  # 20 ben (Done✅)
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 2  # 21 ben (Done✅)
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 3  # 22 ben (In progress)
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 4  # 23 ben (In progress)
sbatch tau-experiment.sh airline react Qwen/Qwen3-4B-Instruct-2507 5  # 24 ben (In progress)
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 1  # 25
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 2  # 26
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 3  # 27
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 4  # 28
sbatch tau-experiment.sh airline react Qwen/Qwen3-8B-Instruct-2507 5  # 29
sbatch tau-experiment.sh airline react Qwen/Qwen3-14B-Instruct-2507 1 # 30
sbatch tau-experiment.sh airline react Qwen/Qwen3-14B-Instruct-2507 2 # 31
sbatch tau-experiment.sh airline react Qwen/Qwen3-14B-Instruct-2507 3 # 32
sbatch tau-experiment.sh airline react Qwen/Qwen3-14B-Instruct-2507 4 # 33
sbatch tau-experiment.sh airline react Qwen/Qwen3-14B-Instruct-2507 5 # 34
sbatch tau-experiment.sh airline react Qwen/Qwen3-32B-Instruct-2507 1 # 35
sbatch tau-experiment.sh airline react Qwen/Qwen3-32B-Instruct-2507 2 # 36
sbatch tau-experiment.sh airline react Qwen/Qwen3-32B-Instruct-2507 3 # 37
sbatch tau-experiment.sh airline react Qwen/Qwen3-32B-Instruct-2507 4 # 38
sbatch tau-experiment.sh airline react Qwen/Qwen3-32B-Instruct-2507 5 # 39
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 1     # 40 ben (In progress)
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 2     # 41 ben (In progress)
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 3     # 42 ben (In progress)
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 4     # 43 ben (In progress)
sbatch tau-experiment.sh airline fc Qwen/Qwen3-4B-Instruct-2507 5     # 44 ben (In progress)
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 1     # 45 
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 2     # 46 
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 3     # 47 
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 4     # 48 
sbatch tau-experiment.sh airline fc Qwen/Qwen3-8B-Instruct-2507 5     # 49 
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 1    # 50
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 2    # 51
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 3    # 52
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 4    # 53
sbatch tau-experiment.sh airline fc Qwen/Qwen3-14B-Instruct-2507 5    # 54
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 1    # 55
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 2    # 56
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 3    # 57
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 4    # 58
sbatch tau-experiment.sh airline fc Qwen/Qwen3-32B-Instruct-2507 5    # 59
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 1     # 60 ben (Done✅)
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 2     # 61 ben (In progress)
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 3     # 62 ben (In progress)
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 4     # 63 ben (In progress)
sbatch tau-experiment.sh retail act Qwen/Qwen3-4B-Instruct-2507 5     # 64 ben (In progress)
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 1     # 65
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 2     # 66
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 3     # 67
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 4     # 68
sbatch tau-experiment.sh retail act Qwen/Qwen3-8B-Instruct-2507 5     # 69
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 1    # 70
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 2    # 71
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 3    # 72
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 4    # 73
sbatch tau-experiment.sh retail act Qwen/Qwen3-14B-Instruct-2507 5    # 74
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 1    # 75
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 2    # 76
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 3    # 77
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 4    # 78
sbatch tau-experiment.sh retail act Qwen/Qwen3-32B-Instruct-2507 5    # 79
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 1   # 80 ben (In progress)
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 2   # 81 ben (In progress)
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 3   # 82 ben (In progress)
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 4   # 83 ben (In progress)
sbatch tau-experiment.sh retail react Qwen/Qwen3-4B-Instruct-2507 5   # 84 ben (In progress)
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 1   # 85
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 2   # 86
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 3   # 87
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 4   # 88
sbatch tau-experiment.sh retail react Qwen/Qwen3-8B-Instruct-2507 5   # 89
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 1  # 90
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 2  # 91
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 3  # 92
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 4  # 93
sbatch tau-experiment.sh retail react Qwen/Qwen3-14B-Instruct-2507 5  # 94
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 1  # 95
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 2  # 96
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 3  # 97
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 4  # 98
sbatch tau-experiment.sh retail react Qwen/Qwen3-32B-Instruct-2507 5  # 99
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 1      # 100 ben (In progress)
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 2      # 101 ben (In progress)
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 3      # 102 ben (In progress)
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 4      # 103 ben (In progress)
sbatch tau-experiment.sh retail fc Qwen/Qwen3-4B-Instruct-2507 5      # 104 ben (In progress)
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 1      # 105
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 2      # 106
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 3      # 107
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 4      # 108
sbatch tau-experiment.sh retail fc Qwen/Qwen3-8B-Instruct-2507 5      # 109
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 1     # 110
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 2     # 111
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 3     # 112
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 4     # 113
sbatch tau-experiment.sh retail fc Qwen/Qwen3-14B-Instruct-2507 5     # 114
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 1     # 115
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 2     # 116
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 3     # 117
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 4     # 118
sbatch tau-experiment.sh retail fc Qwen/Qwen3-32B-Instruct-2507 5     # 119
```

---

**Arguments (in order):**

1. Environment (e.g., `retail`)
2. Agent strategy (e.g., `react`)
3. Assistant model ID
4. `num_trials`

**Optional task range** (`--start-index`, `--end-index`): pass these before other arguments to limit which tasks run. Defaults: `0` and `-1` (all tasks).

```bash
# Run tasks 0–49 for retail with act agent
sbatch tau-experiment.sh --start-index 0 --end-index 50 retail act "Qwen/Qwen3-4B-Instruct-2507"

# Array mode, limit to first 10 tasks per job
sbatch --array=0-119 tau-experiment.sh --start-index 0 --end-index 10

# Omit both (defaults: start 0, end -1)
sbatch tau-experiment.sh retail act "Qwen/Qwen3-4B-Instruct-2507"
```

---

### Resuming after job timeout

The function below checks for an existing json file for the submitted job and gets starts the experiment for the next task in line.

**Example directory structure:** home/'asurite'/agent-project/results/retail/react/4B/num_trials-1.json

```bash
# Usage: set START_INDEX for resume; call before the python run.
set_start_index_from_checkpoint() {
  local ckpt="${RUN_DIR}/num_trials-${NUM_TRIALS_VAL}.json"
  START_INDEX=0
  if [[ -f "$ckpt" ]]; then
    local max_id
    max_id=$(jq -r '[.[].task_id] | max // empty' "$ckpt" 2>/dev/null)
    if [[ -n "$max_id" && "$max_id" != "null" ]]; then
      START_INDEX=$((max_id + 1))
      echo "Resuming: last task_id=$max_id, --start-index=$START_INDEX"
    fi
  fi
}

set_start_index_from_checkpoint
python run.py \
  ...
  --start-index "$START_INDEX" \
  --end-index -1 \ # Last task
  ...
```

### Other files edited to make this work: tau-bench/tau-bench/run.py

```python
def run(config: RunConfig) -> List[EnvRunResult]:

    ...
    ckpt_path = (
        f"{config.log_dir}/num_trials-{config.num_trials}.json"
    )
    ...
```

---

### Summary Flow

**High-level order of operations:**

1. Set permissions on scripts
2. Load mamba + activate environment
3. Install Python dependencies
4. Launch user vLLM server
5. Verify API endpoint
6. Export `USER_MODEL_API_BASE`
7. Submit Tau-Bench jobs

---

If anything fails, the most common issues are:

- Environment not activated
- vLLM server not fully started before Tau-Bench runs
- Missing or mismatched dependency versions
- Incorrect hostname or port in `USER_MODEL_API_BASE`

This document should be kept in sync with any future changes to scripts or dependency versions.
