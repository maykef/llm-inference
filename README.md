# LLM Inference Workstation — Core Setup (Blackwell)

This README explains what the `setup_llm_core.sh` script does, what it installs, and how to use, verify, customize, and remove it.

> **Scope:** This sets up *core* components only (Steps 1–3.3 in your plan): compiler toolchain, CUDA Toolkit, Miniforge (mamba), a Python env with **PyTorch 2.10 nightly (CUDA 12.8)**, Hugging Face stack, **vLLM**, and a ready-to-use workspace.  
> **Target hardware:** AMD Threadripper 7970X + **RTX Pro 6000 Blackwell (96 GB VRAM)**  
> **Target OS:** Ubuntu Desktop

---

## Contents

- [What this script installs](#what-this-script-installs)
- [Important notes & caveats](#important-notes--caveats)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [What the script actually does](#what-the-script-actually-does)
- [Verification & sanity checks](#verification--sanity-checks)
- [Daily usage](#daily-usage)
- [Workspace files created](#workspace-files-created)
- [Troubleshooting](#troubleshooting)
- [Customization tips](#customization-tips)
- [Uninstall / rollback](#uninstall--rollback)
- [FAQ](#faq)

---

## What this script installs

- **GCC 12 / G++ 12** (and sets them as default via `update-alternatives`) — required by CUDA on Ubuntu systems that default to GCC 13+
- **CUDA Toolkit 12.4** (toolkit + samples; **driver is not installed**)
- **Miniforge (mamba)** and shell init
- **Python env `llm-inference` (Python 3.11)** with:
  - **PyTorch 2.10 nightly** (pre-release) with **CUDA 12.8** wheels
  - `torchvision`, `torchaudio`
  - Hugging Face stack: `transformers`, `accelerate`, `sentencepiece`, `protobuf`
  - Quantization/tools: `bitsandbytes`
  - Monitoring: `nvitop`, `gpustat`
  - **vLLM**
- **Storage layout** at `/scratch`:
  - `/scratch/cache` (HF cache via `HF_HOME`)
  - `/scratch/models` and `/scratch/inference`
- A **workspace** at `~/llm-workspace` with sample scripts
- A **startup helper**: `~/start-llm-inference.sh`

---

## Important notes & caveats

- ⚠️ **Do not run as root**. The script will prompt for `sudo` where needed.
- ⚠️ **PyTorch nightly** is used **specifically for Blackwell (sm_120)** support. Nightly builds can change/break over time.
- ⚠️ **FlashAttention-2 is intentionally skipped** due to ABI incompatibility with nightly builds. Scripts use **SDPA** (`attn_implementation="sdpa"`) instead.
- ⚠️ **NVIDIA drivers are not installed** here. You must have a working driver + `nvidia-smi` before running this.

---

## Prerequisites

- Ubuntu Desktop with a functioning **NVIDIA driver** (`nvidia-smi` works).
- **≥ 30 GB free** in `/home` (download + env + caches).
- Internet connectivity (CUDA installer ~3 GB, Python wheels ~5 GB).
- Will prompt for `sudo` to install packages and create `/scratch`.

---

## Quick start

1. Save the script as `setup_llm_core.sh`.
2. Make it executable:
   ```bash
   chmod +x setup_llm_core.sh
   ```
3. Run it (not as root):
   ```bash
   bash setup_llm_core.sh
   ```
4. After completion, **close and reopen** your terminal (loads env vars).
5. Test:
   ```bash
   mamba activate llm-inference
   cd ~/llm-workspace
   python test_inference.py
   ```
6. Daily driver:
   ```bash
   ~/start-llm-inference.sh
   ```

---

## What the script actually does

### Phase 0 — Pre-flight
- Validates Ubuntu, detects GPU via `nvidia-smi`, checks for **Blackwell** keywords, and verifies free disk.
- Summarizes what will be installed and asks for confirmation.

### Phase 1 — GCC 12
- Installs **gcc-12/g++-12**, configures `update-alternatives` to set them as default.
- Retains other gcc versions as lower-priority alternatives where present.

### Phase 2 — CUDA Toolkit 12.4
- Downloads the NVIDIA **local installer** and installs **toolkit + samples** (no driver).
- Appends CUDA env to `~/.bashrc`:
  ```
  export CUDA_HOME=/usr/local/cuda-12.4
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

### Phase 3 — Miniforge (mamba)
- Installs Miniforge to `~/miniforge3` and initializes **conda** and **mamba** for bash.

### Phase 4 — Storage
- Creates `/scratch/{inference,cache,models}` and sets you as owner.
- Sets `HF_HOME=/scratch/cache` in your `~/.bashrc`.

### Phase 5 — Python env
- Creates **`llm-inference`** (Python 3.11).
- Activates it.

### Phase 6 — PyTorch nightly (CUDA 12.8)
- Installs **torch/vision/audio** from nightly CUDA **12.8** index.
- Verifies CUDA availability and **sm_12x** compute capability at runtime.

### Phase 7 — Hugging Face stack
- Installs `transformers`, `accelerate`, `sentencepiece`, `protobuf`, `bitsandbytes`, `nvitop`, `gpustat`.

### Phase 8 — vLLM
- Installs **vLLM** (pip).
- Quick import/version check.

### Phase 9 — Workspace
Creates `~/llm-workspace` with:

- `test_inference.py` — runs **Mistral-7B-Instruct** with **SDPA**, BF16.
- `daily_inference.py` — simple CLI with optional **4-bit** (`--4bit`) via bitsandbytes.
- `vllm_inference.py` — minimal **vLLM** example (BF16).
- `~/start-llm-inference.sh` — interactive launcher/monitor.

### Phase 10 — Final verification
- Prints versions of GCC, CUDA, mamba, PyTorch, CUDA in PyTorch, transformers, vLLM, and confirms **Blackwell** if `sm_120`.

Also writes a summary log to `~/llm-core-install.log`.

---

## Verification & sanity checks

After a new terminal:

```bash
mamba activate llm-inference

# CUDA toolchain
nvcc --version

# PyTorch CUDA
python - << 'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
PY

# HF cache location
echo $HF_HOME   # should be /scratch/cache
```

Run the included tests:

```bash
cd ~/llm-workspace
python test_inference.py
python vllm_inference.py
```

---

## Daily usage

### Option A — Startup helper (recommended)
```bash
~/start-llm-inference.sh
```
- Activates env, shows GPU and PyTorch info, offers:
  1) `daily_inference.py` interactive shell  
  2) Run `vllm_inference.py`  
  3) Monitor GPU via `nvitop`

### Option B — Manual
```bash
mamba activate llm-inference
python ~/llm-workspace/daily_inference.py --prompt "What is quantum computing?"
# or specify a model
python ~/llm-workspace/daily_inference.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --prompt "Summarize..."
# 4-bit quantization
python ~/llm-workspace/daily_inference.py --4bit --model meta-llama/Llama-2-13b-chat-hf --prompt "..."
```

> **Tip:** First model download can take time and disk space; models cache under `/scratch/cache`.

---

## Workspace files created

- `~/llm-workspace/test_inference.py` — Mistral-7B-Instruct, **SDPA**, BF16, simple prompt.
- `~/llm-workspace/daily_inference.py` — Small utility class + CLI. Supports `--4bit`.
- `~/llm-workspace/vllm_inference.py` — Minimal vLLM sample (BF16).
- `~/start-llm-inference.sh` — Helper to activate, inspect, and run workflows.

---

## Troubleshooting

**“Please do not run this script as root or with sudo”**  
Run it as a normal user. It will request `sudo` only when needed.

**`nvidia-smi: command not found` or reports no GPU**  
Install the correct NVIDIA driver first. This script does *not* install drivers.

**CUDA installed, but `torch.cuda.is_available()` is False**  
- Close/reopen terminal (load env vars).
- Verify you’re in the **`llm-inference`** env.
- Ensure driver version supports your GPU and CUDA version.
- Conflicting CUDA paths? Remove older `CUDA_HOME`/`LD_LIBRARY_PATH` exports in `~/.bashrc`.

**`bitsandbytes` fails to load**  
- Nightly + new CUDA can lag for prebuilt binaries. The script installs it via pip; if loading fails, try CPU/offload or skip `--4bit` until matching wheels land.

**vLLM import/launch errors**  
- vLLM & nightly PyTorch move quickly. If import fails, try pinning a recent known-good vLLM version or reinstalling it within the same env after PyTorch is installed.

**GCC default changed unexpectedly**  
Re-run `update-alternatives` to set gcc/g++ back to 12:
```bash
sudo update-alternatives --set gcc /usr/bin/gcc-12
sudo update-alternatives --set g++ /usr/bin/g++-12
```

---

## Customization tips

- **Change Python version:** edit the `mamba create -n llm-inference python=3.11` line.
- **Stick to stable PyTorch:** replace the nightly install command with the stable CUDA wheel index and adjust CUDA versions accordingly (not Blackwell-ready until official release includes sm_120).
- **Different CUDA Toolkit:** update `CUDA_URL`, `CUDA_HOME`, and the path exports.
- **Alternative cache location:** change `/scratch/cache` and update `HF_HOME` accordingly.
- **Different models:** the sample scripts use Mistral-7B/Llama-2 by default. Swap `model=` values as needed (respect license and tokens where required).

---

## Uninstall / rollback

> **Proceed carefully** — you may have other software depending on these components.

- **Remove the conda env & Miniforge**
  ```bash
  mamba deactivate || true
  mamba env remove -n llm-inference -y
  rm -rf ~/miniforge3
  # Remove conda init blocks from ~/.bashrc
  sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ~/.bashrc
  ```
- **Remove CUDA 12.4** (if you installed only for this env)
  ```bash
  sudo rm -rf /usr/local/cuda-12.4
  # Remove the CUDA exports block from ~/.bashrc
  sed -i '/# CUDA 12.4 Configuration/,+3d' ~/.bashrc
  ```
- **Restore GCC alternatives** (example to set gcc-13 back, if present)
  ```bash
  sudo update-alternatives --set gcc /usr/bin/gcc-13
  sudo update-alternatives --set g++ /usr/bin/g++-13
  ```
- **Remove workspace & logs**
  ```bash
  rm -rf ~/llm-workspace ~/start-llm-inference.sh ~/llm-core-install.log
  ```
- **Remove /scratch data** *(irreversible!)*  
  ```bash
  sudo rm -rf /scratch/inference /scratch/models /scratch/cache
  ```

---

## FAQ

**Why GCC 12?**  
CUDA 12.x toolchains require GCC/G++ versions within supported ranges. Ubuntu may default to GCC 13+, so we install and set **GCC 12** as default.

**Why PyTorch nightly with CUDA 12.8?**  
To enable **Blackwell (sm_120)** support prior to official stable wheels including it.

**Why skip FlashAttention-2?**  
Nightly ABI changes often break FA-2 wheels. The scripts use **SDPA**, which is compatible and performant.

**Do I need root?**  
Only for system-level steps (Apt, `/scratch` creation, `update-alternatives`). Run the script as a regular user; it will prompt for `sudo` when needed.

---

## License

This README describes your setup script. Use and modify freely within your project’s license.
