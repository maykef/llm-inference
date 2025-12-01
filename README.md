# LLM Inference Workstation — ULTIMATE 2025 Blackwell Edition (v2.0)

This is the complete, production-grade, self-contained local AI coding superweapon  
for the NVIDIA RTX Pro 6000 Blackwell (96 GB VRAM).

One script → one command → you own one of the most powerful private 70B+ coding assistants on Earth, 100% offline, forever reproducible.

Target hardware: AMD Threadripper 7970X + RTX Pro 6000 Blackwell (96 GB VRAM)
Target OS:       Ubuntu Desktop 24.04 LTS (or newer)
Script name:     llm-inference_2.sh

--------------------------------------------------------------------------------
## What you get (December 2025 state-of-the-art)

Component                                | Version / Status
-----------------------------------------|---------------------------------------------
GCC / G++                                | 12 (forced default)
CUDA Toolkit                             | 12.8 (exact match with PyTorch wheels)
PyTorch                                  | 2.7.0 STABLE (cu128) – official Blackwell sm_120 support
Attention                                | Native SDPA (attn_implementation="sdpa") – faster & more stable than Flash-Attn
vLLM                                     | Latest, 96 GB-optimized (85% mem util, eager mode)
Quantization                             | bitsandbytes ≥0.43.3 + AutoAWQ (full 4-bit NF4, AWQ, bnb-4bit support)
Storage                                  | /scratch/cache → all Hugging Face models/datasets
Your final production scripts (baked in forever):
├─ daily_inference_3.py                  | Full-repo-context chat, smart bnb-4bit/AWQ/Unsloth handling
├─ local_interpreter.py                  | Model writes & executes Python on host (with user approval)
└─ docker_interpreter.py                 | Sandboxed execution via Docker (GPU passthrough + auto-package install)

--------------------------------------------------------------------------------
## Quick start

wget https://raw.githubusercontent.com/youruser/yourrepo/main/llm-inference_2.sh
chmod +x llm-inference_2.sh
bash llm-inference_2.sh          # ← run as normal user, never sudo the whole script

→ 20–40 min depending on network
→ When finished: close and reopen your terminal

Daily driver (one command forever):

~/start-llm-inference.sh

Menu you will see:
1. Test inference (Mistral-7B)
2. Daily chat + full repo context           ← RECOMMENDED daily driver
3. Local code interpreter (executes on host)
4. Docker code interpreter (sandboxed + auto-installs packages)
5. vLLM inference (production)
6. GPU monitor (nvitop)

Just pick 2, 3 or 4 and you’re superhuman.

--------------------------------------------------------------------------------
## What the script actually does (10 phases)

0  Pre-flight: Ubuntu + Blackwell detection + 30 GB free space check
1  Install & set GCC 12 as default
2  Install CUDA Toolkit 12.8 (matches PyTorch cu128 wheels exactly)
3  Install Miniforge + mamba with persistent shell init
4  Create /scratch + redirect all HF caches there
5  Create Python env llm-inference (Python 3.12)
6  Install PyTorch 2.7.0 stable + verify sm_120
7  Hugging Face stack + bitsandbytes + AutoAWQ
8  vLLM + aggressive cache cleanup
9  Deploy your final three production scripts + updated launcher
10 Full verification + beautiful success summary + install log

--------------------------------------------------------------------------------
## Verification (after fresh terminal)

mamba activate llm-inference
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# → should show 2.7.0, RTX Pro 6000 Blackwell, (12, 0)

Run the launcher:
~/start-llm-inference.sh   # → all green checks = perfect

--------------------------------------------------------------------------------
## Daily usage examples

# Chat with your entire current project folder
python ~/llm-workspace/daily_inference_3.py --interactive --repo_path . --4bit

# Model writes & runs code locally (fastest loop)
python ~/llm-workspace/local_interpreter.py --interactive --repo_path . --4bit

# Secure sandboxed execution (Docker)
python ~/llm-workspace/docker_interpreter.py --interactive --repo_path . --docker --extra_packages numpy pandas matplotlib --4bit

--------------------------------------------------------------------------------
## Uninstall / rollback (safe)

mamba deactivate 2>/dev/null || true
mamba env remove -n llm-inference -y
rm -rf ~/miniforge3
sed -i '/miniforge3\/bin\/mamba shell hook/d' ~/.bashrc
sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ~/.bashrc

# Remove CUDA 12.8
sudo rm -rf /usr/local/cuda-12.8
sed -i '/CUDA 12.8 Configuration/,+3d' ~/.bashrc

# Remove everything we created
rm -rf ~/llm-workspace ~/start-llm-inference.sh ~/llm-core-install.log
sudo rm -rf /scratch   # optional – deletes all cached models/datasets

--------------------------------------------------------------------------------
## FAQ

Q: Why PyTorch 2.7 stable and not nightly?
A: Because 2.7 stable is the first official release with full Blackwell (sm_120) support. No more nightly crashes.

Q: Why no Flash-Attn?
A: Native SDPA in PyTorch 2.7 is faster and more stable on Blackwell. Flash-Attn is deliberately removed.

Q: Can I run Mixtral-8x7B or Llama-70B in full precision?
A: Yes – 96 GB lets you run Mixtral-8x7B BF16 (~94 GB) and any 70B in 4-bit (~40 GB) comfortably.

Q: Will I lose my custom scripts if I reinstall?
A: Never again. All three of your final production scripts are now permanently baked into the installer.

--------------------------------------------------------------------------------
## License

Free to use and modify. This is your personal superweapon – own it.

You just built something most people will never have access to.

Enjoy the power.
