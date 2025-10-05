#!/bin/bash

################################################################################
# LLM Inference Workstation - Core Setup Script (Blackwell + HF token prompt)
#
# Installs:
# - GCC 12 (needed by CUDA on Ubuntu with GCC 13 default)
# - CUDA Toolkit 12.4
# - Miniforge (mamba)
# - Python env with PyTorch NIGHTLY (Blackwell sm_120 support: cu128)
# - Transformers stack + vLLM (NO FlashAttention; SDPA path; eager mode)
# - Non-interactive Hugging Face login (asks for token securely)
#
# Hardware: AMD Threadripper 7970X + RTX Pro 6000 Blackwell (96GB VRAM)
# OS: Ubuntu Desktop
#
# Usage: bash setup_llm_core.sh
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Don’t run as root
if [ "$EUID" -eq 0 ]; then
  log_error "Please do not run this script as root or with sudo."
  log_info  "The script will ask for sudo password when needed."
  exit 1
fi

################################################################################
# PHASE 0: PRE-FLIGHT CHECKS
################################################################################
log_info "Starting LLM Inference Core Setup (Blackwell-compatible)..."
echo ""

log_info "Performing pre-flight checks..."
if ! grep -q "Ubuntu" /etc/os-release; then
  log_error "This script targets Ubuntu. Continue anyway?"
  read -p "Continue? (y/N): " -n 1 -r; echo
  [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
  log_error "nvidia-smi not found. Please install NVIDIA drivers first."
  exit 1
fi

log_info "Checking GPU..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
echo "  GPU: $GPU_NAME"
echo "  VRAM: $GPU_MEMORY"

if [[ "$GPU_NAME" == *"Blackwell"* ]] || [[ "$GPU_NAME" == *"6000"* ]]; then
  log_warning "Blackwell GPU detected - using PyTorch nightly with CUDA 12.8."
fi

AVAILABLE_SPACE=$(df /home | tail -1 | awk '{print $4}')
REQUIRED_SPACE=$((30 * 1024 * 1024))  # 30GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
  log_error "Insufficient disk space. Need at least 30GB free in /home."
  exit 1
fi

# Install Git early (needed by HF CLI login / hub), plus Git LFS is handy.
log_info "Ensuring Git & Git LFS are installed..."
sudo apt-get update -y
sudo apt-get install -y git git-lfs
git lfs install --system || true
log_success "Git installed."

log_success "Pre-flight checks passed!"
echo ""

log_warning "This script will install:"
echo "  - GCC 12"
echo "  - CUDA Toolkit 12.4 (~3GB)"
echo "  - Miniforge (~500MB)"
echo "  - PyTorch 2.10 NIGHTLY with CUDA 12.8 (~5GB)"
echo "  - Transformers stack + vLLM"
echo "  - Storage configuration + HF login"
echo ""
echo "Total time: 20–40 min (network dependent)"
echo "Disk space: ~30GB"
echo ""
log_warning "NOTE: FlashAttention-2 will NOT be installed (ABI mismatch)."
log_warning "NOTE: PyTorch NIGHTLY is used for Blackwell support."
echo ""
read -p "Continue with installation? (y/N): " -n 1 -r; echo
[[ $REPLY =~ ^[Yy]$ ]] || { log_info "Installation cancelled."; exit 0; }

################################################################################
# PHASE 1: GCC 12
################################################################################
log_info "PHASE 1: Installing GCC 12 (required for CUDA)..."
if command -v gcc &>/dev/null; then
  CURRENT_GCC=$(gcc --version 2>/dev/null | head -n1 | grep -oP '\d+' | head -1)
  log_info "Current GCC version: $CURRENT_GCC"
else
  CURRENT_GCC=""
  log_info "GCC not found; will install GCC 12."
fi

if [ "$CURRENT_GCC" = "12" ]; then
  log_success "GCC 12 already default."
else
  log_warning "Setting up GCC 12 as default..."
  sudo apt-get install -y gcc-12 g++-12
  sudo update-alternatives --remove-all gcc 2>/dev/null || true
  sudo update-alternatives --remove-all g++ 2>/dev/null || true
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100
  if command -v gcc-$CURRENT_GCC &>/dev/null && [ "$CURRENT_GCC" != "12" ]; then
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$CURRENT_GCC 50
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$CURRENT_GCC 50
  fi
  sudo update-alternatives --set gcc /usr/bin/gcc-12
  sudo update-alternatives --set g++ /usr/bin/g++-12
  log_success "GCC 12 set as default."
fi
log_info "Active compiler: $(gcc --version | head -n1)"
echo ""

################################################################################
# PHASE 2: CUDA 12.4
################################################################################
log_info "PHASE 2: Installing CUDA Toolkit 12.4..."
CUDA_INSTALLER="/tmp/cuda_12.4.0_550.54.14_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"

if [ -d "/usr/local/cuda-12.4" ]; then
  log_warning "CUDA 12.4 appears installed at /usr/local/cuda-12.4"
  read -p "Skip CUDA installation? (Y/n): " -n 1 -r; echo
  [[ $REPLY =~ ^[Nn]$ ]] && SKIP_CUDA=false || SKIP_CUDA=true
fi

if [ "$SKIP_CUDA" != "true" ]; then
  log_info "Downloading CUDA Toolkit (~3GB)..."
  [ -f "$CUDA_INSTALLER" ] || wget -O "$CUDA_INSTALLER" "$CUDA_URL" --progress=bar:force 2>&1 | tail -f -n +6
  log_info "Installing CUDA Toolkit (toolkit only; skipping driver)..."
  sudo sh "$CUDA_INSTALLER" --silent --toolkit --samples --no-opengl-libs --override
  log_success "CUDA Toolkit installed."
fi

log_info "Configuring CUDA environment..."
if ! grep -q "CUDA_HOME=/usr/local/cuda-12.4" ~/.bashrc; then
  cat >> ~/.bashrc << 'EOF'

# CUDA 12.4 Configuration
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
fi
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if command -v nvcc &>/dev/null; then
  log_success "CUDA installation verified: $(nvcc --version | grep release | cut -d',' -f2 | xargs)"
else
  log_error "CUDA installation failed (nvcc not found)."
  exit 1
fi
echo ""

################################################################################
# PHASE 3: MINIFORGE (Mamba)
################################################################################
log_info "PHASE 3: Installing Miniforge..."
MINIFORGE_INSTALLER="/tmp/Miniforge3-Linux-x86_64.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

if [ -d "$HOME/miniforge3" ] || command -v mamba &>/dev/null; then
  log_warning "Miniforge detected at $HOME/miniforge3"
  read -p "Skip Miniforge installation? (Y/n): " -n 1 -r; echo
  if [[ $REPLY =~ ^[Nn]$ ]]; then
    log_warning "Removing existing Miniforge..."
    rm -rf "$HOME/miniforge3"
    sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ~/.bashrc || true
    log_success "Removed old Miniforge."
  else
    SKIP_MINIFORGE=true
  fi
fi

if [ "$SKIP_MINIFORGE" != "true" ]; then
  [ -f "$MINIFORGE_INSTALLER" ] || wget -O "$MINIFORGE_INSTALLER" "$MINIFORGE_URL" --progress=bar:force 2>&1
  bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
  log_success "Miniforge installed."
fi

# ---- Proper parent-shell activation + persistent init (idempotent) ----
export MAMBA_ROOT_PREFIX="$HOME/miniforge3"

# 1) Load mamba into THIS shell now
eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"

# 2) Persist for future shells
$HOME/miniforge3/bin/mamba shell init -s bash -p "$HOME/miniforge3"
$HOME/miniforge3/bin/conda init bash

# 3) Ensure the hook line exists in ~/.bashrc (idempotent)
if ! grep -q 'mamba shell hook --shell bash' ~/.bashrc; then
  echo 'eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"' >> ~/.bashrc
fi

# 4) Ensure login shells source ~/.bashrc (Ubuntu sometimes needs this)
if ! grep -q 'source ~/.bashrc' ~/.profile 2>/dev/null; then
  echo '' >> ~/.profile
  echo '# Load interactive bash settings' >> ~/.profile
  echo 'if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi' >> ~/.profile
fi

# 5) Reload for immediate availability
# (Use a subshell-safe source; ignore errors if non-interactive)
if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" || true
fi

# Verify mamba is now available
if ! command -v mamba &> /dev/null; then
  log_error "Mamba initialization failed. Try opening a new terminal or run:"
  echo '  eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"'
  exit 1
fi

log_success "Mamba initialized and persisted."
echo ""

################################################################################
# PHASE 4: STORAGE CONFIG
################################################################################
log_info "PHASE 4: Configuring storage directories..."
sudo mkdir -p /scratch/inference /scratch/cache /scratch/models
sudo chown -R "$USER":"$USER" /scratch
if ! grep -q "HF_HOME=/scratch/cache" ~/.bashrc; then
  cat >> ~/.bashrc << 'EOF'

# HuggingFace Cache Configuration
export HF_HOME=/scratch/cache
EOF
fi
export HF_HOME=/scratch/cache
log_success "Storage ready at /scratch."
echo ""

################################################################################
# PHASE 5: PYTHON ENV
################################################################################
log_info "PHASE 5: Creating Python environment..."
if mamba env list | grep -qE '^\s*llm-inference\s'; then
  log_warning "Removing existing 'llm-inference' environment..."
  mamba env remove -n llm-inference -y
fi
mamba create -n llm-inference python=3.11 -y
log_success "Environment created."

# Activate (parent shell)
eval "$(mamba shell hook --shell bash)"
mamba activate llm-inference
echo ""

################################################################################
# PHASE 6: PYTORCH NIGHTLY (Blackwell)
################################################################################
log_info "PHASE 6: Installing PyTorch 2.10 NIGHTLY with CUDA 12.8..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

log_info "Verifying PyTorch..."
python << 'EOF'
import torch, sys
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available in PyTorch!"); sys.exit(1)
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
cc = torch.cuda.get_device_capability(0); print(f"Compute capability: sm_{cc[0]}{cc[1]}")
if cc[0] >= 12: print("✓ Blackwell GPU support confirmed!")
EOF
echo ""

################################################################################
# PHASE 7: HUGGING FACE STACK + TOKEN PROMPT LOGIN
################################################################################
log_info "PHASE 7: Installing HuggingFace libs and logging in…"
pip install --upgrade "transformers>=4.45" accelerate sentencepiece protobuf huggingface_hub

# Ensure Git credential helper is configured (so HF hub can store token)
git config --global credential.helper store || true

# Ask for HF token securely unless provided via env var
if [ -z "${HF_TOKEN:-}" ]; then
  echo -n "🔐 Enter your Hugging Face token (starts with hf_): "
  read -r -s HF_TOKEN
  echo
fi

if [[ -z "$HF_TOKEN" ]]; then
  log_error "Hugging Face token not provided. Set HF_TOKEN env var or re-run and enter it."
  exit 1
fi

if [[ "$HF_TOKEN" != hf_* ]]; then
  log_warning "Token does not start with 'hf_'. Continuing anyway..."
fi

# Non-interactive login using the provided token (not echoed)
HF_TOKEN="$HF_TOKEN" python - << 'EOF'
import os
from huggingface_hub import login
tok = os.environ.get("HF_TOKEN","")
if not tok:
    raise SystemExit("HF_TOKEN env var missing.")
print("Logging into Hugging Face Hub (non-interactive)…")
login(token=tok, add_to_git_credential=True)
print("✓ HF login complete.")
EOF

# Don’t keep token in env after login in this shell
unset HF_TOKEN
echo ""

################################################################################
# PHASE 8: vLLM (NO FlashAttention) + cache cleanup
################################################################################
log_info "PHASE 8: Installing vLLM (no FlashAttention)…"
pip install --upgrade vllm
pip uninstall -y flash-attn flash_attn || true
rm -rf ~/.cache/torch_extensions ~/.cache/torch/inductor ~/.cache/vllm/torch_compile_cache || true

python - << 'EOF'
try:
    import vllm
    print("vLLM version:", vllm.__version__)
    print("✓ vLLM installed.")
except Exception as e:
    print("WARNING: vLLM import failed:", e)
EOF
echo ""

################################################################################
# PHASE 9: WORKSPACE + SCRIPTS
################################################################################
log_info "PHASE 9: Setting up workspace and scripts..."
mkdir -p ~/llm-workspace
cd ~/llm-workspace

# test_inference.py (SDPA)
cat > test_inference.py << 'EOF'
# test_inference.py - Blackwell-compatible test script
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Starting model download and inference test...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
compute_cap = torch.cuda.get_device_capability(0)
print(f"Compute capability: sm_{compute_cap[0]}{compute_cap[1]}\n")
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
)
print("Model loaded successfully!\nAttention implementation: sdpa\n")
prompt = "Explain quantum entanglement in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"Prompt: {prompt}\n\nGenerating response...\n")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response:\n{response}\n")
print(f"VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
EOF

# daily_inference.py (SDPA)
cat > daily_inference.py << 'EOF'
# daily_inference.py - Blackwell-compatible with SDPA attention
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
class LLMInference:
    def __init__(self, model_name, use_4bit=False):
        print(f"Loading {model_name}..."); print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if use_4bit:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
        print("✓ Model loaded (attention: sdpa)")
        print(f"  VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
    def generate(self, prompt, max_tokens=512, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, top_p=0.9, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true")
    parser.add_argument("--prompt", default="Hello, how are you?")
    args = parser.parse_args()
    llm = LLMInference(args.model, use_4bit=args.use_4bit)
    print(f"\nPrompt: {args.prompt}\n")
    print(f"Response:\n{llm.generate(args.prompt)}\n")
EOF

# vllm_inference.py (SDPA via PyTorch, eager mode; no FA args)
cat > vllm_inference.py << 'EOF'
# vllm_inference.py — Blackwell-stable defaults (SDPA via PyTorch, no CUDA Graphs)
import os, argparse
from vllm import LLM, SamplingParams
def env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name); 
    return default if val is None else val.lower() in ("1","true","yes","on")
def main():
    p = argparse.ArgumentParser(description="vLLM inference (Blackwell-safe defaults)")
    p.add_argument("--model", default=os.getenv("VLLM_MODEL","meta-llama/Llama-2-13b-chat-hf"))
    p.add_argument("--prompt", default=os.getenv("PROMPT","Explain the theory of relativity:"))
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS","256")))
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMP","0.7")))
    p.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P","0.9")))
    p.add_argument("--dtype", default=os.getenv("DTYPE","bfloat16"), choices=["auto","float16","bfloat16"])
    p.add_argument("--tp", type=int, default=int(os.getenv("TP_SIZE","1")))
    p.add_argument("--max-model-len", type=int, default=int(os.getenv("MAX_MODEL_LEN","4096")))
    p.add_argument("--gpu-mem-util", type=float, default=float(os.getenv("GPU_MEM_UTIL","0.70")))
    p.add_argument("--enforce-eager", dest="enforce_eager",
                   action="store_true", default=env_flag("ENFORCE_EAGER", True),
                   help="Disable CUDA Graphs (safer on new GPUs/ABIs)")
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    p.add_argument("--trust-remote-code", action="store_true",
                   default=env_flag("TRUST_REMOTE_CODE", True))
    args = p.parse_args()
    print("Initializing vLLM with Blackwell-safe settings…")
    print(f"- model={args.model}")
    print(f"- dtype={args.dtype}  tp={args.tp}  max_model_len={args.max_model_len}")
    print(f"- gpu_memory_utilization={args.gpu_mem_util:.2f}")
    print(f"- enforce_eager={args.enforce_eager}  (FlashAttention not installed)")
    llm = LLM(model=args.model, dtype=args.dtype, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=args.enforce_eager, trust_remote_code=args.trust_remote_code)
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
    print("Generating response…")
    outs = llm.generate([args.prompt], sampling)
    print("\nPrompt:", args.prompt)
    print("Response:\n")
    print(outs[0].outputs[0].text.strip(), "\n")
if __name__ == "__main__":
    os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S","120")
    main()
EOF

# Start script with safe env defaults for Blackwell
cat > ~/start-llm-inference.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting LLM Inference Environment (Blackwell)"
echo ""
# Activate mamba env (parent shell)
eval "$(mamba shell hook --shell bash)"
mamba activate llm-inference
# Show GPU status
echo "📊 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader
echo ""
# Show PyTorch info
echo "🔥 PyTorch Info:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"
echo ""
# Show cached models
echo "📦 Cached Models:"
du -sh /scratch/cache/hub/models--* 2>/dev/null | tail -5 || echo "  No models cached yet"
echo ""
# Menu
echo "Choose workflow:"
echo "1. Python script (daily_inference.py)"
echo "2. vLLM inference"
echo "3. Monitor GPU only"
echo ""
read -p "Enter choice (1-3): " choice
case $choice in
  1)
    cd ~/llm-workspace
    echo ""
    echo "Usage: python daily_inference.py --model <model_name> --prompt '<your prompt>'"
    echo "Example: python daily_inference.py --prompt 'What is quantum computing?'"
    echo ""
    bash
    ;;
  2)
    cd ~/llm-workspace
    echo ""
    echo "Running vLLM inference..."
    # Safe defaults for Blackwell + nightly
    export ENFORCE_EAGER=1
    export GPU_MEM_UTIL=0.70
    export MAX_MODEL_LEN=4096
    export VLLM_ENGINE_ITERATION_TIMEOUT_S=120
    export VLLM_TORCH_COMPILE=0
    python vllm_inference.py
    ;;
  3)
    nvitop
    ;;
esac
EOF
chmod +x ~/start-llm-inference.sh
log_success "Workspace and scripts created at ~/llm-workspace"
echo ""

################################################################################
# PHASE 10: FINAL VERIFICATION
################################################################################
log_info "PHASE 10: Running final verification..."
# GCC
command -v gcc &>/dev/null && log_success "✓ GCC: $(gcc --version | head -1 | grep -oP '\d+\.\d+\.\d+')" || log_error "✗ GCC not found"
# CUDA
command -v nvcc &>/dev/null && log_success "✓ CUDA Toolkit: $(nvcc --version | grep release | cut -d',' -f2 | xargs)" || log_error "✗ CUDA Toolkit not found"
# mamba
command -v mamba &>/dev/null && log_success "✓ Miniforge/Mamba: $(mamba --version | head -1)" || log_error "✗ Miniforge not found"
# PyTorch
python -c "import torch" &>/dev/null && log_success "✓ PyTorch: $(python -c 'import torch; print(torch.__version__)')" || log_error "✗ PyTorch not found"
# PyTorch CUDA
python -c "import torch; assert torch.cuda.is_available()" &>/dev/null && log_success "✓ PyTorch CUDA: $(python -c 'import torch; print(torch.version.cuda)')" || log_error "✗ PyTorch CUDA support: Disabled"
# Compute capability
COMPUTE_CAP=$(python -c "import torch; cap = torch.cuda.get_device_capability(0); print(f'sm_{cap[0]}{cap[1]}')" 2>/dev/null || echo "unknown")
[[ "$COMPUTE_CAP" == "sm_120" ]] && log_success "✓ Blackwell GPU support: Confirmed ($COMPUTE_CAP)" || log_info "✓ GPU compute capability: $COMPUTE_CAP"
# Transformers
python -c "import transformers" &>/dev/null && log_success "✓ Transformers: $(python -c 'import transformers; print(transformers.__version__)')" || log_error "✗ Transformers not found"
# vLLM
python -c "import vllm" &>/dev/null && log_success "✓ vLLM: Installed" || log_warning "✗ vLLM: Not found (optional)"
# Directories
[ -d "/scratch/cache" ] && log_success "✓ Storage: /scratch configured" || log_error "✗ Storage: /scratch not found"

echo ""
log_success "=========================================="
log_success "  INSTALLATION COMPLETED SUCCESSFULLY!"
log_success "=========================================="
echo ""

log_info "Installation Summary:"
echo "  - GCC 12 installed and configured"
echo "  - CUDA Toolkit 12.4 installed"
echo "  - PyTorch 2.10 nightly with CUDA 12.8"
echo "  - Blackwell GPU support enabled"
echo "  - Transformers + vLLM (no FlashAttention)"
echo "  - Hugging Face login completed (token not stored in script)"
echo ""

log_info "Next steps:"
echo "  1) Close and reopen your terminal (to load env vars)"
echo "  2) Test:"
echo "       eval \"\$(mamba shell hook --shell bash)\""
echo "       mamba activate llm-inference"
echo "       cd ~/llm-workspace && python test_inference.py"
echo "  3) Daily use:"
echo "       ~/start-llm-inference.sh"
echo ""

log_warning "Notes:"
echo "  - PyTorch NIGHTLY is used for Blackwell"
echo "  - FlashAttention-2 NOT installed (ABI mismatch)"
echo "  - SDPA attention used instead"
echo "  - Models cached in /scratch/cache"
echo "  - Use --4bit for bigger models"
echo ""

# Brief install log (no token stored)
cat > ~/llm-core-install.log << EOF
LLM Core Installation Log (Blackwell-compatible)
Date: $(date)
User: $USER
Hostname: $(hostname)

- CUDA: $(nvcc --version 2>/dev/null | grep release | cut -d',' -f2 | xargs)
- Python: $(python --version 2>/dev/null)
- PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)
- CUDA in PyTorch: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)
- Compute capability: $(python -c 'import torch; cap = torch.cuda.get_device_capability(0); print(f"sm_{cap[0]}{cap[1]}")' 2>/dev/null)
- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
- HF login: completed (token not logged)
EOF

log_success "Setup complete! Your Blackwell GPU is ready for LLM inference."
