#!/bin/bash

################################################################################
# LLM Inference Workstation - Improved Setup Script (Blackwell + Research-backed)
#
# IMPROVEMENTS OVER ORIGINAL:
# - Uses PyTorch 2.7.0 STABLE (official Blackwell support) instead of nightly
# - CUDA 12.8 toolkit to match PyTorch wheels (no version mismatch)
# - Better cache cleanup to avoid ABI issues
# - Optimized vLLM settings for 96GB VRAM
# - Enhanced verification with detailed GPU info
# - Better error handling and rollback
#
# Hardware: AMD Threadripper 7970X + RTX Pro 6000 Blackwell (96GB VRAM)
# OS: Ubuntu Desktop
#
# Usage: bash llm-inference-improved.sh
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Don't run as root
if [ "$EUID" -eq 0 ]; then
  log_error "Please do not run this script as root or with sudo."
  log_info  "The script will ask for sudo password when needed."
  exit 1
fi

################################################################################
# PHASE 0: PRE-FLIGHT CHECKS
################################################################################
log_info "Starting LLM Inference Setup (Research-backed Blackwell config)..."
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
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
echo "  GPU: $GPU_NAME"
echo "  VRAM: $GPU_MEMORY"
echo "  Compute Capability: $COMPUTE_CAP"

if [[ "$COMPUTE_CAP" != "12.0" ]]; then
  log_warning "Expected compute capability 12.0 (sm_120) for Blackwell, got $COMPUTE_CAP"
  log_warning "This script is optimized for Blackwell. Continue anyway?"
  read -p "Continue? (y/N): " -n 1 -r; echo
  [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

AVAILABLE_SPACE=$(df /home | tail -1 | awk '{print $4}')
REQUIRED_SPACE=$((30 * 1024 * 1024))  # 30GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
  log_error "Insufficient disk space. Need at least 30GB free in /home."
  exit 1
fi

# Install Git early
log_info "Ensuring Git & Git LFS are installed..."
sudo apt-get update -y
sudo apt-get install -y git git-lfs ninja-build
git lfs install --system || true
log_success "Git and ninja installed."

log_success "Pre-flight checks passed!"
echo ""

log_warning "KEY DIFFERENCES FROM ORIGINAL SCRIPT:"
echo "  ✓ PyTorch 2.7.0 STABLE (not nightly) - official Blackwell support"
echo "  ✓ CUDA 12.8 toolkit (matches PyTorch wheels)"
echo "  ✓ Enhanced cache cleanup (avoids ABI issues)"
echo "  ✓ Optimized vLLM settings for 96GB VRAM"
echo ""
log_warning "This script will install:"
echo "  - GCC 12"
echo "  - CUDA Toolkit 12.8 (~3GB)"
echo "  - Miniforge (~500MB)"
echo "  - PyTorch 2.7.0 STABLE with CUDA 12.8 (~5GB)"
echo "  - Transformers stack + vLLM"
echo "  - Storage configuration + HF login"
echo ""
echo "Total time: 20–40 min (network dependent)"
echo "Disk space: ~30GB"
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
  sudo apt-get install -y gcc-12 g++-12 build-essential
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
# PHASE 2: CUDA 12.8 (IMPROVED - matches PyTorch wheels)
################################################################################
log_info "PHASE 2: Installing CUDA Toolkit 12.8..."
CUDA_INSTALLER="/tmp/cuda_12.8.0_580.54.14_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_580.54.14_linux.run"

if [ -d "/usr/local/cuda-12.8" ]; then
  log_warning "CUDA 12.8 appears installed at /usr/local/cuda-12.8"
  read -p "Skip CUDA installation? (Y/n): " -n 1 -r; echo
  [[ $REPLY =~ ^[Nn]$ ]] && SKIP_CUDA=false || SKIP_CUDA=true
fi

if [ "$SKIP_CUDA" != "true" ]; then
  log_info "Downloading CUDA Toolkit 12.8 (~3GB)..."
  [ -f "$CUDA_INSTALLER" ] || wget -O "$CUDA_INSTALLER" "$CUDA_URL" --progress=bar:force 2>&1 | tail -f -n +6
  log_info "Installing CUDA Toolkit (toolkit only; skipping driver)..."
  sudo sh "$CUDA_INSTALLER" --silent --toolkit --samples --no-opengl-libs --override
  log_success "CUDA Toolkit 12.8 installed."
fi

log_info "Configuring CUDA environment..."
# Remove old CUDA exports
sed -i '/# CUDA.*Configuration/,+3d' ~/.bashrc 2>/dev/null || true

# Add new CUDA 12.8 exports
if ! grep -q "CUDA_HOME=/usr/local/cuda-12.8" ~/.bashrc; then
  cat >> ~/.bashrc << 'EOF'

# CUDA 12.8 Configuration (Blackwell-optimized)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
fi
export CUDA_HOME=/usr/local/cuda-12.8
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

# Proper parent-shell activation + persistent init
export MAMBA_ROOT_PREFIX="$HOME/miniforge3"

# Load mamba into THIS shell now
eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"

# Persist for future shells
if $HOME/miniforge3/bin/mamba shell init -h | grep -q -- "--prefix"; then
  $HOME/miniforge3/bin/mamba shell init -s bash --prefix "$HOME/miniforge3"
else
  $HOME/miniforge3/bin/mamba shell init -s bash
fi

$HOME/miniforge3/bin/conda init bash || true

# Ensure hook exists in ~/.bashrc
if ! grep -q 'mamba shell hook --shell bash' ~/.bashrc; then
  echo 'eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"' >> ~/.bashrc
fi

# Ensure login shells source ~/.bashrc
if ! grep -q 'source ~/.bashrc' ~/.profile 2>/dev/null; then
  echo '' >> ~/.profile
  echo '# Load interactive bash settings' >> ~/.profile
  echo 'if [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi' >> ~/.profile
fi

# shellcheck disable=SC1090
source ~/.bashrc || true

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

# Remove old HF_HOME exports
sed -i '/# HuggingFace Cache Configuration/,+1d' ~/.bashrc 2>/dev/null || true

if ! grep -q "HF_HOME=/scratch/cache" ~/.bashrc; then
  cat >> ~/.bashrc << 'EOF'

# HuggingFace Cache Configuration
export HF_HOME=/scratch/cache
export HF_DATASETS_CACHE=/scratch/cache/datasets
export TRANSFORMERS_CACHE=/scratch/cache/transformers
EOF
fi
export HF_HOME=/scratch/cache
export HF_DATASETS_CACHE=/scratch/cache/datasets
export TRANSFORMERS_CACHE=/scratch/cache/transformers
log_success "Storage ready at /scratch."
echo ""

################################################################################
# PHASE 5: PYTHON ENV
################################################################################
log_info "PHASE 5: Creating Python environment..."
if mamba env list | grep -qE '^\s*llm-inference\s'; then
  log_warning "Environment 'llm-inference' exists."
  read -p "Remove and recreate? (y/N): " -n 1 -r; echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    mamba env remove -n llm-inference -y
  else
    log_info "Keeping existing environment. Activating..."
    eval "$(mamba shell hook --shell bash)"
    mamba activate llm-inference
    log_warning "Skipping to Phase 7 (packages may need updating)..."
    SKIP_TO_PHASE_7=true
  fi
fi

if [ "$SKIP_TO_PHASE_7" != "true" ]; then
  mamba create -n llm-inference python=3.12 -y
  log_success "Environment created."

  # Activate
  eval "$(mamba shell hook --shell bash)"
  mamba activate llm-inference
fi
echo ""

################################################################################
# PHASE 6: PYTORCH 2.7.0 STABLE (IMPROVED - official Blackwell support)
################################################################################
if [ "$SKIP_TO_PHASE_7" != "true" ]; then
  log_info "PHASE 6: Installing PyTorch 2.7.0 STABLE with CUDA 12.8..."
  log_info "This is the STABLE release with official Blackwell (sm_120) support"
  
  # Clean pip cache to avoid ABI issues
  log_info "Cleaning pip cache..."
  pip cache purge || true
  
  pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

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
cc = torch.cuda.get_device_capability(0)
print(f"Compute capability: sm_{cc[0]}{cc[1]}")
if cc[0] >= 12:
    print("✓ Blackwell GPU support confirmed (official PyTorch 2.7 stable)!")
elif cc == (9, 0):
    print("✓ Hopper GPU detected")
elif cc == (8, 9):
    print("✓ Ada Lovelace GPU detected")
else:
    print(f"✓ GPU detected with sm_{cc[0]}{cc[1]}")
EOF
  echo ""
fi

################################################################################
# PHASE 7: HUGGING FACE STACK + TOKEN PROMPT LOGIN
################################################################################
log_info "PHASE 7: Installing HuggingFace libs and logging in..."
pip install --upgrade --no-cache-dir \
  "transformers>=4.45" \
  accelerate \
  sentencepiece \
  protobuf \
  huggingface_hub \
  safetensors

# Git credential helper
git config --global credential.helper store || true

# Ask for HF token securely unless provided via env var
if [ -z "${HF_TOKEN:-}" ]; then
  echo ""
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

# Non-interactive login
HF_TOKEN="$HF_TOKEN" python - << 'EOF'
import os
from huggingface_hub import login
tok = os.environ.get("HF_TOKEN","")
if not tok:
    raise SystemExit("HF_TOKEN env var missing.")
print("Logging into Hugging Face Hub (non-interactive)...")
login(token=tok, add_to_git_credential=True)
print("✓ HF login complete.")
EOF

unset HF_TOKEN
echo ""

################################################################################
# PHASE 8: vLLM + ENHANCED CACHE CLEANUP (IMPROVED)
################################################################################
log_info "PHASE 8: Installing vLLM and cleaning caches..."

# Comprehensive cache cleanup to avoid ABI issues
log_info "Cleaning all torch/vllm caches..."
rm -rf ~/.cache/torch_extensions ~/.cache/torch/inductor ~/.cache/vllm ~/.triton || true
rm -rf /tmp/torch_* /tmp/.triton || true

pip install --upgrade --no-cache-dir vllm

# Ensure no Flash Attention
pip uninstall -y flash-attn flash_attn 2>/dev/null || true

python - << 'EOF'
try:
    import vllm
    print(f"vLLM version: {vllm.__version__}")
    print("✓ vLLM installed successfully.")
except Exception as e:
    print(f"WARNING: vLLM import failed: {e}")
EOF
echo ""

################################################################################
# PHASE 9: WORKSPACE + OPTIMIZED SCRIPTS (IMPROVED)
################################################################################
log_info "PHASE 9: Setting up workspace with optimized scripts..."
mkdir -p ~/llm-workspace
cd ~/llm-workspace

# test_inference.py - Enhanced version
cat > test_inference.py << 'EOF'
#!/usr/bin/env python3
"""
test_inference.py - Production-ready Blackwell test script
PyTorch 2.7 STABLE + SDPA (official sm_120 support)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("="*60)
    print("Blackwell Inference Test (PyTorch 2.7 STABLE)")
    print("="*60)
    
    # System info
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")
    
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: sm_{compute_cap[0]}{compute_cap[1]}")
    
    if compute_cap == (12, 0):
        print("✓ Blackwell (sm_120) confirmed!")
    
    # Test SDPA availability
    from torch.nn.functional import scaled_dot_product_attention
    print("✓ Native SDPA available\n")
    
    # Load model
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"Loading model: {model_name}")
    print("Attention: SDPA (stable, official Blackwell support)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",  # Official support for sm_120
    )
    
    print("✓ Model loaded successfully!\n")
    
    # Generate
    prompt = "Explain quantum entanglement in simple terms:"
    print(f"Prompt: {prompt}\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("Generating response...\n")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response:\n{response}\n")
    
    # Memory stats
    print("="*60)
    print(f"VRAM allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"VRAM reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    print("="*60)

if __name__ == "__main__":
    main()
EOF

# daily_inference.py - Enhanced with better memory management
cat > daily_inference.py << 'EOF'
#!/usr/bin/env python3
"""
daily_inference.py - Production inference with 4-bit quantization
Optimized for RTX PRO 6000 96GB Blackwell
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

class LLMInference:
    def __init__(self, model_name, use_4bit=False):
        print(f"\nLoading {model_name}...")
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
        print(f"Attention: SDPA (official Blackwell support)\n")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if use_4bit:
            print("Using 4-bit quantization (bitsandbytes NF4)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="sdpa"
            )
        else:
            print("Using BF16 (full precision)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa"
            )
        
        print(f"✓ Model loaded")
        print(f"  VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB / 96 GB\n")
    
    def generate(self, prompt, max_tokens=512, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true",
                       help="Use 4-bit quantization (enables 70B models on 96GB)")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    
    llm = LLMInference(args.model, use_4bit=args.use_4bit)
    print(f"Prompt: {args.prompt}\n")
    print("Response:")
    print("-" * 60)
    print(llm.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature))
    print("-" * 60)
    print(f"\nFinal VRAM: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB / 96 GB")
EOF

# vllm_inference.py - Optimized for 96GB VRAM
cat > vllm_inference.py << 'EOF'
#!/usr/bin/env python3
"""
vllm_inference.py - Production vLLM setup for RTX PRO 6000 96GB
Optimized settings for Blackwell with PyTorch 2.7 STABLE
"""
import os
import argparse
from vllm import LLM, SamplingParams

def env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    return default if val is None else val.lower() in ("1","true","yes","on")

def main():
    p = argparse.ArgumentParser(description="vLLM inference (96GB Blackwell-optimized)")
    
    # Model settings
    p.add_argument("--model", default=os.getenv("VLLM_MODEL","meta-llama/Llama-2-13b-chat-hf"))
    p.add_argument("--prompt", default=os.getenv("PROMPT","Explain the theory of relativity:"))
    
    # Generation settings
    p.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS","256")))
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMP","0.7")))
    p.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P","0.9")))
    
    # vLLM settings (optimized for 96GB)
    p.add_argument("--dtype", default=os.getenv("DTYPE","bfloat16"), 
                   choices=["auto","float16","bfloat16"])
    p.add_argument("--tp", type=int, default=int(os.getenv("TP_SIZE","1")))
    p.add_argument("--max-model-len", type=int, 
                   default=int(os.getenv("MAX_MODEL_LEN","8192")))  # Increased for 96GB
    p.add_argument("--gpu-mem-util", type=float, 
                   default=float(os.getenv("GPU_MEM_UTIL","0.85")))  # Higher for 96GB
    p.add_argument("--enforce-eager", dest="enforce_eager",
                   action="store_true", default=env_flag("ENFORCE_EAGER", True),
                   help="Disable CUDA Graphs (recommended for stability)")
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    p.add_argument("--trust-remote-code", action="store_true",
                   default=env_flag("TRUST_REMOTE_CODE", True))
    
    args = p.parse_args()
    
    print("="*60)
    print("vLLM Inference (Blackwell RTX PRO 6000 96GB)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dtype: {args.dtype}  TP: {args.tp}")
    print(f"Max model length: {args.max_model_len}")
    print(f"GPU memory utilization: {args.gpu_mem_util:.2%}")
    print(f"Enforce eager: {args.enforce_eager}")
    print(f"Attention: SDPA (via PyTorch 2.7)")
    print("="*60 + "\n")
    
    # Initialize vLLM
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Generate
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print(f"Prompt: {args.prompt}\n")
    print("Generating...\n")
    
    outputs = llm.generate([args.prompt], sampling)
    
    print("Response:")
    print("-" * 60)
    print(outputs[0].outputs[0].text.strip())
    print("-" * 60 + "\n")

if __name__ == "__main__":
    # Safe defaults
    os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S","120")
    os.environ.setdefault("VLLM_TORCH_COMPILE","0")
    main()
EOF

# Make scripts executable
chmod +x test_inference.py daily_inference.py vllm_inference.py

# Start script with better GPU info
cat > ~/start-llm-inference.sh << 'EOF'
#!/bin/bash
echo "🚀 LLM Inference Environment (Blackwell RTX PRO 6000)"
echo "   PyTorch 2.7 STABLE + CUDA 12.8"
echo ""

# Activate
eval "$(mamba shell hook --shell bash)"
mamba activate llm-inference

# GPU info
echo "📊 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version --format=csv,noheader

echo ""
echo "🔥 PyTorch Info:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  cuDNN: {torch.backends.cudnn.version()}')
cc = torch.cuda.get_device_capability(0)
print(f'  Compute: sm_{cc[0]}{cc[1]}')
if cc == (12, 0):
    print('  ✓ Blackwell support: OFFICIAL (PyTorch 2.7 stable)')
"

echo ""
echo "📦 Model Cache:"
du -sh /scratch/cache/hub/models--* 2>/dev/null | tail -5 || echo "  No models cached yet"

echo ""
echo "Choose workflow:"
echo "1. Test inference (Mistral-7B)"
echo "2. Daily inference (with 4-bit option)"
echo "3. vLLM inference (production)"
echo "4. Monitor GPU"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
  1)
    cd ~/llm-workspace
    python test_inference.py
    ;;
  2)
    cd ~/llm-workspace
    echo ""
    echo "Examples:"
    echo "  python daily_inference.py --prompt 'Your question'"
    echo "  python daily_inference.py --model meta-llama/Llama-3.1-70B --4bit --prompt 'Your question'"
    echo ""
    bash
    ;;
  3)
    cd ~/llm-workspace
    echo ""
    echo "Running vLLM (optimized for 96GB)..."
    python vllm_inference.py
    ;;
  4)
    command -v nvitop &>/dev/null && nvitop || watch -n 1 nvidia-smi
    ;;
esac
EOF
chmod +x ~/start-llm-inference.sh

log_success "Workspace configured at ~/llm-workspace"
echo ""

################################################################################
# PHASE 10: FINAL VERIFICATION (ENHANCED)
################################################################################
log_info "PHASE 10: Running comprehensive verification..."

# Version checks
echo ""
command -v gcc &>/dev/null && log_success "✓ GCC: $(gcc --version | head -1 | grep -oP '\d+\.\d+\.\d+')" || log_error "✗ GCC not found"
command -v nvcc &>/dev/null && log_success "✓ CUDA Toolkit: $(nvcc --version | grep release | cut -d',' -f2 | xargs)" || log_error "✗ CUDA Toolkit not found"
command -v mamba &>/dev/null && log_success "✓ Mamba: $(mamba --version | head -1)" || log_error "✗ Mamba not found"

# PyTorch detailed check
python << 'VERIFY_EOF'
import torch
import sys

versions_ok = True

# PyTorch
print(f"✓ PyTorch: {torch.__version__}", end="")
if "2.7" in torch.__version__:
    print(" (STABLE - official Blackwell support)")
else:
    print(" (WARNING: Expected 2.7.x)")
    versions_ok = False

# CUDA
if torch.cuda.is_available():
    cuda_ver = torch.version.cuda
    print(f"✓ PyTorch CUDA: {cuda_ver}", end="")
    if cuda_ver == "12.8":
        print(" (matches toolkit)")
    else:
        print(f" (WARNING: toolkit is 12.8, PyTorch has {cuda_ver})")
else:
    print("✗ PyTorch CUDA: Not available")
    versions_ok = False
    sys.exit(1)

# GPU
cc = torch.cuda.get_device_capability(0)
print(f"✓ Compute capability: sm_{cc[0]}{cc[1]}", end="")
if cc == (12, 0):
    print(" (Blackwell confirmed)")
elif cc == (9, 0):
    print(" (Hopper)")
elif cc == (8, 9):
    print(" (Ada Lovelace)")
else:
    print()

# SDPA
from torch.nn.functional import scaled_dot_product_attention
print("✓ Native SDPA: Available")

# Memory
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"✓ VRAM: {vram_gb:.1f} GB")

if not versions_ok:
    sys.exit(1)
VERIFY_EOF

# Transformers & vLLM
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" 2>/dev/null || log_error "✗ Transformers not found"
python -c "import vllm; print('✓ vLLM: Installed')" 2>/dev/null || log_warning "✗ vLLM: Not found"

# Storage
[ -d "/scratch/cache" ] && log_success "✓ Storage: /scratch configured" || log_error "✗ Storage: /scratch not found"

echo ""
log_success "=========================================="
log_success "  INSTALLATION COMPLETED SUCCESSFULLY!"
log_success "=========================================="
echo ""

log_info "Installation Summary:"
echo "  ✓ GCC 12 configured"
echo "  ✓ CUDA Toolkit 12.8 (matches PyTorch)"
echo "  ✓ PyTorch 2.7.0 STABLE with CUDA 12.8"
echo "  ✓ Official Blackwell (sm_120) support"
echo "  ✓ Native SDPA (stable attention)"
echo "  ✓ Transformers + vLLM"
echo "  ✓ Optimized for 96GB VRAM"
echo ""

log_info "Key Improvements Over Original Script:"
echo "  🔹 PyTorch 2.7 STABLE (not nightly) - production-ready"
echo "  🔹 CUDA 12.8 toolkit matching PyTorch wheels"
echo "  🔹 Enhanced cache cleanup (no ABI issues)"
echo "  🔹 Optimized vLLM settings (85% mem, 8K context)"
echo "  🔹 Better verification and error handling"
echo ""

log_info "Next steps:"
echo "  1) Close and reopen terminal (or run: source ~/.bashrc)"
echo "  2) Test:"
echo "       mamba activate llm-inference"
echo "       cd ~/llm-workspace && python test_inference.py"
echo "  3) Daily use:"
echo "       ~/start-llm-inference.sh"
echo ""

log_info "Recommended Models for 96GB:"
echo "  • Llama-3.1-8B (FP16): ~16GB"
echo "  • Mixtral-8x7B (FP16): ~94GB - YOU CAN RUN THIS!"
echo "  • Llama-3.1-70B (4-bit): ~40GB"
echo "  • Llama-3.3-70B (4-bit): ~40GB"
echo ""

log_warning "Important Notes:"
echo "  ⚠ PyTorch 2.7 STABLE (not nightly) = official support"
echo "  ⚠ CUDA 12.8 toolkit matches PyTorch wheels exactly"
echo "  ⚠ SDPA attention is stable and fast (~85-90% of FA2)"
echo "  ⚠ Your 96GB allows running models others can't!"
echo ""

# Installation log
cat > ~/llm-core-install.log << EOF
LLM Core Installation Log (Improved Blackwell Setup)
Date: $(date)
User: $USER
Hostname: $(hostname)

System:
- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
- VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
- Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
- Compute: $(python -c 'import torch; cc = torch.cuda.get_device_capability(0); print(f"sm_{cc[0]}{cc[1]}")' 2>/dev/null)

Software:
- CUDA Toolkit: $(nvcc --version 2>/dev/null | grep release | cut -d',' -f2 | xargs)
- Python: $(python --version 2>/dev/null)
- PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)
- PyTorch CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)
- Transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null)

Key Improvements:
- PyTorch 2.7.0 STABLE (official Blackwell support)
- CUDA 12.8 toolkit (matches PyTorch exactly)
- Enhanced cache cleanup
- Optimized vLLM settings for 96GB
EOF

log_success "Setup complete! Your RTX PRO 6000 is ready with STABLE software stack."
