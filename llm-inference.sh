#!/bin/bash

################################################################################
# LLM Inference Workstation - Core Setup Script (Fixed for Blackwell)
# 
# This script installs ONLY the core components (Steps 1-3.3):
# - GCC 12 (required for CUDA on Ubuntu with GCC 13 default)
# - CUDA Toolkit 12.4
# - Miniforge (mamba)
# - Python environment with PyTorch NIGHTLY (for Blackwell sm_120 support)
#
# Hardware: AMD Threadripper 7970X + RTX Pro 6000 Blackwell (96GB VRAM)
# OS: Ubuntu Desktop
#
# IMPORTANT: This uses PyTorch 2.10 nightly with CUDA 12.8 for Blackwell support
# FlashAttention-2 is SKIPPED due to ABI incompatibility with nightly builds
#
# Usage: bash setup_llm_core.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    log_error "Please do not run this script as root or with sudo"
    log_info "The script will ask for sudo password when needed"
    exit 1
fi

################################################################################
# PHASE 0: PRE-FLIGHT CHECKS
################################################################################

log_info "Starting LLM Inference Core Setup (Blackwell-compatible)..."
echo ""

log_info "Performing pre-flight checks..."

# Check Ubuntu version
if ! grep -q "Ubuntu" /etc/os-release; then
    log_error "This script is designed for Ubuntu. Your OS may not be supported."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

log_info "Checking GPU..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
echo "  GPU: $GPU_NAME"
echo "  VRAM: $GPU_MEMORY"

# Check for Blackwell architecture
if [[ "$GPU_NAME" == *"Blackwell"* ]] || [[ "$GPU_NAME" == *"6000"* ]]; then
    log_warning "Blackwell GPU detected - will use PyTorch nightly with CUDA 12.8"
fi

# Check available disk space
AVAILABLE_SPACE=$(df /home | tail -1 | awk '{print $4}')
REQUIRED_SPACE=$((30 * 1024 * 1024))  # 30GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    log_error "Insufficient disk space. Need at least 30GB free in /home"
    exit 1
fi

log_success "Pre-flight checks passed!"
echo ""

# Confirmation prompt
log_warning "This script will install:"
echo "  - GCC 12 (required for CUDA)"
echo "  - CUDA Toolkit 12.4 (~3GB)"
echo "  - Miniforge (~500MB)"
echo "  - PyTorch 2.10 NIGHTLY with CUDA 12.8 (~5GB)"
echo "  - HuggingFace transformers stack"
echo "  - Storage configuration"
echo ""
echo "Total installation time: 20-40 minutes (depending on internet speed)"
echo "Total disk space required: ~30GB"
echo ""
log_warning "NOTE: FlashAttention-2 will NOT be installed due to ABI incompatibility"
log_warning "NOTE: This uses PyTorch NIGHTLY for Blackwell GPU support"
echo ""
read -p "Continue with installation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Installation cancelled."
    exit 0
fi

################################################################################
# PHASE 1: GCC 12 INSTALLATION
################################################################################

log_info "PHASE 1: Installing GCC 12 (required for CUDA)..."

# Check current GCC version
if command -v gcc &> /dev/null; then
    CURRENT_GCC=$(gcc --version 2>/dev/null | head -n1 | grep -oP '\d+' | head -1)
    log_info "Current GCC version: $CURRENT_GCC"
else
    CURRENT_GCC=""
    log_info "GCC not found, will install GCC 12"
fi

if [ "$CURRENT_GCC" = "12" ]; then
    log_success "GCC 12 is already the default compiler"
else
    log_warning "GCC ${CURRENT_GCC:-not found} detected, CUDA requires GCC 12"
    
    # Check if GCC 12 is installed
    if ! command -v gcc-12 &> /dev/null; then
        log_info "Installing GCC 12..."
        sudo apt-get update
        sudo apt-get install -y gcc-12 g++-12
        log_success "GCC 12 installed"
    else
        log_info "GCC 12 already installed"
    fi
    
    # Set GCC 12 as default using update-alternatives
    log_info "Setting GCC 12 as default compiler..."
    
    # Remove existing alternatives if they exist
    sudo update-alternatives --remove-all gcc 2>/dev/null || true
    sudo update-alternatives --remove-all g++ 2>/dev/null || true
    
    # Install alternatives for GCC 12
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100
    
    # Also add the newer GCC if it exists
    if command -v gcc-$CURRENT_GCC &> /dev/null && [ "$CURRENT_GCC" != "12" ]; then
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$CURRENT_GCC 50
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$CURRENT_GCC 50
    fi
    
    # Set GCC 12 as default
    sudo update-alternatives --set gcc /usr/bin/gcc-12
    sudo update-alternatives --set g++ /usr/bin/g++-12
    
    log_success "GCC 12 set as default compiler"
fi

# Verify
NEW_GCC=$(gcc --version | head -n1)
log_info "Active compiler: $NEW_GCC"

echo ""

################################################################################
# PHASE 2: CUDA TOOLKIT INSTALLATION
################################################################################

log_info "PHASE 2: Installing CUDA Toolkit 12.4..."

CUDA_INSTALLER="/tmp/cuda_12.4.0_550.54.14_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"

# Check if CUDA is already installed
if [ -d "/usr/local/cuda-12.4" ]; then
    log_warning "CUDA 12.4 appears to be already installed at /usr/local/cuda-12.4"
    read -p "Skip CUDA installation? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        log_info "Skipping CUDA installation..."
        SKIP_CUDA=true
    fi
fi

if [ "$SKIP_CUDA" != "true" ]; then
    log_info "Downloading CUDA Toolkit (~3GB, this may take a while)..."
    
    if [ ! -f "$CUDA_INSTALLER" ]; then
        wget -O "$CUDA_INSTALLER" "$CUDA_URL" --progress=bar:force 2>&1 | tail -f -n +6
    else
        log_info "CUDA installer already downloaded."
    fi
    
    log_info "Installing CUDA Toolkit (requires sudo)..."
    log_info "Installing toolkit only (skipping driver)..."
    
    sudo sh "$CUDA_INSTALLER" --silent --toolkit --samples --no-opengl-libs --override
    
    log_success "CUDA Toolkit installed!"
fi

# Add CUDA to environment
log_info "Configuring CUDA environment variables..."

if ! grep -q "CUDA_HOME=/usr/local/cuda-12.4" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# CUDA 12.4 Configuration
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    log_success "CUDA environment variables added to ~/.bashrc"
else
    log_info "CUDA environment variables already configured"
fi

# Source for current session
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep release | cut -d',' -f2)
    log_success "CUDA installation verified:$NVCC_VERSION"
else
    log_error "CUDA installation failed. nvcc not found."
    exit 1
fi

echo ""

################################################################################
# PHASE 3: MINIFORGE INSTALLATION
################################################################################

log_info "PHASE 3: Installing Miniforge..."

MINIFORGE_INSTALLER="/tmp/Miniforge3-Linux-x86_64.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

# Check if Miniforge directory exists OR if mamba command exists
if [ -d "$HOME/miniforge3" ] || command -v mamba &> /dev/null; then
    log_warning "Miniforge appears to be already installed at $HOME/miniforge3"
    read -p "Skip Miniforge installation? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        log_info "Skipping Miniforge installation..."
        SKIP_MINIFORGE=true
    else
        log_warning "Removing existing Miniforge installation..."
        rm -rf "$HOME/miniforge3"
        # Also remove conda initialization from bashrc
        sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ~/.bashrc
        log_success "Existing installation removed"
    fi
fi

if [ "$SKIP_MINIFORGE" != "true" ]; then
    log_info "Downloading Miniforge..."
    
    if [ ! -f "$MINIFORGE_INSTALLER" ]; then
        wget -O "$MINIFORGE_INSTALLER" "$MINIFORGE_URL" --progress=bar:force 2>&1
    else
        log_info "Miniforge installer already downloaded."
    fi
    
    log_info "Installing Miniforge..."
    bash "$MINIFORGE_INSTALLER" -b -p "$HOME/miniforge3"
    
    # Initialize using the new mamba method
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    $HOME/miniforge3/bin/conda init bash
    
    # Use the new mamba shell init (replaces deprecated mamba.sh)
    $HOME/miniforge3/bin/mamba shell init -s bash -p "$HOME/miniforge3"
    
    log_success "Miniforge installed!"
fi

# Ensure mamba is properly initialized for this session
# Source conda first
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi

# For mamba, use the modern initialization
export MAMBA_ROOT_PREFIX="$HOME/miniforge3"
if [ -f "$HOME/miniforge3/bin/mamba" ]; then
    eval "$($HOME/miniforge3/bin/mamba shell hook -s bash)"
fi

# Verify mamba is now available
if ! command -v mamba &> /dev/null; then
    log_error "Mamba initialization failed. Please close and reopen your terminal, then run the script again."
    exit 1
fi

log_success "Mamba initialized successfully"

echo ""

################################################################################
# PHASE 4: STORAGE CONFIGURATION
################################################################################

log_info "PHASE 4: Configuring storage directories..."

# Create directories
log_info "Creating /scratch directories..."
sudo mkdir -p /scratch/inference
sudo mkdir -p /scratch/cache
sudo mkdir -p /scratch/models
sudo chown -R $USER:$USER /scratch

log_success "Storage directories created at /scratch"

# Configure HuggingFace cache
log_info "Configuring HuggingFace cache location..."

if ! grep -q "HF_HOME=/scratch/cache" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# HuggingFace Cache Configuration
export HF_HOME=/scratch/cache
EOF
    log_success "HuggingFace cache configured"
else
    log_info "HuggingFace cache already configured"
fi

# Source for current session
export HF_HOME=/scratch/cache

echo ""

################################################################################
# PHASE 5: PYTHON ENVIRONMENT SETUP
################################################################################

log_info "PHASE 5: Creating Python environment..."

# Source mamba for this session
eval "$(mamba shell.bash hook 2>/dev/null)" || eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

# Remove existing llm-inference environment if it exists
if mamba env list | grep -q "llm-inference"; then
    log_warning "Removing existing 'llm-inference' environment..."
    mamba env remove -n llm-inference -y
    log_success "Old environment removed"
fi

log_info "Creating fresh llm-inference environment with Python 3.11..."
mamba create -n llm-inference python=3.11 -y
log_success "Environment created!"

# Activate environment
mamba activate llm-inference

echo ""

################################################################################
# PHASE 6: PYTORCH NIGHTLY INSTALLATION (BLACKWELL SUPPORT)
################################################################################

log_info "PHASE 6: Installing PyTorch 2.10 NIGHTLY with CUDA 12.8 support..."
log_warning "This is required for Blackwell GPU (sm_120 compute capability)"

log_info "This will download ~5GB of packages..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch installation
log_info "Verifying PyTorch installation..."
python << 'EOF'
import torch
import sys

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available in PyTorch!")
    sys.exit(1)

print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check compute capability
compute_capability = torch.cuda.get_device_capability(0)
print(f"Compute capability: sm_{compute_capability[0]}{compute_capability[1]}")

if compute_capability[0] >= 12:
    print("✓ Blackwell GPU support confirmed!")
EOF

if [ $? -eq 0 ]; then
    log_success "PyTorch with CUDA support installed successfully!"
else
    log_error "PyTorch installation verification failed!"
    exit 1
fi

echo ""

################################################################################
# PHASE 7: HUGGINGFACE STACK INSTALLATION
################################################################################

log_info "PHASE 7: Installing HuggingFace and inference libraries..."

log_info "Installing core libraries..."
pip install transformers accelerate sentencepiece protobuf

log_info "Installing quantization support (bitsandbytes)..."
pip install bitsandbytes

log_info "Installing monitoring tools..."
pip install nvitop gpustat

log_success "HuggingFace stack installed!"

echo ""

################################################################################
# PHASE 8: VLLM INSTALLATION
################################################################################

log_info "PHASE 8: Installing vLLM for high-performance inference..."

log_info "This will download ~2GB of packages..."
pip install vllm

# Verify vLLM
if python -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    log_success "vLLM installed successfully! Version: $VLLM_VERSION"
else
    log_warning "vLLM installation may have issues, but continuing..."
fi

echo ""

################################################################################
# PHASE 9: WORKSPACE SETUP
################################################################################

log_info "PHASE 9: Setting up workspace and scripts..."

# Create workspace directory
mkdir -p ~/llm-workspace
cd ~/llm-workspace

# Create test inference script with SDPA attention (no FlashAttention-2)
log_info "Creating test_inference.py..."
cat > test_inference.py << 'EOF'
# test_inference.py - Blackwell-compatible test script
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Starting model download and inference test...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check compute capability
compute_cap = torch.cuda.get_device_capability(0)
print(f"Compute capability: sm_{compute_cap[0]}{compute_cap[1]}")
print()

# Download and load model (will cache to /scratch/cache)
# Using Mistral-7B-Instruct instead of Llama (no authentication required)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use SDPA attention (compatible with PyTorch nightly)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",  # Use scaled dot-product attention
)

print(f"Model loaded successfully!")
print(f"Attention implementation: sdpa (scaled dot-product)")
print()

# Test inference
prompt = "Explain quantum entanglement in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print(f"Prompt: {prompt}\n")
print("Generating response...\n")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response:\n{response}\n")

# Check VRAM usage
print(f"VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
EOF

# Create daily inference script
log_info "Creating daily_inference.py..."
cat > daily_inference.py << 'EOF'
# daily_inference.py - Blackwell-compatible with SDPA attention
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

class LLMInference:
    def __init__(self, model_name, use_4bit=False):
        """Initialize model for inference"""
        print(f"Loading {model_name}...")
        print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if use_4bit:
            # 4-bit quantization - fits larger models in VRAM
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="sdpa",  # Use SDPA, not flash_attention_2
            )
        else:
            # Full precision (BF16)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",  # Use SDPA, not flash_attention_2
            )
        
        print(f"✓ Model loaded (attention: sdpa)")
        print(f"  VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
    
    def generate(self, prompt, max_tokens=512, temperature=0.7):
        """Generate response"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-13b-chat-hf", help="Model name")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Prompt text")
    args = parser.parse_args()
    
    # Initialize
    llm = LLMInference(args.model, use_4bit=args.use_4bit)
    
    # Generate
    print(f"\nPrompt: {args.prompt}\n")
    response = llm.generate(args.prompt)
    print(f"Response:\n{response}\n")
EOF

# Create vLLM inference script
log_info "Creating vllm_inference.py..."
cat > vllm_inference.py << 'EOF'
# vllm_inference.py - Blackwell-compatible
from vllm import LLM, SamplingParams

print("Initializing vLLM with Blackwell GPU support...")

# Initialize (loads model once)
llm = LLM(
    model="meta-llama/Llama-2-13b-chat-hf",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,  # Use 90% of VRAM
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

# Single prompt
prompts = ["Explain the theory of relativity:"]

print("Generating response...")
# Generate (much faster than standard HuggingFace)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"\nPrompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}\n")
EOF

# Create startup script
log_info "Creating startup script..."
cat > ~/start-llm-inference.sh << 'EOF'
#!/bin/bash
# Start LLM inference environment

echo "🚀 Starting LLM Inference Environment (Blackwell)"
echo ""

# Activate mamba environment
source ~/miniforge3/bin/activate llm-inference

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

# Options
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

log_info "Checking installations..."

# Check GCC
if command -v gcc &> /dev/null; then
    log_success "✓ GCC: $(gcc --version | head -1 | grep -oP '\d+\.\d+\.\d+')"
else
    log_error "✗ GCC not found"
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    log_success "✓ CUDA Toolkit: $(nvcc --version | grep release | cut -d',' -f2 | xargs)"
else
    log_error "✗ CUDA Toolkit not found"
fi

# Check mamba
if command -v mamba &> /dev/null; then
    log_success "✓ Miniforge/Mamba: $(mamba --version | head -1)"
else
    log_error "✗ Miniforge not found"
fi

# Check PyTorch
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)')
    log_success "✓ PyTorch: $TORCH_VERSION"
else
    log_error "✗ PyTorch not found"
fi

# Check CUDA in PyTorch
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VERSION=$(python -c 'import torch; print(torch.version.cuda)')
    log_success "✓ PyTorch CUDA support: $CUDA_VERSION"
else
    log_error "✗ PyTorch CUDA support: Disabled"
fi

# Check compute capability
COMPUTE_CAP=$(python -c "import torch; cap = torch.cuda.get_device_capability(0); print(f'sm_{cap[0]}{cap[1]}')" 2>/dev/null || echo "unknown")
if [[ "$COMPUTE_CAP" == "sm_120" ]]; then
    log_success "✓ Blackwell GPU support: Confirmed ($COMPUTE_CAP)"
else
    log_info "✓ GPU compute capability: $COMPUTE_CAP"
fi

# Check transformers
if python -c "import transformers" 2>/dev/null; then
    TRANS_VERSION=$(python -c 'import transformers; print(transformers.__version__)')
    log_success "✓ Transformers: $TRANS_VERSION"
else
    log_error "✗ Transformers not found"
fi

# Check vLLM
if python -c "import vllm" 2>/dev/null; then
    log_success "✓ vLLM: Installed"
else
    log_warning "✗ vLLM: Not found (optional)"
fi

# Check directories
if [ -d "/scratch/cache" ]; then
    log_success "✓ Storage: /scratch configured"
else
    log_error "✗ Storage: /scratch not found"
fi

echo ""

################################################################################
# COMPLETION
################################################################################

log_success "=========================================="
log_success "  INSTALLATION COMPLETED SUCCESSFULLY!"
log_success "=========================================="
echo ""

log_info "Installation Summary:"
echo "  - GCC 12 installed and configured"
echo "  - CUDA Toolkit 12.4 installed"
echo "  - PyTorch 2.10 nightly with CUDA 12.8"
echo "  - Blackwell GPU support enabled"
echo "  - HuggingFace transformers + vLLM"
echo "  - SDPA attention (FlashAttention-2 skipped)"
echo ""

log_info "Next steps:"
echo ""
echo "1. Close and reopen your terminal (to load environment variables)"
echo ""
echo "2. Test your installation:"
echo "   mamba activate llm-inference"
echo "   cd ~/llm-workspace"
echo "   python test_inference.py"
echo ""
echo "3. Use the startup script for daily use:"
echo "   ~/start-llm-inference.sh"
echo ""

log_info "Quick reference:"
echo "  - Activate environment: mamba activate llm-inference"
echo "  - Run inference: python daily_inference.py --prompt 'Your question'"
echo "  - Monitor GPU: nvitop"
echo ""

log_warning "Important notes:"
echo "  - Using PyTorch NIGHTLY for Blackwell GPU support"
echo "  - FlashAttention-2 NOT installed (ABI incompatibility)"
echo "  - Using SDPA attention instead (still efficient)"
echo "  - First model download will take 10-30 minutes"
echo "  - Models are cached in /scratch/cache"
echo "  - Use --4bit flag for models >40B parameters"
echo ""

log_info "Installation log saved to: ~/llm-core-install.log"

# Save installation info
cat > ~/llm-core-install.log << EOF
LLM Core Installation Log (Blackwell-compatible)
Date: $(date)
User: $USER
Hostname: $(hostname)

Installation Summary:
- GCC: $(gcc --version 2>/dev/null | head -1 || echo "Not found")
- CUDA: $(nvcc --version 2>/dev/null | grep release || echo "Not found")
- Python: $(python --version 2>/dev/null || echo "Not found")
- PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "Not found")
- CUDA in PyTorch: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo "Not found")
- Compute capability: $(python -c 'import torch; cap = torch.cuda.get_device_capability(0); print(f"sm_{cap[0]}{cap[1]}")' 2>/dev/null || echo "Not found")
- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)

Directories:
- Workspace: ~/llm-workspace
- Cache: /scratch/cache
- Models: /scratch/models

Scripts:
- Startup: ~/start-llm-inference.sh
- Daily inference: ~/llm-workspace/daily_inference.py
- Test: ~/llm-workspace/test_inference.py
- vLLM: ~/llm-workspace/vllm_inference.py

Notes:
- PyTorch NIGHTLY used for Blackwell GPU support
- FlashAttention-2 NOT installed (ABI incompatibility with nightly)
- Using SDPA attention instead
- All scripts configured to use SDPA
EOF

log_success "Setup complete! Your Blackwell GPU is ready for LLM inference!"
echo ""
