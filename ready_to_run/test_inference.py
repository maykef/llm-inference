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
