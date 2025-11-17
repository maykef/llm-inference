#!/usr/bin/env python3
"""
test_inference.py - Blackwell-compatible LLM inference test script

This script tests the basic inference setup with a Mistral-7B model using:
- PyTorch with CUDA support
- SDPA (Scaled Dot Product Attention) instead of FlashAttention
- BFloat16 precision for optimal performance on modern GPUs
- Automatic device mapping for multi-GPU setups

Designed for: RTX Pro 6000 Blackwell (96GB VRAM) with PyTorch nightly
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import time
from typing import Optional


def print_system_info():
    """Display system and GPU information."""
    print("=" * 70)
    print("System Information")
    print("=" * 70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            compute_cap = torch.cuda.get_device_capability(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: sm_{compute_cap[0]}{compute_cap[1]}")
            print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Available memory: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1e9:.1f} GB")
            
            # Check for Blackwell support
            if compute_cap[0] >= 12:
                print(f"  ✓ Blackwell GPU support confirmed!")
    else:
        print("ERROR: CUDA is not available! Please check your installation.")
        sys.exit(1)
    
    print("=" * 70)
    print()


def load_model(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Load the model and tokenizer with Blackwell-optimized settings.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run as the model downloads...")
    
    start_time = time.time()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present (required for batch processing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimized settings for Blackwell
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # BF16 for better numerical stability
        device_map="auto",  # Automatic device placement
        attn_implementation="sdpa",  # SDPA instead of FlashAttention
        trust_remote_code=True,  # Allow custom model code if needed
    )
    
    load_time = time.time() - start_time
    print(f"✓ Model loaded successfully in {load_time:.1f} seconds")
    print(f"  Attention implementation: SDPA (Scaled Dot Product Attention)")
    print(f"  Precision: bfloat16")
    print(f"  Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
    print()
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
):
    """
    Generate a response for the given prompt.
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling (vs greedy decoding)
    
    Returns:
        Generated text response
    """
    print(f"Prompt: {prompt}\n")
    print("Generating response...\n")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move to GPU
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Track generation time
    start_time = time.time()
    
    # Generate with specified parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - start_time
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calculate tokens per second
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
    
    print("Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    print()
    print(f"Generation stats:")
    print(f"  Tokens generated: {num_tokens}")
    print(f"  Time: {generation_time:.2f} seconds")
    print(f"  Speed: {tokens_per_second:.1f} tokens/second")
    
    return response


def print_memory_usage():
    """Display current GPU memory usage."""
    if torch.cuda.is_available():
        print("\nMemory usage:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}:")
            print(f"    Allocated: {allocated:.2f} GB")
            print(f"    Reserved: {reserved:.2f} GB")
            print(f"    Total: {total:.2f} GB")
            print(f"    Free: {total - reserved:.2f} GB")


def run_test_suite():
    """Run a comprehensive test of the inference setup."""
    test_prompts = [
        "Explain quantum entanglement in simple terms:",
        "Write a haiku about artificial intelligence:",
        "What are the key differences between supervised and unsupervised learning?",
    ]
    
    print("\n" + "=" * 70)
    print("Running Inference Test Suite")
    print("=" * 70 + "\n")
    
    # Load model once
    model, tokenizer = load_model()
    
    # Test each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}/{len(test_prompts)}")
        print("=" * 50)
        
        try:
            response = generate_response(
                model, 
                tokenizer, 
                prompt,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9
            )
            print("✓ Test passed")
        except Exception as e:
            print(f"✗ Test failed: {e}")
            continue
        
        if i < len(test_prompts):
            print("\nPress Enter to continue to next test...")
            input()
    
    # Final memory report
    print_memory_usage()
    
    print("\n" + "=" * 70)
    print("Test Suite Completed!")
    print("=" * 70)


def main():
    """Main entry point."""
    try:
        # Print system information
        print_system_info()
        
        # Run basic inference test
        print("Running basic inference test...")
        print("=" * 70 + "\n")
        
        # Load model
        model, tokenizer = load_model()
        
        # Test with a simple prompt
        test_prompt = "Explain the concept of neural networks in one paragraph:"
        response = generate_response(
            model,
            tokenizer,
            test_prompt,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )
        
        # Show memory usage
        print_memory_usage()
        
        print("\n" + "=" * 70)
        print("✓ Inference test completed successfully!")
        print("=" * 70)
        print("\nYour Blackwell GPU setup is working correctly.")
        print("You can now run more complex inference tasks.")
        
        # Offer to run full test suite
        print("\nWould you like to run the full test suite? (y/n): ", end="")
        if input().lower() == 'y':
            run_test_suite()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("\nPlease check your installation and ensure:")
        print("1. CUDA is properly installed")
        print("2. PyTorch has CUDA support")
        print("3. You have activated the llm-inference environment")
        print("4. You have sufficient GPU memory")
        sys.exit(1)


if __name__ == "__main__":
    main()
