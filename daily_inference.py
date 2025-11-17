#!/usr/bin/env python3
"""
daily_inference.py - Production-ready LLM inference with Blackwell GPU support

A flexible inference script supporting:
- Multiple model architectures (Llama, Mistral, Mixtral, etc.)
- 4-bit quantization via bitsandbytes for memory efficiency
- Interactive and batch modes
- SDPA attention (FlashAttention-free for compatibility)
- System prompt customization
- Conversation history management

Optimized for: RTX Pro 6000 Blackwell (96GB VRAM)
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TextStreamer
)
import argparse
import json
import sys
import os
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ModelSize(Enum):
    """Common model size categories."""
    SMALL = "7B"     # ~13-14GB in bf16
    MEDIUM = "13B"   # ~26-28GB in bf16
    LARGE = "30B"    # ~60-65GB in bf16
    XLARGE = "70B"   # ~140GB in bf16 (needs multi-GPU or quantization)


@dataclass
class ModelConfig:
    """Configuration for different models."""
    name: str
    size: ModelSize
    default_system_prompt: str = ""
    chat_template: bool = True
    trust_remote_code: bool = False


# Predefined model configurations
MODEL_REGISTRY = {
    "mistral-7b": ModelConfig(
        name="mistralai/Mistral-7B-Instruct-v0.2",
        size=ModelSize.SMALL,
        default_system_prompt="You are a helpful AI assistant.",
        chat_template=True
    ),
    "mixtral-8x7b": ModelConfig(
        name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        size=ModelSize.LARGE,
        default_system_prompt="You are a helpful AI assistant.",
        chat_template=True
    ),
    "llama2-7b": ModelConfig(
        name="meta-llama/Llama-2-7b-chat-hf",
        size=ModelSize.SMALL,
        default_system_prompt="You are a helpful, respectful and honest assistant.",
        chat_template=True
    ),
    "llama2-13b": ModelConfig(
        name="meta-llama/Llama-2-13b-chat-hf",
        size=ModelSize.MEDIUM,
        default_system_prompt="You are a helpful, respectful and honest assistant.",
        chat_template=True
    ),
    "llama2-70b": ModelConfig(
        name="meta-llama/Llama-2-70b-chat-hf",
        size=ModelSize.XLARGE,
        default_system_prompt="You are a helpful, respectful and honest assistant.",
        chat_template=True
    ),
    "codellama-7b": ModelConfig(
        name="codellama/CodeLlama-7b-Instruct-hf",
        size=ModelSize.SMALL,
        default_system_prompt="You are an expert programming assistant.",
        chat_template=True
    ),
    "phi-2": ModelConfig(
        name="microsoft/phi-2",
        size=ModelSize.SMALL,
        default_system_prompt="",
        chat_template=False,
        trust_remote_code=True
    ),
}


class LLMInference:
    """
    Main inference class with support for various models and configurations.
    """
    
    def __init__(
        self,
        model_name: str,
        use_4bit: bool = False,
        use_8bit: bool = False,
        max_memory: Optional[Dict] = None,
        device_map: str = "auto",
        load_in_4bit_params: Optional[Dict] = None
    ):
        """
        Initialize the LLM inference engine.
        
        Args:
            model_name: HuggingFace model ID or local path
            use_4bit: Enable 4-bit quantization
            use_8bit: Enable 8-bit quantization
            max_memory: Memory allocation per device
            device_map: Device placement strategy
            load_in_4bit_params: Custom 4-bit loading parameters
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        
        print(f"\n{'='*70}")
        print(f"Initializing LLM Inference Engine")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                compute_cap = torch.cuda.get_device_capability(i)
                print(f"GPU {i}: {torch.cuda.get_device_name(i)} (sm_{compute_cap[0]}{compute_cap[1]})")
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model loading arguments
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": True,
            "attn_implementation": "sdpa",  # Use SDPA for Blackwell compatibility
        }
        
        # Configure quantization
        if use_4bit:
            print("Configuring 4-bit quantization...")
            if load_in_4bit_params:
                bnb_config = BitsAndBytesConfig(**load_in_4bit_params)
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            model_kwargs["quantization_config"] = bnb_config
            print("  Quantization: 4-bit NF4")
            print("  Compute dtype: bfloat16")
            print("  Double quantization: enabled")
        elif use_8bit:
            print("Configuring 8-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = bnb_config
            print("  Quantization: 8-bit")
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
            print("Using bfloat16 precision (no quantization)")
        
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        
        # Load model
        print("\nLoading model (this may take a few minutes)...")
        start_time = time.time()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.1f} seconds")
        
        # Print memory usage
        self._print_memory_usage()
        
        # Store device for later use
        self.device = next(self.model.parameters()).device
        
        print(f"\n✓ Inference engine ready!")
        print(f"{'='*70}\n")
    
    def _print_memory_usage(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            print("\nMemory usage:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                
                print(f"  GPU {i}:")
                print(f"    Model: {allocated:.2f} GB")
                print(f"    Reserved: {reserved:.2f} GB")
                print(f"    Available: {total - reserved:.2f} GB / {total:.2f} GB")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stream: bool = False,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response for a given prompt.
        
        Args:
            prompt: User input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Use sampling vs greedy decoding
            stream: Stream output token by token
            system_prompt: System prompt to prepend
        
        Returns:
            Generated text response
        """
        # Format prompt with system message if provided
        if system_prompt and hasattr(self.tokenizer, 'chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Configure generation
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Stream or generate
        if stream:
            streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            gen_kwargs["streamer"] = streamer
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """
        Chat with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Stream output
        
        Returns:
            Generated response
        """
        if hasattr(self.tokenizer, 'chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Simple formatting for models without chat template
            prompt = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in messages
            ]) + "\nassistant: "
        
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=stream
        )
    
    def benchmark(self, prompt: str = "Hello, how are you?", num_runs: int = 3) -> Dict:
        """
        Benchmark inference performance.
        
        Args:
            prompt: Test prompt
            num_runs: Number of benchmark runs
        
        Returns:
            Benchmark statistics
        """
        print(f"\nRunning benchmark ({num_runs} runs)...")
        
        times = []
        tokens_generated = []
        
        for i in range(num_runs):
            start = time.time()
            response = self.generate(prompt, max_new_tokens=100, do_sample=False)
            elapsed = time.time() - start
            
            num_tokens = len(self.tokenizer.tokenize(response))
            times.append(elapsed)
            tokens_generated.append(num_tokens)
            
            print(f"  Run {i+1}: {elapsed:.2f}s ({num_tokens} tokens)")
        
        avg_time = sum(times) / len(times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens / avg_time
        
        stats = {
            "average_time": avg_time,
            "average_tokens": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "all_times": times,
            "all_tokens": tokens_generated
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Average tokens: {avg_tokens:.1f}")
        print(f"  Speed: {tokens_per_second:.1f} tokens/s")
        
        return stats


def interactive_mode(llm: LLMInference, system_prompt: Optional[str] = None):
    """
    Run interactive chat mode.
    
    Args:
        llm: LLMInference instance
        system_prompt: Optional system prompt
    """
    print("\n" + "="*70)
    print("Interactive Mode")
    print("="*70)
    print("Commands:")
    print("  /quit or /exit - Exit the program")
    print("  /clear - Clear conversation history")
    print("  /save <filename> - Save conversation")
    print("  /system <prompt> - Change system prompt")
    print("  /help - Show this help")
    print("="*70 + "\n")
    
    messages = []
    if system_prompt:
        print(f"System: {system_prompt}\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command = user_input.split()[0].lower()
                
                if command in ["/quit", "/exit"]:
                    print("Goodbye!")
                    break
                
                elif command == "/clear":
                    messages = []
                    print("Conversation cleared.")
                    continue
                
                elif command == "/save":
                    try:
                        filename = user_input.split(maxsplit=1)[1]
                        with open(filename, "w") as f:
                            json.dump(messages, f, indent=2)
                        print(f"Conversation saved to {filename}")
                    except (IndexError, IOError) as e:
                        print(f"Error saving: {e}")
                    continue
                
                elif command == "/system":
                    try:
                        system_prompt = user_input.split(maxsplit=1)[1]
                        print(f"System prompt updated: {system_prompt}")
                    except IndexError:
                        print("Usage: /system <prompt>")
                    continue
                
                elif command == "/help":
                    print("\nCommands:")
                    print("  /quit or /exit - Exit")
                    print("  /clear - Clear history")
                    print("  /save <file> - Save conversation")
                    print("  /system <prompt> - Set system prompt")
                    print("  /help - Show help")
                    continue
                
                else:
                    print(f"Unknown command: {command}")
                    continue
            
            # Prepare messages
            current_messages = messages.copy()
            if system_prompt:
                current_messages.insert(0, {"role": "system", "content": system_prompt})
            current_messages.append({"role": "user", "content": user_input})
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            if len(current_messages) == 1 or (len(current_messages) == 2 and system_prompt):
                # First message, use simple generation
                response = llm.generate(
                    prompt=user_input,
                    system_prompt=system_prompt,
                    stream=True,
                    temperature=0.7
                )
            else:
                # Use chat with history
                response = llm.chat(
                    messages=current_messages,
                    stream=True,
                    temperature=0.7
                )
            
            # Update conversation history
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\nUse /quit to exit properly.")
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="LLM Inference Tool - Optimized for Blackwell GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python daily_inference.py --prompt "Explain quantum computing"
  
  # Use a specific model
  python daily_inference.py --model mistralai/Mistral-7B-Instruct-v0.2 --prompt "Hello"
  
  # Use 4-bit quantization for large models
  python daily_inference.py --4bit --model meta-llama/Llama-2-70b-chat-hf
  
  # Interactive mode
  python daily_inference.py --interactive
  
  # Benchmark performance
  python daily_inference.py --benchmark
  
  # Use a preset model
  python daily_inference.py --preset llama2-13b --prompt "Write a poem"
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model ID or path"
    )
    parser.add_argument(
        "--preset",
        choices=list(MODEL_REGISTRY.keys()),
        help="Use a preset model configuration"
    )
    
    # Quantization options
    parser.add_argument(
        "--4bit",
        dest="use_4bit",
        action="store_true",
        help="Use 4-bit quantization (reduces memory usage)"
    )
    parser.add_argument(
        "--8bit",
        dest="use_8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )
    
    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="System prompt to set context"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output token by token"
    )
    
    # Modes
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    
    # Memory management
    parser.add_argument(
        "--max-memory",
        type=str,
        help="Max memory per GPU in GB (e.g., '0:80GB,1:80GB')"
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device placement strategy (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Handle preset
    if args.preset:
        preset = MODEL_REGISTRY[args.preset]
        args.model = preset.name
        if not args.system_prompt:
            args.system_prompt = preset.default_system_prompt
    
    # Parse max memory if provided
    max_memory = None
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory.split(","):
            device, memory = item.split(":")
            max_memory[int(device)] = memory
    
    # Check for conflicting quantization options
    if args.use_4bit and args.use_8bit:
        print("Error: Cannot use both 4-bit and 8-bit quantization")
        sys.exit(1)
    
    try:
        # Initialize model
        llm = LLMInference(
            model_name=args.model,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            max_memory=max_memory,
            device_map=args.device_map
        )
        
        # Run appropriate mode
        if args.interactive:
            interactive_mode(llm, args.system_prompt)
        
        elif args.benchmark:
            test_prompt = args.prompt or "Explain the theory of relativity in simple terms."
            llm.benchmark(prompt=test_prompt)
        
        elif args.prompt:
            print(f"\nPrompt: {args.prompt}\n")
            print("Response:\n" + "="*50)
            
            response = llm.generate(
                prompt=args.prompt,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=not args.no_sample,
                stream=args.stream
            )
            
            if not args.stream:
                print(response)
            
            print("="*50)
            
        else:
            print("\nNo action specified. Use one of:")
            print("  --prompt <text>  : Generate response")
            print("  --interactive    : Start chat mode")
            print("  --benchmark      : Run benchmark")
            print("\nUse --help for more options")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
