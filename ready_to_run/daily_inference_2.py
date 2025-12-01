import torch
import argparse
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Extensions to read when scanning a repo
CODE_EXTENSIONS = {'.py', '.sh', '.md', '.json', '.js', '.ts', '.c', '.cpp', '.h', '.java', '.go', '.rs', '.yml', '.yaml'}
# Folders to ignore
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', 'venv', 'env', '.idea', '.vscode', 'build', 'dist', 'target'}

class DailyInference:
    def __init__(self, model_id, load_in_4bit=False, load_in_8bit=False, repo_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.repo_context = ""

        # Pre-load repo content if provided
        if repo_path:
            self.repo_context = self.load_repo_context(repo_path)
            print(f"--- Loaded {len(self.repo_context)} characters from repository at {repo_path} ---")

        print(f"--- Loading model: {self.model_id} ---")
        
        # Configure Quantization
        bnb_config = None
        if load_in_4bit:
            print("--- 4-bit Quantization Enabled ---")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            print("--- 8-bit Quantization Enabled ---")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        # Fix padding token if missing (common in Llama/Mistral)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # AWQ-aware Model Loading
        if "AWQ" in self.model_id.upper():
            try:
                print("--- Detected AWQ model: Using AutoAWQForCausalLM ---")
                from awq import AutoAWQForCausalLM
                self.model = AutoAWQForCausalLM.from_pretrained(
                    self.model_id,
                    fuse_layers=True,  # Optional: Fuses QKV for slight speed-up
                    trust_remote_code=True,
                    safetensors=True,
                    device_map="auto",
                    attn_implementation="sdpa"  # If your Transformers supports it
                )
                # AWQ uses bfloat16 by default; no explicit dtype needed
                # Ignore bnb_config for AWQ (it's pre-quantized)
            except ImportError:
                raise ImportError("AWQ model detected but 'autoawq' not installed. Run: pip install autoawq")
        else:
            # Original BitsAndBytes path for non-AWQ models
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa"
            )
        
        # Sync pad token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def load_repo_context(self, folder_path):
        """Walks the folder and reads all code files into a single string."""
        if not os.path.exists(folder_path):
            print(f"Warning: Repo path {folder_path} does not exist.")
            return ""

        context_parts = ["### REPOSITORY CONTEXT ###\n"]
        file_count = 0
        
        for root, dirs, files in os.walk(folder_path):
            # Modify dirs in-place to skip ignored folders
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in CODE_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, folder_path)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                            content = f.read()
                            # Limit very large files to avoid choking the context
                            if len(content) > 100000: 
                                content = content[:100000] + "\n...[TRUNCATED]..."
                            context_parts.append(f"--- FILE: {rel_path} ---\n{content}\n")
                            file_count += 1
                    except Exception as e:
                        print(f"Skipping file {rel_path}: {e}")
        
        print(f"--- Scanned {file_count} files ---")
        return "\n".join(context_parts)

    def generate_response(self, messages, max_new_tokens=2048, temperature=0.7):
        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def run_interactive_chat(self, temperature=0.7, max_tokens=2048):
        print("\n" + "="*50)
        print(f"INTERACTIVE CHAT MODE (Temp: {temperature})")
        
        if self.repo_context:
            print(f"\033[1;33m(Repository Context Loaded: {len(self.repo_context)} chars)\033[0m")
            # Inject context as the first message
            system_msg = {
                "role": "user", 
                "content": f"Here is the codebase I am working on:\n\n{self.repo_context}\n\nPlease use this context to answer my questions."
            }
            # Start history with context
            messages = [system_msg]
            # Add a dummy assistant acknowledgement to keep the chat flow natural
            messages.append({"role": "assistant", "content": "I have read the codebase. How can I help you?"})
        else:
            messages = []

        print("Type 'exit' or 'quit' to stop.")
        print("="*50 + "\n")

        while True:
            try:
                user_input = input("\033[1;32mUser:\033[0m ") 
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting...")
                    break
                
                messages.append({"role": "user", "content": user_input})

                print("\033[1;34mModel:\033[0m Generating...", end="\r")
                
                response = self.generate_response(messages, max_new_tokens=max_tokens, temperature=temperature)
                
                print(f"\033[1;34mModel:\033[0m {response}\n")

                messages.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                print("\nInterrupted. Exiting...")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--prompt", type=str, default="Tell me a joke.", help="Ignored in interactive mode")
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization (Recommended for 70B)")
    parser.add_argument("--8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--interactive", action="store_true", help="Enable chat mode")
    parser.add_argument("--repo_path", type=str, help="Path to local folder to read code from")
    parser.add_argument("--temperature", type=float, default=0.7, help="Creativity (0.0 - 1.0)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max length of response")

    args = parser.parse_args()

    # Safety check for 70B
    if "70B" in args.model and not (args.__dict__["4bit"] or args.__dict__["8bit"]):
        print("\n⚠️  WARNING: You are loading a 70B model without quantization.")
        print("⚠️  This requires >140GB VRAM. If you have less, add --4bit or --8bit.\n")

    inference = DailyInference(
        args.model, 
        load_in_4bit=args.__dict__["4bit"], 
        load_in_8bit=args.__dict__["8bit"], 
        repo_path=args.repo_path
    )

    if args.interactive:
        inference.run_interactive_chat(temperature=args.temperature, max_tokens=args.max_tokens)
    else:
        # Single prompt mode with context
        final_prompt = args.prompt
        if args.repo_path and inference.repo_context:
            final_prompt = f"Context:\n{inference.repo_context}\n\nQuestion: {args.prompt}"
        
        msgs = [{"role": "user", "content": final_prompt}]
        print(inference.generate_response(msgs, max_new_tokens=args.max_tokens, temperature=args.temperature))
