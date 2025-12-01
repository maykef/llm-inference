import torch
import argparse
import os
import sys
import re
import subprocess
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Extensions to read when scanning a repo
CODE_EXTENSIONS = {'.py', '.sh', '.md', '.json', '.js', '.ts', '.c', '.cpp', '.h', '.java', '.go', '.rs', '.yml', '.yaml'}
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', 'venv', 'env', '.idea', '.vscode', 'build', 'dist', 'target'}

class LocalInterpreter:
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
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def load_repo_context(self, folder_path):
        """Walks the folder and reads all code files into a single string."""
        if not os.path.exists(folder_path):
            print(f"Warning: Repo path {folder_path} does not exist.")
            return ""

        context_parts = ["### REPOSITORY CONTEXT ###\n"]
        file_count = 0
        
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in CODE_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, folder_path)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                            content = f.read()
                            if len(content) > 50000: # Slightly lower limit to save context
                                content = content[:50000] + "\n...[TRUNCATED]..."
                            context_parts.append(f"--- FILE: {rel_path} ---\n{content}\n")
                            file_count += 1
                    except Exception as e:
                        print(f"Skipping file {rel_path}: {e}")
        
        print(f"--- Scanned {file_count} files ---")
        return "\n".join(context_parts)

    def extract_python_code(self, text):
        """Extracts the first code block marked with python."""
        # Regex looks for ```python ... ```
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def execute_code(self, code):
        """Runs the code in a subprocess and returns the output."""
        print("\033[1;33m--- Executing Code in Subprocess ---\033[0m")
        
        # Write to a temp file to ensure it runs exactly as a script would
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
            
        try:
            # Run using the current python executable (inherits environment)
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=60 # 60 second timeout to prevent hangs
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            if not output and not result.stderr:
                output = "[Process finished with no output]"
        except subprocess.TimeoutExpired:
            output = "Execution timed out after 60 seconds."
        except Exception as e:
            output = f"Execution failed: {str(e)}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        return output

    def generate_response(self, messages, max_new_tokens=2048, temperature=0.7):
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
        print(f"INTERACTIVE INTERPRETER MODE (Temp: {temperature})")
        print("\033[1;35mCapabilities: Can write and EXECUTE Python code on this machine.\033[0m")
        
        system_prompt = (
            "You are an expert coding assistant with access to a Python interpreter. "
            "If the user asks a question that requires calculation, inspection, or verifying code, "
            "you should write a Python script to solve it. "
            "Wrap your code in ```python ... ``` blocks. "
            "The user will execute the code and paste the results back to you. "
            "Analyze the results and give a final answer."
        )

        if self.repo_context:
            print(f"\033[1;33m(Repository Context Loaded: {len(self.repo_context)} chars)\033[0m")
            system_prompt += f"\n\nHere is the codebase available to inspect:\n{self.repo_context}"

        messages = [{"role": "user", "content": system_prompt}]
        messages.append({"role": "assistant", "content": "Understood. I am ready to analyze the code and execute Python scripts to verify my findings."})

        print("Type 'exit' to stop.")
        print("="*50 + "\n")

        while True:
            try:
                user_input = input("\033[1;32mUser:\033[0m ") 
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                messages.append({"role": "user", "content": user_input})
                
                # --- The Execution Loop ---
                # We loop here to allow: Model -> Code -> Result -> Model -> Text/More Code
                while True:
                    print("\033[1;34mModel:\033[0m Generating...", end="\r")
                    response = self.generate_response(messages, max_new_tokens=max_tokens, temperature=temperature)
                    print(f"\033[1;34mModel:\033[0m {response}\n")
                    
                    messages.append({"role": "assistant", "content": response})

                    # Check for code block
                    code_to_run = self.extract_python_code(response)
                    
                    if code_to_run:
                        # SECURITY CHECK
                        print("\033[1;33m[Detected Python Code Block]\033[0m")
                        perm = input("\033[1;31m>>> Execute this code on your local machine? (y/n):\033[0m ")
                        
                        if perm.lower().startswith('y'):
                            output = self.execute_code(code_to_run)
                            print(f"\033[1;30mOutput:\n{output}\033[0m\n")
                            
                            # Feed result back to model
                            result_msg = f"Execution Output:\n{output}\n\nPlease interpret this result."
                            messages.append({"role": "user", "content": result_msg})
                            # Loop continues to let model respond to the output
                            continue 
                        else:
                            print("Execution skipped by user.")
                            messages.append({"role": "user", "content": "I chose not to run that code. Please provide an answer without execution or suggest an alternative."})
                            continue
                    else:
                        # No code block, finished turn
                        break

            except KeyboardInterrupt:
                print("\nInterrupted. Exiting...")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--prompt", type=str, default="Tell me a joke.", help="Ignored in interactive mode")
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--interactive", action="store_true", help="Enable chat mode")
    parser.add_argument("--repo_path", type=str, help="Path to local folder to read code from")
    parser.add_argument("--temperature", type=float, default=0.7, help="Creativity")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max length")

    args = parser.parse_args()

    inference = LocalInterpreter(
        args.model, 
        load_in_4bit=args.__dict__["4bit"], 
        load_in_8bit=args.__dict__["8bit"], 
        repo_path=args.repo_path
    )

    if args.interactive:
        inference.run_interactive_chat(temperature=args.temperature, max_tokens=args.max_tokens)
    else:
        # Fallback for single prompt mode
        prompt = args.prompt
        if args.repo_path:
            prompt = f"Context:\n{inference.repo_context}\n\nQuestion: {args.prompt}"
        print(inference.generate_response([{"role": "user", "content": prompt}], max_new_tokens=args.max_tokens))
