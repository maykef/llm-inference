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

class DockerInterpreter:
    def __init__(self, model_id, load_in_4bit=False, load_in_8bit=False, repo_path=None, 
                 use_docker=False, docker_image="python:3.10", extra_packages=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        
        # Paths
        self.repo_path = os.path.abspath(repo_path) if repo_path else None
        self.repo_context = ""
        
        # Docker Config
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.extra_packages = extra_packages  # List of strings: ["numpy", "scipy"]

        if self.repo_path:
            self.repo_context = self.load_repo_context(self.repo_path)
            print(f"--- Loaded {len(self.repo_context)} characters from {self.repo_path} ---")

        print(f"--- Loading model: {self.model_id} ---")
        
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def load_repo_context(self, folder_path):
        if not os.path.exists(folder_path):
            return ""
        context_parts = ["### REPOSITORY CONTEXT ###\n"]
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
                            if len(content) > 20000: 
                                content = content[:20000] + "\n...[TRUNCATED]..."
                            context_parts.append(f"--- FILE: {rel_path} ---\n{content}\n")
                    except Exception:
                        pass
        return "\n".join(context_parts)

    def extract_python_code(self, text):
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else None

    def execute_code(self, code):
        """Runs the code in Docker with optional package installation."""
        
        # Inject pip install at the top of the script if packages are requested
        # We perform a check first to avoid reinstalling if it's already there (optimizes slightly)
        if self.extra_packages:
            pkgs = ' '.join(self.extra_packages)
            install_cmd = (
                f"import subprocess, sys\n"
                f"subprocess.run([sys.executable, '-m', 'pip', 'install', '{pkgs}'], check=True)\n"
            )
            code = install_cmd + "\n" + code

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            if self.use_docker:
                print(f"\033[1;33m--- Executing in Docker ({self.docker_image}) ---\033[0m")
                cmd = [
                    "docker", "run", "--rm",
                    "--gpus", "all",
                    "-v", f"{self.repo_path}:/workspace",
                    "-v", f"{tmp_path}:/script.py",
                    "-w", "/workspace",
                    self.docker_image,
                    "python", "/script.py"
                ]
            else:
                print("\033[1;33m--- Executing Locally ---\033[0m")
                cmd = [sys.executable, tmp_path]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            if not output and not result.stderr:
                output = "[Process finished with no output]"
                
        except subprocess.TimeoutExpired:
            output = "Execution timed out after 120 seconds."
        except Exception as e:
            output = f"Execution failed: {str(e)}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        return output

    def generate_response(self, messages, max_new_tokens=2048):
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def run_interactive_chat(self):
        print("\n" + "="*50)
        mode_str = f"DOCKER MODE ({self.docker_image})" if self.use_docker else "LOCAL MODE"
        print(f"INTERACTIVE INTERPRETER - {mode_str}")
        if self.extra_packages:
            print(f"Auto-Installing: {', '.join(self.extra_packages)}")

        system_prompt = (
            "You are a coding assistant. "
            "You can execute Python code to verify your answers. "
            "Wrap code in ```python ... ``` blocks. "
            "The repository is mounted at /workspace."
        )
        
        if self.repo_context:
            system_prompt += f"\n\nCodebase Context:\n{self.repo_context}"

        messages = [{"role": "user", "content": system_prompt}]
        messages.append({"role": "assistant", "content": "Ready. I can analyze the scripts and run tests."})

        while True:
            try:
                user_input = input("\033[1;32mUser:\033[0m ") 
                if user_input.lower() in ["exit", "quit"]: break
                messages.append({"role": "user", "content": user_input})
                
                while True:
                    print("\033[1;34mModel:\033[0m Generating...", end="\r")
                    response = self.generate_response(messages)
                    print(f"\033[1;34mModel:\033[0m {response}\n")
                    messages.append({"role": "assistant", "content": response})

                    code_to_run = self.extract_python_code(response)
                    if code_to_run:
                        print("\033[1;33m[Detected Python Code Block]\033[0m")
                        if input("\033[1;31m>>> Execute? (y/n):\033[0m ").lower().startswith('y'):
                            output = self.execute_code(code_to_run)
                            print(f"\033[1;30mOutput:\n{output}\033[0m\n")
                            messages.append({"role": "user", "content": f"Output:\n{output}\nInterpret this."})
                            continue 
                        else:
                            messages.append({"role": "user", "content": "Skipped execution."})
                            continue
                    else:
                        break
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--4bit", action="store_true")
    parser.add_argument("--8bit", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--image", type=str, default="python:3.10")
    parser.add_argument("--extra_packages", type=str, nargs="+", help="List of packages to install (e.g. numpy dadi)")
    
    args = parser.parse_args()

    inference = DockerInterpreter(
        args.model, 
        load_in_4bit=args.__dict__["4bit"], 
        load_in_8bit=args.__dict__["8bit"], 
        repo_path=args.repo_path,
        use_docker=args.docker,
        docker_image=args.image,
        extra_packages=args.extra_packages
    )

    if args.interactive:
        inference.run_interactive_chat()
