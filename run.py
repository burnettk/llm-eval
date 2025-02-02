from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import json
from pathlib import Path
import argparse
import importlib.util
import sys

# Add directory to Python path
sys.path.append(str(Path(__file__).parent))

# Initialize the LLM using OpenRouter
api_key = os.environ['OPENROUTER_API_KEY']

def get_model(model_name):
    """Get the appropriate model configuration"""
    if model_name == "mythomax":
        return ChatOpenAI(
            model_name="gryphe/mythomax-l2-13b",
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    elif model_name == "deepseek":
        return ChatOpenAI(
            model_name="deepseek/deepseek-r1-distill-qwen-32b",
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def run_eval(eval_dir, llm):
    """Run evaluation for a specific directory"""
    eval_path = Path(f"evals/{eval_dir}")
    
    # Load prompt configuration
    with open(eval_path / "prompt.txt") as f:
        prompt_config = f.read().strip().splitlines()
    
    # Handle both direct prompts and XML prompts
    if prompt_config[0].startswith("prompt_path:"):
        # Load XML prompt
        prompt_path = Path(prompt_config[0].split(": ")[1])
        operation = prompt_config[1].split(": ")[1]
        
        from xml.etree import ElementTree as ET
        xml_prompt = ET.parse(prompt_path)
        instruction = xml_prompt.find(f".//{operation}/instruction").text
        template = xml_prompt.find(f".//{operation}/template").text
        
        # Replace placeholder in XML with the operation
        prompt_template = xml_prompt.replace("{operation}", operation)
    else:
        prompt_template = "\n".join(prompt_config)

    # Load placeholders
    with open(eval_path / "placeholders.json") as f:
        placeholders = json.load(f)
    
    # Load expected output
    with open(eval_path / "expected.txt") as f:
        expected_output = f.read().strip()
    
    # Run the test
    prompt = ChatPromptTemplate.from_template(prompt_template)
    response = llm.invoke(prompt.format_messages(**placeholders))
    
    assert expected_output in response.content, (
        f"Expected '{expected_output}' to be in '{response.content}', but it wasn't"
    )
    print(f"✅ {eval_dir} passed")

def execute_script(script_path):
    """Execute a script and return the module"""
    spec = importlib.util.spec_from_file_location("script_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run evaluations with different models')
    parser.add_argument('--model', type=str, choices=['mythomax', 'deepseek'], 
                      default='mythomax', help='Model to use for evaluation')
    parser.add_argument('--evals', nargs='*', default=[], 
                      help='Specific evaluations to run (leave empty to run all)')
    args = parser.parse_args()
    
    # Initialize the LLM based on selected model
    llm = get_model(args.model)
    
    # Determine which evaluations to run
    if args.evals:
        eval_dirs = args.evals
        print(f"Running specific evaluations: {', '.join(eval_dirs)}")
    else:
        eval_dirs = ["math", "translation", "script_edit_append", "script_edit_overwrite", "script_edit_modify"]
        print("Running all evaluations")
    
    # Run the specified evaluations
    for eval_dir in eval_dirs:
        try:
            run_eval(eval_dir, llm)
        except AssertionError as e:
            print(f"❌ {eval_dir} failed: {str(e)}")
