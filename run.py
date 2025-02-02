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
    
    # Load prompt template
    with open(eval_path / "prompt.txt") as f:
        prompt_template = f.read().strip()
    
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
    args = parser.parse_args()
    
    # Initialize the LLM based on selected model
    llm = get_model(args.model)
    
    # Run all evaluations
    for eval_dir in ["math", "translation", "script_edit_append", "script_edit_overwrite", "script_edit_modify"]:
        try:
            run_eval(eval_dir, llm)
        except AssertionError as e:
            print(f"❌ {eval_dir} failed: {str(e)}")
