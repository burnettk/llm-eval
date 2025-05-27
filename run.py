from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import yaml
import json
from pathlib import Path
import argparse
import importlib.util
import sys
import logging

# Get a logger instance for this module
logger = logging.getLogger(__name__)

def get_placeholders_from_template(template):
    """Extract placeholders from a prompt template string"""
    placeholders = set()
    # Split on curly braces and look for variable names
    parts = template.split('{')
    for part in parts:
        if '}' in part:
            placeholder = part.split('}')[0].strip()
            if placeholder:
                placeholders.add(placeholder)
    return placeholders

def run_eval(eval_dir, llm):
    """Run evaluation for a specific directory"""
    eval_path = Path(f"evals/{eval_dir}")
    
    # Load prompt configuration
    with open(eval_path / "prompt.yaml") as f:
        prompt_config = yaml.safe_load(f)
    
    # Extract placeholders from prompt.yaml
    placeholders = {key: value for key, value in prompt_config.items() if key != "expected_output" and key != "prompt_path" and key != "prompt_template"}
    
    # Get prompt template from either prompt_path or prompt_template
    if "prompt_path" in prompt_config:
        template_path = eval_path / prompt_config["prompt_path"]
        with open(template_path) as f:
            prompt_template = f.read().strip()
    elif "prompt_template" in prompt_config:
        prompt_template = prompt_config["prompt_template"]
    else:
        raise ValueError(f"Neither prompt_path nor prompt_template found in {eval_dir}/prompt.yaml")
    
    # Validate placeholders
    expected_placeholders = get_placeholders_from_template(prompt_template)
    missing = expected_placeholders - set(placeholders.keys())
    if missing:
        raise ValueError(f"Missing required placeholders in {eval_dir}: {', '.join(missing)}")
    
    # Run the test
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_with_filled_placeholders = prompt.format_messages(**placeholders)
    logger.debug(f"Prompt with filled placeholders: {prompt_with_filled_placeholders}")
    response = llm.invoke(prompt_with_filled_placeholders)
    
    # Strip whitespace before comparison
    expected_output = prompt_config["expected_output"].strip()
    response_content = response.content.strip()
    
    assert expected_output in response_content, (
        f"Expected '{expected_output}' to be in '{response_content}', but it wasn't"
    )
    logger.info(f"✅ {eval_dir} passed")

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
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (e.g., DEBUG, INFO)')
    args = parser.parse_args()
    
    # Set the logging level based on the argument
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logging.basicConfig(level=log_level_map[args.log_level.upper()], format='%(levelname)s: %(message)s')

    # Initialize the LLM based on selected model
    llm = get_model(args.model)
    
    # Determine which evaluations to run
    if args.evals:
        eval_dirs = args.evals
        logger.info(f"Running specific evaluations: {', '.join(eval_dirs)}")
    else:
        eval_dirs = ["math", "translation", "script_edit_append", "script_edit_overwrite", "script_edit_modify"]
        logger.info("Running all evaluations")
    
    # Run the specified evaluations
    for eval_dir in eval_dirs:
        try:
            run_eval(eval_dir, llm)
        except AssertionError as e:
            logger.error(f"❌ {eval_dir} failed: {str(e)}")

