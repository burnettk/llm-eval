import litellm # Import the main litellm module
from litellm import completion as litellm_completion
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

    # Format the prompt content
    formatted_content = prompt_template.format(**placeholders)
    
    # Prepare messages for LiteLLM
    # Assuming all prompts are single user messages based on current structure
    llm_messages = [{"role": "user", "content": formatted_content}]
    logger.debug(f"Messages for LiteLLM: {llm_messages}")

    # Run the test using LiteLLM
    # The 'llm' variable (model_name string) is passed to LiteLLM.
    response = litellm_completion(model=llm, messages=llm_messages)
    
    # Strip whitespace before comparison
    expected_output = prompt_config["expected_output"].strip()
    # LiteLLM response structure: response.choices[0].message.content
    response_content = response.choices[0].message.content.strip()
    
    assert expected_output in response_content, (
        f"Expected '{expected_output}' to be in '{response_content}', but it wasn't"
    )
    logger.info(f"✅ {eval_dir} passed")

# Add directory to Python path
sys.path.append(str(Path(__file__).parent))

# Note: OPENROUTER_API_KEY should be set in the environment for LiteLLM to use OpenRouter.

def get_model(model_name):
    """Get the appropriate model name for LiteLLM, ensuring OpenRouter prefix."""
    # OPENROUTER_API_KEY environment variable is used by LiteLLM for OpenRouter.
    if model_name == "mythomax":
        # Specific alias mapping
        return "openrouter/gryphe/mythomax-l2-13b"
    elif model_name == "deepseek":
        # Specific alias mapping
        return "openrouter/deepseek/deepseek-r1-distill-qwen-32b"
    else:
        # For other models, ensure they are prefixed for OpenRouter
        return model_name

def execute_script(script_path):
    """Execute a script and return the module"""
    spec = importlib.util.spec_from_file_location("script_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run evaluations with different models')
    parser.add_argument('--model', type=str,
                      default='gemini/gemini-1.5-flash-8b-exp-0924', help='Model to use for evaluation')
    parser.add_argument('--evals', nargs='*', default=[], 
                      help='Specific evaluations to run (leave empty to run all)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (e.g., DEBUG, INFO)')
    args = parser.parse_args()
    
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logging.basicConfig(level=log_level_map[args.log_level.upper()], format='%(name)s:%(levelname)s: %(message)s')
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    llm = get_model(args.model)
    
    if args.evals:
        eval_dirs = args.evals
        logger.info(f"Running specific evaluations: {', '.join(eval_dirs)}")
    else:
        eval_dirs = ["math", "translation", "script_edit_append", "script_edit_overwrite", "script_edit_modify"]
        logger.info("Running all evaluations")
    
    for eval_dir in eval_dirs:
        try:
            run_eval(eval_dir, llm)
        except AssertionError as e:
            logger.error(f"❌ {eval_dir} failed: {str(e)}")

