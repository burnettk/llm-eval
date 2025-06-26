import litellm
from litellm import completion as litellm_completion
import yaml
from pathlib import Path
import argparse
import importlib.util
import sys
import logging
import re

# Get a logger instance for this module
logger = logging.getLogger(__name__)

def get_placeholders_from_template(template):
    """Extract placeholders from a prompt template string"""
    # Use regex to find all placeholders in the format {placeholder}
    return set(re.findall(r'\{([^}]+)\}', template))

def load_and_validate_prompt(eval_dir, eval_path):
    """Load and validate the prompt configuration and template."""
    # Load prompt configuration
    with open(eval_path / "prompt.yaml") as f:
        prompt_config = yaml.safe_load(f)

    # Define reserved keys that are not placeholders
    reserved_keys = {"expected_output", "prompt_path", "prompt_template"}
    
    # Extract placeholders from prompt.yaml by excluding reserved keys
    placeholders = {key: value for key, value in prompt_config.items() if key not in reserved_keys}
    
    # Get prompt template from either prompt_path or prompt_template
    if "prompt_path" in prompt_config:
        template_path = eval_path / prompt_config["prompt_path"]
        with open(template_path) as f:
            prompt_template = f.read().strip()
    elif "prompt_template" in prompt_config:
        prompt_template = prompt_config["prompt_template"]
    else:
        raise ValueError(f"Neither prompt_path nor prompt_template found in {eval_dir}/prompt.yaml")
    
    # Validate that all placeholders in the template are provided
    expected_placeholders = get_placeholders_from_template(prompt_template)
    provided_placeholders = set(placeholders.keys())
    
    missing = expected_placeholders - provided_placeholders
    if missing:
        raise ValueError(f"Missing required placeholders in {eval_dir}: {', '.join(missing)}")
    
    return prompt_config, prompt_template, placeholders

def run_llm_and_assert(llm, llm_messages, expected_output):
    """Run the LLM and assert the response."""
    response = litellm_completion(model=llm, messages=llm_messages)
    
    # Strip whitespace before comparison
    response_content = response.choices[0].message.content.strip()
    
    assert expected_output.strip() in response_content, (
        f"Expected '{expected_output.strip()}' to be in '{response_content}', but it wasn't"
    )

def run_eval(eval_dir, llm):
    """Run evaluation for a specific directory"""
    eval_path = Path(f"evals/{eval_dir}")
    
    prompt_config, prompt_template, placeholders = load_and_validate_prompt(eval_dir, eval_path)

    # Format the prompt content
    formatted_content = prompt_template.format(**placeholders)
    
    # Prepare messages for LiteLLM
    llm_messages = [{"role": "user", "content": formatted_content}]
    logger.debug(f"Messages for LiteLLM: {llm_messages}")

    # Run the test using LiteLLM and assert the response
    run_llm_and_assert(llm, llm_messages, prompt_config["expected_output"])
    
    logger.info(f"✅ {eval_dir} passed")

# Add directory to Python path
sys.path.append(str(Path(__file__).parent))

# Note: OPENROUTER_API_KEY should be set in the environment for LiteLLM to use OpenRouter.

def get_model(model_name):
    """Get the appropriate model name for LiteLLM, ensuring OpenRouter prefix."""
    # OPENROUTER_API_KEY environment variable is used by LiteLLM for OpenRouter.
    model_aliases = {
        "mythomax": "openrouter/gryphe/mythomax-l2-13b",
        "deepseek": "openrouter/deepseek/deepseek-r1-distill-qwen-32b",
    }
    
    if model_name in model_aliases:
        return model_aliases[model_name]
    
    if "/" not in model_name:
        # For other models, if no provider is specified, assume OpenRouter
        return f"openrouter/{model_name}"
    
    # For other models that have a provider, pass them as is
    return model_name

def execute_script(script_path):
    """Execute a script and return the module"""
    spec = importlib.util.spec_from_file_location("script_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def main():
    """Main function to run the evaluations."""
    parser = argparse.ArgumentParser(description='Run evaluations with different models')

    # fast models:
    # gemini/gemini-1.5-flash-8b
    # gemini/gemini-2.0-flash-exp
    # gemini/gemini-2.0-flash-lite
    # gemini/gemini-2.5-pro-preview-03-25
    parser.add_argument('--model', type=str,
                      default='gemini/gemini-1.5-flash-8b-exp-0924', help='Model to use for evaluation')
    parser.add_argument('--evals', nargs='*', default=[], 
                      help='Specific evaluations to run (leave empty to run all)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (e.g., DEBUG, INFO)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.getLevelName(args.log_level.upper()), format='%(name)s:%(levelname)s: %(message)s')
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    llm = get_model(args.model)
    
    if args.evals:
        eval_dirs = args.evals
        logger.info(f"Running specific evaluations: {', '.join(eval_dirs)}")
    else:
        # Scan the evals directory for all available evaluations
        evals_path = Path("evals")
        eval_dirs = [d.name for d in evals_path.iterdir() if d.is_dir()]
        logger.info("Running all evaluations")
    
    for eval_dir in eval_dirs:
        try:
            run_eval(eval_dir, llm)
        except AssertionError as e:
            logger.error(f"❌ {eval_dir} failed: {str(e)}")

if __name__ == "__main__":
    main()

