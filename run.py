from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import json
from pathlib import Path

# Initialize the LLM using OpenRouter
api_key = os.environ['OPENROUTER_API_KEY']
llm = ChatOpenAI(
    model_name="gryphe/mythomax-l2-13b",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

def run_eval(eval_dir):
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

if __name__ == "__main__":
    # Run all evaluations
    for eval_dir in ["math", "translation"]:
        try:
            run_eval(eval_dir)
        except AssertionError as e:
            print(f"❌ {eval_dir} failed: {str(e)}")
