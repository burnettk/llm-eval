from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

# Initialize the LLM using OpenRouter
api_key = os.environ['OPENROUTER_API_KEY']
llm = ChatOpenAI(
    model_name="gryphe/mythomax-l2-13b",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# # Create a prompt template
# prompt = ChatPromptTemplate.from_template("What is 2 + 2?")
#
# # Generate a response
# response = llm(prompt.format_messages())
# printm(response.content)

# Define a test case
def test_case(prompt_string, expected_output):
    prompt = ChatPromptTemplate.from_template(prompt_string)
    response = llm(prompt.format_messages())
    assert expected_output in response.content, f"Expected {expected_output} to be in {response.content}, but it wasn't"

# Run the test case
test_case("What is 2 + 2?", "4")

