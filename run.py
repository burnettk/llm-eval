from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

# Initialize the LLM (using OpenRouter)
api_key = os.environ['OPENROUTER_API_KEY']
llm = ChatOpenAI(
    model_name="gryphe/mythomax-l2-13b",
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Define a test case
def test_case(prompt, expected_output):
    response = llm([HumanMessage(content=prompt)])
    assert response.content == expected_output, f"Expected {expected_output}, but got {response.content}"

# Run the test case
test_case("What is 2 + 2?", "The answer is 4.")
