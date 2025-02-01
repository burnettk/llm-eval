from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

# Initialize the LLM using OpenRouter
llm = ChatOpenAI(
    model_name="openai/gpt-4",  # Or any other model available through OpenRouter
    openai_api_key=os.environ['OPENROUTER_API_KEY'],
    base_url="https://openrouter.ai/api/v1"
)

# Create a prompt template
prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")

# Generate a response
response = llm(prompt.format_messages(country="France"))
print(response.content)
