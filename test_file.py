def test_case(question, expected_output):
    # Simulate a response from an LLM
    response = type('obj', (object,), {'content': "The answer is 4."})
    
    # Check if expected output is in the response
    assert expected_output in response.content, f"Expected '{expected_output}' to be in response, but got '{response.content}'"

# Test case to verify presence of "4" in the response
test_case("What is 2 + 2?", "4")
