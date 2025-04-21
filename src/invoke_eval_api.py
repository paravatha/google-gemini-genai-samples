import os
import json
import requests
import tiktoken  # Ensure you have the `tiktoken` library installed
from dotenv import load_dotenv


def calculate_tokens(text, encoding_name="cl100k_base"):
    """Calculate the number of tokens in a given text using the specified encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def main():
    # Load the JSON file
    with open("test_payload.json", "r") as file:
        data = json.load(file)

    # Iterate over citations and calculate token counts
    citations = data.get("prediction", {}).get("citations", [])
    token_counts = []
    for i, citation in enumerate(citations):
        token_count = calculate_tokens(citation)
        token_counts.append({"citation_index": i, "token_count": token_count})

    # Print the token counts
    for entry in token_counts:
        print(f"Citation {entry['citation_index']}: {entry['token_count']} tokens")

    # Calculate token counts for 'context' and 'query' keys
    context = data.get("context", "")
    query = data.get("query", "")

    context_token_count = calculate_tokens(context)
    query_token_count = calculate_tokens(query)

    print(f"Context: {context_token_count} tokens")
    print(f"Query: {query_token_count} tokens")
    load_dotenv()
    # Define the API URL
    api_url = os.getenv("EVAL_API_URL")
    api_key = os.getenv("EVAL_API_KEY")
    if not api_url or not api_key:
        raise ValueError("EVAL_API_URL and EVAL_API_KEY environment variables must be set.")

    # Send the POST request
    try:
        import time  # Import the time module to measure response time

        headers = {"x-functions-key": api_key } 
        
        # Measure the start time
        start_time = time.time()
        
        response = requests.post(api_url, json=data, headers=headers)  # api_url is guaranteed to be a valid string here
        response.raise_for_status()  # Raise an error for HTTP errors
        
        # Measure the end time
        end_time = time.time()
        
        # Calculate and print the response time
        response_time = end_time - start_time
        print(f"API Response Time: {response_time:.2f} seconds")
        
        print("API Response:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error while calling the API:", e)

if __name__ == "__main__":
    main()
