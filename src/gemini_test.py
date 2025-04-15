import os
from dotenv import load_dotenv
import mlflow
import google.genai as genai

# Load environment variables from .env file
load_dotenv()


# Turn on auto tracing for Gemini
mlflow.gemini.autolog()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
gemini_model_name = os.getenv("GEMINI_MODEL_NAME")
google_api_key = os.getenv("GEMINI_API_KEY")


# check if the environment variables are set
if not all([mlflow_tracking_uri, gemini_model_name, google_api_key]):
    raise ValueError(
        "Please set the MLFLOW_TRACKING_URI, GEMINI_MODEL_NAME, and GEMINI_API_KEY environment variables."
    )
# Print the environment variables to verify they are loaded correctly
print("Environment Variables Loaded:")
print(f"MLFlow Tracking URI: {mlflow_tracking_uri}\n Gemini Model Name: {gemini_model_name}")

# Optional: Set a tracking URI and an experiment
mlflow.set_tracking_uri(mlflow_tracking_uri)
# Optional: Set an experiment name
mlflow.set_experiment("test-gemini")


# Configure the SDK with your API key.
client = genai.Client(api_key=google_api_key)

contents = "What is the capital of France?"
print(f"Prompt: {contents}")

# Use the generate_content method to generate responses to your prompts.
response = client.models.generate_content(model="gemini-1.5-flash", contents=contents)

print(f"Response: {response}")
