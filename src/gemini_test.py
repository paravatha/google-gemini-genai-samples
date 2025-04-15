import os
import logging
from dotenv import load_dotenv
import mlflow
from pprint import pprint
import google.genai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_env_variables():
    """Load and validate environment variables."""
    load_dotenv()
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    gemini_model_name = os.getenv("GEMINI_MODEL_NAME")
    google_api_key = os.getenv("GEMINI_API_KEY")

    if not mlflow_tracking_uri:
        raise ValueError("Environment variable MLFLOW_TRACKING_URI is not set.")
    if not gemini_model_name:
        raise ValueError("Environment variable GEMINI_MODEL_NAME is not set.")
    if not google_api_key:
        raise ValueError("Environment variable GEMINI_API_KEY is not set.")

    logging.info("Environment variables loaded successfully.")
    logging.info(f"MLFlow Tracking URI: {mlflow_tracking_uri}")
    logging.info(f"Gemini Model Name: {gemini_model_name}")
    return mlflow_tracking_uri, gemini_model_name, google_api_key

def configure_mlflow(mlflow_tracking_uri, model_name="gemini"):
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(f"test-{model_name}")
    mlflow.gemini.autolog()



def generate_content(client, model_name, prompt):
    """Generate content using the Gemini model."""
    logging.info(f"Prompt: {prompt}")
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        # Extract and pprint the text response
        if response and response.text:
            text_response = response.text
            logging.info(f"text_response: {text_response}")
        else:
            logging.warning("No valid response received.")
        # logging.info("Response:")
    except Exception as e:
        logging.error(f"Error generating content: {e}")

def main():
    """Main function to execute the script."""
    mlflow_tracking_uri, gemini_model_name, google_api_key = load_env_variables()
    configure_mlflow(mlflow_tracking_uri, gemini_model_name)

    # Configure the SDK with the API key
    client = genai.Client(api_key=google_api_key)

    # Define the prompt
    contents = "What is the capital of France?"
    generate_content(client, gemini_model_name, contents)

if __name__ == "__main__":
    main()
