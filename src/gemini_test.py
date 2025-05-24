import os
import logging
from dotenv import load_dotenv
import mlflow
import google.genai as genai
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_env_variables() -> Tuple[str, str, str]:
    """Load and validate required environment variables."""
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
    logging.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    logging.info(f"Gemini Model Name: {gemini_model_name}")
    return mlflow_tracking_uri, gemini_model_name, google_api_key

def configure_mlflow(mlflow_tracking_uri: str, model_name: str = "gemini") -> None:
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(f"test-{model_name}")
    mlflow.gemini.autolog()
    logging.info("MLflow configured and autologging enabled.")

def generate_content(client: genai.Client, model_name: str, prompt: str) -> str:
    """Generate content using the Gemini model and return the response text."""
    logging.info(f"Prompt: {prompt}")
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        if response and response.text:
            logging.info(f"Model response: {response.text}")
            return response.text
        else:
            logging.warning("No valid response received from model.")
            return ""
    except Exception as e:
        logging.error(f"Error generating content: {e}")
        return ""

def main() -> None:
    """Main function to execute the script."""
    try:
        mlflow_tracking_uri, gemini_model_name, google_api_key = load_env_variables()
        configure_mlflow(mlflow_tracking_uri, gemini_model_name)
        client = genai.Client(api_key=google_api_key)
        prompt = "What is the capital of France?"
        result = generate_content(client, gemini_model_name, prompt)
        if result:
            print(f"Gemini response: {result}")
        else:
            print("No response generated.")
    except Exception as e:
        logging.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
