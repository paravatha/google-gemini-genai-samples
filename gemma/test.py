import time

from mlx_lm.generate import generate
from mlx_lm.utils import load

# these models requires over 36 GB of memory
# model_name = "mlx-community/medgemma-27b-text-it-8bit"
# model_name = "mlx-community/medgemma-27b-text-it-bf16"
# model_name = "mlx-community/Llama-4-Scout-17B-16E-Instruct-8bit"

# image models
# model_name = "mlx-community/MedGemma-4B-IT-8bit"
# model_name = "mlx-community/medgemma-4b-it-bf16"

# text models
# model_name = "mlx-community/medgemma-27b-text-it-6bit"
# model_name = "mlx-community/medgemma-27b-text-it-8bit"

model_name = "mlx-community/medgemma-27b-text-it-4bit"

model, tokenizer = load(model_name)
start_time = time.time()
prompt = "You are a medical expert answering medical questions re You will answer in a concise way and in a single sentence. "

question = "What is the treatment for VTE?"

# Use the prompt as-is, since apply_chat_template is not supported by this tokenizer.


response = generate(model, tokenizer, prompt=prompt + question, verbose=False)
end_time = time.time()

print(f"Using model: {model_name} ")
print(f"Prompt: {prompt + question}")
print(f"Response: {response}")
print(f"Response generated in {end_time - start_time:.2f} seconds.")
