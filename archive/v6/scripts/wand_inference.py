import openai
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from a .env file if present

client = openai.OpenAI(
    # The custom base URL points to Serverless Inference
    base_url='https://api.inference.wandb.ai/v1',

    # Get an API key from https://wandb.ai/authorize
    # Consider setting it in the environment as OPENAI_API_KEY instead for safety
    api_key=os.getenv("WANDB_API_KEY"),

    # Optional: Team and project for usage tracking
    project="codhe-synextra/inference",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ],
)

print(response.choices[0].message.content)