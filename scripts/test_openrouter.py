from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# First API call with reasoning
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "How many r's are in the word 'strawberry'?"}
    ],
    extra_body={"reasoning": {"enabled": True}},
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message
print(response)

# Preserve the assistant message with reasoning_details
messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": response.content,
        "reasoning_details": response.reasoning_details,  # Pass back unmodified
    },
    {"role": "user", "content": "Are you sure? Think carefully."},
]

# Second API call - model continues reasoning from where it left off
response2 = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"reasoning": {"enabled": True}},
)

print(response2)
