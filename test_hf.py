# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai"
# ]
# ///
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_API_KEY"],
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "How many 'G's in 'huggingface'?"
        }
    ],
)
print(completion.choices[0].message.content)