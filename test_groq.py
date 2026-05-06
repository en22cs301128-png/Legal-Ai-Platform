import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

client = Groq(api_key=os.getenv("OPENAI_API_KEY"))

try:
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
