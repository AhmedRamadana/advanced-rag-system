"""
Quick sanity check: confirms the Gemini API key works before we build
anything else on top of it.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load variables from the .env file into the environment
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-3.6-flash")
response = model.generate_content("Reply with exactly one word: Connected")

print("Response from Gemini:", response.text.strip())