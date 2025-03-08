# minimal demo to test chatlas
# adapted from https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html
# run with: python 02_chatlas_demo.py

import os
from chatlas import ChatOpenAI

MODEL = 'google/gemini-2.0-flash-lite-preview-02-05:free'

chat = ChatOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=MODEL,
    system_prompt="You are a helpful assistant.",
)
chat.console()