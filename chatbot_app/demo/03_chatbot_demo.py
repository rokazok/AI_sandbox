# ------------------------------------------------------------------------------------
# A basic Shiny Chat example powered by OpenAI.
# ------------------------------------------------------------------------------------
# You may need to change the filename to app.py.
# Run with shiny run --port 64343 --reload --autoreload-port 64344 chat-ai/demo/03_chatbot_demo.py
import os

from chatlas import ChatOpenAI

from shiny.express import ui

MODEL = 'google/gemini-2.0-flash-lite-preview-02-05:free'

# ChatOpenAI() requires an API key from OpenAI.
# See the docs for more information on how to obtain one.
# https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html

chat_model = ChatOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=MODEL,
    system_prompt="You are a helpful assistant.",
)



# Create and display a Shiny chat component
chat = ui.Chat(
    id="chat",
    messages=["Hello! How can I help you today?"],
)
chat.ui()


# Generate a response when the user submits a message
@chat.on_user_submit
async def handle_user_input(user_input: str):
    response = await chat_model.stream_async(user_input)
    await chat.append_message_stream(message=response)
