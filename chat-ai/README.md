# AI Chatbot

Purpose:
1. Create a chatbot.
2. Preserve state so the chatbot has context.
3. Add UI components to control.

Pre-requisites:
An Open Router API key, saved in your bash profile as `OPENROUTER_API_KEY`.

Run this in terminal from the project directory with
```bash
shiny run --port 57942 --reload --autoreload-port 57943 chat-ai/app.py
```


# Development
The requirements are basic:
1. UI components allowing users to configure the LLM, including temperature, p, and/or k.
2. The chat.
3. The LLM API calls, and preserving state.

Stretch goal(s):
Fill out the shiny dashboard, allowing users to pass arguments to the LLM API.
1. UI components controlling functions.
2. Display the accumulated token count.
3. Allow the user to change models, but preserve the context window.
4. Show logits.


The goal was to set this up quickly, so we started with a [template](https://shiny.posit.co/py/components/display-messages/chat/#ai-quick-start) from shiny. The template does not work out of the box, and I prefer to use shiny core syntax instead of the lighter shiny express syntax.

# Concepts
For our chatbot to work, we need 3 things:
1. a chat user interface (UI) from `shiny`.
2. an API to call a LLM. `chatlas` wraps a bunch of different LLM providers.
3. preservation of the state of the conversation. `chatlas` does this for us with [Turns](https://posit-dev.github.io/chatlas/reference/Turn.html).
   1. [OpenAI suggests](https://platform.openai.com/docs/guides/text-generation#conversations-and-context) we preserve state by accumulating the conversation's tokens, so the longer the conversation goes, the more expensive the calls may get and we may hit the context window limit.


From the documentation:
1. ui.Chat() is the shiny chatbot and includes methods for working with the chat's state with `.append_message()`. 
   1. Its `@chat.on_user_submit` decorator automatically passes user input into the decorated function, where you then pass it the the LLM chat model.
   2. `append_message_stream()` [streams](https://shiny.posit.co/py/components/display-messages/chat/#appending-messages) the LLM response. This is the chat window output.
   3. These two methods should cover most use cases. [ui.Chat](https://shiny.posit.co/py/api/core/ui.Chat.html) documentation.
2. The [chatlas]([chatlas](https://posit-dev.github.io/chatlas/#model-choice)) package is the recommended tool for LLM response generation.
   1. Its `.stream_async()` [method](https://posit-dev.github.io/chatlas/reference/Chat.html#chatlas.Chat.stream_async) to generates a response to the user input passed to it.
   2. chat class, [OpenAI](https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html) example
   3. [chat object methods](https://posit-dev.github.io/chatlas/reference/Chat.html) documentation.
3. We use the openAI client [Chat](https://platform.openai.com/docs/api-reference/chat) functionality.
   1. For our stretch-goal to allow users to change the model parameters, we will pass args from shiny UI --> chatlas ChatOpenAI().stream_async() --> openAI().