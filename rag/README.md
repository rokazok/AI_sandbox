# Retrieval Augmented Generation
LLMs have limitations, including limited specialty knowledge or lack of knowledge for events that happened after the model was trained. Retrieval Augmented Generation (RAG) provides the LLM with additional context. This demo allows users to provide a url for additional context, such as a Wikipedia link. The process is generally:
1. Provide and process new data to augment the LLM.
2. When the user provides a prompt, find the most similar augmented data to the prompt and send it as context to the LLM with the user's prompt.
3. The LLM references the context when inferring an answer.

The result of this demo:
![RAG demo](https://drive.google.com/file/d/1x9ND51sIzMB1gUoIe2sMYgSbjeW475i8/view?usp=drive_link)


In more detail, RAG includes:
1. Ingesting new, augmented data.
   1. The new data and how they are loaded can vary, including [document loaders](https://python.langchain.com/v0.2/docs/concepts/#document-loaders) or web-scraping (this demo).
   1. Once data are loaded, they need to be split into smaller chunks for both indexing and lowering costs by providing less data in the context window.
   1. The splits are converted to embeddings and processed in the same way the LLM processes user text, i.e. the model contains information on the relationships between embeddings.
   1. The split embeddings are stored in a database, vector store, local memory, etc.
1. Compare user prompt to augmented data.
   1. Convert user prompt into embeddings.
   1. Calculate cosine similarity between query embeddings and augmented embeddings.
   1. Decode embeddings back into text and append the text as context to the user query.
1. Send the original user prompt + augmented context to the LLM and return the inference.

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
1. UI components allowing users to configure LLM settings and submit a URL for RAG.
2. The chat.
3. RAG:
   1. URL input + web-scraping.
   1. Chunk and encode web content.
   1. Vectore store
   1. Query vector store and append results as context to user query.

Requirements 1 and 2 are already met with the [chatbot](../chatbot_app/README.md).