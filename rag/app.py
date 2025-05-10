# ------------------------------------------------------------------------------------
# A basic Shiny Chat example powered by OpenAI.
# ------------------------------------------------------------------------------------
import os
import logging

from shiny import ui, App, reactive, render
from chatlas import ChatOpenAI

from sentence_transformers import SentenceTransformer
import trafilatura
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb
from chromadb.utils import embedding_functions as ef
from uuid import uuid4

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
LOG = logging.getLogger("RAG")
LOG.setLevel(logging.DEBUG)

# Models available on OpenRouter: https://openrouter.ai/models?max_price=0&order=top-weekly
MODEL = 'deepseek/deepseek-chat:free'
"""more options to try
google/gemma-3-1b-it:free
meta-llama/llama-3.2-1b-instruct:free
deepseek/deepseek-chat:free
"""
INITIAL_ROLE = "You are a helpful assistant."

# ChatOpenAI() requires an API key from OpenAI.
# See the docs for more information on how to obtain one.
# https://posit-dev.github.io/chatlas/reference/ChatOpenAI.html


# Set some Shiny page options
ui.page_auto(
    title="Hello OpenAI Chat",
    fillable=True,
    fillable_mobile=True,
)



##########
##  UI  ##
##########
# Add page title and sidebar
app_ui = ui.page_sidebar(
    # SIDEBAR
    ui.sidebar(
        ui.markdown('#### Chat parameters'),
        # see OpenAI() documentation: https://platform.openai.com/docs/api-reference/chat/create
        ui.tooltip(
            ui.input_text_area(
                id="system_prompt",
                label="System Prompt",
                value="You are a helpful assistant.",
                placeholder="Enter the system prompt for the chatbot",
            ),
            "The initial instructions for the chatbot to follow.",
            id="tt_system_prompt",
            placement="right",
        ),
        ui.tooltip(
            ui.input_slider(id="temperature", label="Temperature", min=0, max=2, value=1, step=0.1),
            "Randomness in responses. Higher values like 1.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",  
            id="tt_temperature",  
            placement="right",
        ),
        ui.tooltip(
            ui.input_slider(
                id="top_p",
                label="Top P",
                min=0,
                max=1,
                value=1,
                step=0.05,
            ),
            "Controls the diversity of chat responses by only considering the top_p cumulative probabilities of response tokens. ex. 0.1 limits the response to the top 10% likely tokens.",
            id="tt_top_p",
            placement="right",
        ),
        ui.tooltip(
            ui.input_numeric(
                id="max_completion_tokens",
                label="Max Tokens",
                min=1,
                max=2048,
                value=None
            ),
            "The maximum number of tokens to generate in the response.",
            id="tt_max_tokens",
            placement="right",
        ),
        ui.tooltip(
            ui.input_slider(
                id="frequency_penalty",
                label="Frequency Penalty",
                min=-2,
                max=2,
                value=0,
                step=0.1,
            ),
            "Penalizes new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
            id="tt_frequency_penalty",  
            placement="right",
        ),
        ui.tooltip(
            ui.input_slider(
                id="presence_penalty",
                label="Presence Penalty",
                min=-2,
                max=2,
                value=0,
                step=0.1,
            ),
            "Penalizes new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
            id="tt_presence_penalty",
            placement="right",
        ),
    ),

    # MAIN PAGE
    ui.navset_pill(  
        ui.nav_panel("Chat", 
                    ui.markdown('#### Retrieval-Augmented Generation'),
                    ui.markdown("""Submit a URL to provide context for your chat. 
                    The website will be scraped, chunked, encoded, and saved to a vector store (see the `RAG` tab to peek at the vector store). 
                    The `top k` value is how many relevant chunks to append to your chat as context."""),
                    ui.layout_columns(
                        ui.card(
                            ui.input_text(
                                id="url_input",
                                label="URL for Context",
                                placeholder="Enter a URL (e.g., Wikipedia page)",
                                width="100%",
                            ),
                            ui.input_action_button(id="url_submit", label="Submit URL", style="background-color: #c8e7f1;", width="40%"),
                        ),
                        ui.card(
                            ui.input_numeric(
                                id="top_k",
                                label="Top K References",
                                value=3,
                                min=1,
                                max=10,
                            ),
                        ),
                        col_widths={"sm": (8, 4)},
                    ),
                    ui.markdown('---\n## Chat'),
                    ui.chat_ui(id="my_chat")
                    ),
        ui.nav_panel("RAG", 
                      ui.markdown('### RAG'),
                      ui.markdown("Show what's happening under the hood. When a user submits a URL, the text is chunked and stored in the vector store. When a user submits a query, the top K relevant chunks are retrieved and used as context for the chat model."),
                      ui.code('vector_store.peek()'),
                      ui.output_code("vector_store_peek"),
                      ui.output_ui("under_the_hood_tab"),
                     ),
        ui.nav_panel("Troubleshooting",
                    ui.card_header("Troubleshooting"),
                    ui.output_ui("troubleshoot_text"),
                    )
    )
)


##################
###   SERVER   ###
##################

def server(input, output, session):

    # initialize a dict of state values
    my_state = {
        'model': reactive.value(MODEL),
        'current_model': reactive.value(MODEL),
        'model_is_valid': reactive.value(True),
        'sys_prompt': reactive.value(INITIAL_ROLE)
    }
    # debugging state values
    debug_keys = ['top_k', 'top_k_chunks', 'combined_input', 'vector_store_peek']
    debug_info = {k:reactive.Value(None) for k in debug_keys}
    # Note: dict.fromkeys() does not work with reactive values, so we have to initialize them one by one.

    # Define the chat variable
    chat = ui.Chat(id="my_chat", messages=["Hello! How can I help you today?"])

    # Initialize the chat model. Values can be changed later.
    chat_model = ChatOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=MODEL,
        system_prompt="You are a helpful assistant.",
    )
    # To update this model, we need to store it in as a reactive value.
    my_state['model_instance'] = reactive.value(chat_model)

    #########################
    ###   RAG variables   ###
    #########################
    # Set up the chroma vector store
    # Specify a sentence transformer because 1) it's free (versus embedding models) and 
    # 2) Macs can have issues with Intel vs Mac chips with the default embedding function.
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    #embedding_function = ef.SentenceTransformerEmbeddingFunction(st_model)
    embedding_function = ef.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # While chroma.EphemeralClient() would be ideal for this demo use case,
    # Persistent or Client-Server mode is more what we would use in production.
    # so we'll add some clunkier code to delete the old collection.
    client = chromadb.PersistentClient(path="rag/chroma")

    collection = "rag_url_context"
    try:
        client.delete_collection(name=collection)
    except ValueError as e:
        pass

    # Create vector store
    vector_store = client.create_collection(name=collection, embedding_function=embedding_function)

    # Text splitter for chunking text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    ############################
    ###   SERVER FUNCTIONS   ###
    ############################
    ### CHAT FUNCTIONS
    # Generate a response when the user submits a message
    @chat.on_user_submit
    async def handle_user_input(user_input: str):
        # Query the vector store for relevant chunks
        rag_context = query_vector_store(user_input=user_input)

        # Combine the relevant chunks with the user input
        if rag_context is None or rag_context == "":
            context=""
        else:
            context = f"Context: {rag_context} \n\n"
        combined_input = f"{context}User Input: {user_input}"
        LOG.info(f"User query: {user_input};\nRAG context: {context}")

        # User inputs to pass to OpenAI().chat.completion()
        model_args = {
            "temperature": input.temperature(),
            "top_p": input.top_p(),
            "max_tokens": input.max_completion_tokens(),
            "frequency_penalty": input.frequency_penalty(),
            "presence_penalty": input.presence_penalty()
        }
        # LOG.info(model_args)
        # LOG.info(f"PROMPT: {chat_model._turns[0].__dict__}")
        # LOG.info(f"MODEL: {chat_model.__dict__['provider'].__dict__['_model']}")

        # Store debugging information for the "Under the hood" tab
        debug_info['top_k_chunks'].set(rag_context)
        debug_info['combined_input'].set(combined_input)

        # Generate a response from the chat model
        LOG.debug(f"combined_input: {combined_input}")
        response = await chat_model.stream_async(combined_input, kwargs=model_args)

        await chat.append_message_stream(message=response)



    ### RAG FUNCTIONS
    @reactive.effect
    @reactive.event(input.url_submit)
    def add_url_to_vector_store():
        """Fetch and process content from the URL, then chunk and store it in the vector store."""
        url = input.url_input()
        if not url:
            return
        
        # Fetch and process content from the URL
        website = trafilatura.fetch_url(url)
        url_text = trafilatura.extract(website)

        # The text needs to be converted to a langchain Document object so we can use the text splitter
        url_doc = Document(
            page_content=url_text,
            metadata={"source": url}
        )

        # Commented out here because we intialize once outside of the function.
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = text_splitter.split_documents([url_doc])

        LOG.info(f"Adding scraped data from {url} to vector store.")
        LOG.info(f"Preview: {url_text[:100]}...")
        vector_store.add(documents=[t.__dict__['page_content'] for t in texts],
                 metadatas=[t.__dict__['metadata'] for t in texts],
                 ids=[str(uuid4()) for _ in range(len(texts))])
        debug_info["vector_store_peek"].set(vector_store.peek())

        # Reset the input field
        ui.update_text("url_input", value="")
        LOG.info(f"Added {len(texts)} chunks to vector store.")

    def query_vector_store(user_input:str):
        """Query the vector store for the top k relevant chunks."""

        k = input.top_k()  # from UI
        if not user_input:
            return ""
        
        # Query the vector store for the top k relevant chunks
        results = vector_store.query(
            query_texts=[user_input],
            n_results=k,
            include=["documents"]#, "metadatas", "distances"]
        )

        # Store the top k input for debugging
        debug_info['top_k'].set(input.top_k())
        
        # Combine the list of chunks into one string and return
        return " \n ".join(results.get("documents", "")[0])



    ### UI FUNCTIONS
    # OpenAI recommends that users can set temperature OR top_p, but not both.
    # Therefore, we will reset one when the other is changed.
    @reactive.effect
    @reactive.event(input.temperature)
    def _reset_top_p():
        """Reset top_p when temperature is changed."""
        if input.temperature() != 1:
            ui.update_slider("top_p", value=1)


    @reactive.effect
    @reactive.event(input.top_p)
    def _reset_temperature():
        """Reset temperature when top_p is changed."""
        if input.top_p() != 1:
            ui.update_slider("temperature", value=1)


    @reactive.effect
    @reactive.event(input.system_prompt)
    def _update_system_role():
        """ Chatlas allows users to specify a system_prompt,
        but if we want to change it without reinitializing the chat,
        we modify the instance.
        
        The Chat turns contains the conversation.
        # The first item should be the system role.
        # We will grab that item, check for system, and update.
        """
        sys_role = {'role': "system", 'text': input.system_prompt()}
        chat_model._turns[0].__dict__.update(sys_role)

    # Troubleshooting functions to show variables
    @render.ui
    def troubleshoot_text():
        """Display troubleshooting information."""
        sys_role = {k:v for k,v in chat_model._turns[0].__dict__.items() if k in ['role', 'text']}

        return ui.markdown(
f"""```python
input.system_prompt: {input.system_prompt()}
chat_model role: {sys_role}  
current_model: {chat_model.__dict__['provider'].__dict__['_model']}
```
""")


    @render.code
    def vector_store_peek():
        """Display the contents of the vector store."""
        return debug_info.get("vector_store_peek", "No vector store")()

    # Add a new "Under the hood" tab
    @render.ui
    def under_the_hood_tab():
        topk = debug_info.get("top_k", "K")()
        top_k_chunks = debug_info.get("top_k_chunks", [])()
        combined_input = debug_info.get("combined_input", "No combined input")()
        vsp = debug_info.get("vector_store_peek", "No vector store")()
        

        return ui.markdown(
f"""### Under the Hood

#### Top {topk} Relevant Chunks
```python
{top_k_chunks}
```

#### Combined Input
```python
{combined_input}
```
"""
        )


app = App(app_ui, server)