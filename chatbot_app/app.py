# ------------------------------------------------------------------------------------
# A basic Shiny Chat example powered by OpenAI.
# ------------------------------------------------------------------------------------
import os

from shiny import ui, App, reactive, render
from chatlas import ChatOpenAI
from requests.exceptions import HTTPError
import time


#MODEL = 'google/gemini-2.0-flash-lite-preview-02-05:free'
MODEL = 'google/gemma-3-1b-it:free'
"""more options to try
meta-llama/llama-3.2-1b-instruct:free
microsoft/phi-3-mini-128k-instruct:free
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
        ui.markdown('#### Model parameters'),
        # see OpenAI() documentation: https://platform.openai.com/docs/api-reference/chat/create
        ui.tooltip(
            ui.input_text('model', 'Model', value=MODEL),
            "See models from https://openrouter.ai/api/v1",  
            id="tt_model",  
            placement="right",
        ),
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
        ui.input_action_button(id="submit", label="Submit", style="background-color: #c8e7f1;"),
        ui.input_action_button(id="reset", label="Reset", style="background-color: #ffcccc;"),

        ui.markdown('---'),
        ui.markdown('#### Chat parameters'),
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
    # ui.layout_columns(
    #     ui.card(
    #         ui.card_header("Troubleshooting"),
    #         ui.output_ui("troubleshoot_text"),
    #         #height="200px", fill=True
    #     )
    # ),
    ui.chat_ui(id="my_chat")
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

    # We want to trigger the submit button programmatically in addition to the UI
    # so we store its counter as a reactive value.
    submit_btn = reactive.value(0)

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


    # Generate a response when the user submits a message
    @chat.on_user_submit
    async def handle_user_input(user_input: str):

        # User inputs to pass to OpenAI().chat.completion()
        model_args = {
            "temperature": input.temperature(),
            "top_p": input.top_p(),
            "max_tokens": input.max_completion_tokens(),
            "frequency_penalty": input.frequency_penalty(),
            "presence_penalty": input.presence_penalty()
        }
        print(model_args)
        print(f"PROMPT: {chat_model._turns[0].__dict__}")
        print(f"MODEL: {chat_model.__dict__['provider'].__dict__['_model']}")
        # Generate a response from the chat model
        response = await chat_model.stream_async(user_input, kwargs=model_args)

        await chat.append_message_stream(message=response)
    

    # The submit button will change the system role and/or the model. We let one button
    # handle both changes to keep the UI clean. We could use one update function,
    # but we separate the update_system_role() function for future-proofing, where 
    # we may keep the model static, but this may not be necessary, i.e. update_model() can handle both cases.
    @reactive.effect
    @reactive.event(input.submit)
    def _increment_submit_btn():
        """Increment the submit button counter.
        Normally, we could just decorate the submit button function with
        reactive.event(input.submit), but we are also triggering the function
        with the reset button, so both need to increment a secondary counter, submit_btn,
        and change some holding values"""
        # We have to save the inputs into reactive values since we change them programmatically.
        my_state['model'].set(input.model())
        my_state['sys_prompt'].set(input.system_prompt())

        submit_btn.set(submit_btn() + 1)


    @reactive.effect
    @reactive.event(submit_btn)
    def _update_system_role():
        """ Chatlas allows users to specify a system_prompt,
        but if we want to change it without reinitializing the chat,
        we modify the instance.
        
        The Chat turns contains the conversation.
        # The first item should be the system role.
        # We will grab that item, check for system, and update.
        """
        sys_prompt = my_state['sys_prompt']()

        # if the input model is the same as the current model, 
        # then we can update just the system role instead of instantiating a new model.
        if input.model() == my_state['model']():
            print(f"submit w/ system role: {sys_prompt}")

            sys_role = {**chat_model._turns[0].__dict__}
            if sys_role['role'] == "system":
                sys_role['text'] = sys_prompt

                # This is an extra update for consistency, but it is not used.
                sys_role['contents'][0].__dict__['text']
                    
            # Update the system role.
            chat_model._turns[0].__dict__.update(sys_role)

            def rolll(d):
                return {k:v for k,v in d.items() if k=='text'}
            print(f"We tried to change chat_model system role to: {rolll(sys_role)}")
            print(f"We got chat_model role to: {rolll(chat_model._turns[0].__dict__)}")

        else:
            pass



    # @reactive.effect
    # @reactive.event(input.submit)
    # def _store_model():
    #     """ We don't control the model that the user inputs. If the user enters an
    #     incorrect model endpoint, we will throw an error then revert to the last working model.
        
    #     This function stores the last working model and system prompt."""
    #     my_state['model'].set(input.model())

    # @reactive.effect
    # @reactive.event(input.submit)
    # def update_system_role():
    #     """ Chatlas allows users to specify a system_prompt,
    #     but if we want to change it without reinitializing the chat,
    #     we modify the instance.
        
    #     The Chat turns contains the conversation.
    #     # The first item should be the system role.
    #     # We will grab that item, check for system, and update.
    #     """
    #     # if the input model is the same as the current model, 
    #     # then we can update just the system role instead of instantiating a new model.
    #     if input.model() == my_state['model']:

    #         sys_role = my_state['model_instance']._turns[0].__dict__
    #         if sys_role['role'] == "system":
    #             sys_role['text'] = input.system_prompt()

    #             # This is an extra update for consistency, but it is not used.
    #             sys_role['contents'][0].__dict__['text']
                    
    #         # Update the system role.
    #         my_state['model_instance']._turns[0].__dict__.update(sys_role)

    #     else:
    #         pass

    @reactive.effect
    @reactive.event(submit_btn)
    def _update_model():
        """ Chatlas allows users to specify a system_prompt,
        but if we want to change it without reinitializing the chat,
        we modify the instance.
        
        The Chat turns contains the conversation.
        # The first item should be the system role.
        # We will grab that item, check for system, and update.
        """
        # print(f"""update chat_model. Current: {chat_model}\n
        # chat_model.__dict__: {chat_model.__dict__}\n
        # type(chat_model): {type(chat_model)}\n
        # my_state['model_instance']: {my_state['model_instance']()}\n
        # """)
        OLD_MODEL = my_state['current_model']() # DELETE
        new_model = my_state['model']()
        # if new_model != my_state['current_model']():
        #     # export context (turns) from old model
        #     turns = chat_model.get_turns()

        #     # Load context into a new chat instance
        #     chat_model = ChatOpenAI(
        #         api_key=os.environ.get("OPENROUTER_API_KEY"),
        #         base_url="https://openrouter.ai/api/v1",
        #         model=new_model,
        #         system_prompt=my_state['sys_prompt'](),
        #         turns=turns
        #     )

        # We cannot instantiate a new chat_model, so we will update the model in place.
        chat_model.__dict__['provider'].__dict__['_model'] = new_model


        # Update model state
        my_state['current_model'].set(new_model)
        print(f"Model: NEW  {my_state['current_model']()} <-- {OLD_MODEL} (OLD)")

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
    @reactive.event(input.reset)
    def _reset_model():
        """Reset model and system prompt."""
        OLD_SYSTEM_PROMPT = input.system_prompt()  # DELETE

        ui.update_text("model", value=MODEL)
        ui.update_text_area("system_prompt", value=INITIAL_ROLE)

        # There is a strange pattern where the reseted UI values don't reflect 
        # the current state, so we need to store the system prompt and model as 
        # reactive values and change them. Sleeping doesn't help.
        my_state['model'].set(MODEL)
        my_state['sys_prompt'].set(INITIAL_ROLE)

        # Once we reset options, we want to submit them instead of requiring users
        # to click the submit button again.
        print(f"the input.system_prompt() is NOW:  {input.system_prompt()}. It was: {OLD_SYSTEM_PROMPT}. state sys prompt: {my_state['sys_prompt']}")
        submit_btn.set(submit_btn() + 1)


    # Troubleshooting functions to show variables
    @render.ui
    def troubleshoot_text():
        """Display troubleshooting information."""
        sys_role = {k:v for k,v in chat_model._turns[0].__dict__.items() if k in ['role', 'text']}

        return ui.markdown(
f"""```python
input.submit: {input.submit()}
submit_btn: {submit_btn()}
input.system_prompt: {input.system_prompt()}
chat_model role: {sys_role}  
input.model: {input.model()}
my_state['model']: {my_state['model']()}
current_model: {chat_model.__dict__['provider'].__dict__['_model']}
```
""")


app = App(app_ui, server)