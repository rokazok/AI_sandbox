import os
from fastapi import FastAPI
from enum import Enum
from openai import OpenAI
import numpy as np
from typing import List


# CONSTANTS
API_KEY = os.environ.get('OPENROUTER_API_KEY')
MODEL = "meta-llama/llama-3.2-1b-instruct:free"

# Sample prompt as a default value to be replaced by user input.
# If we see this in our output, we know the message was not passed into the prompt() function.
sample_prompt = [
    {"role": "user", "content": "Tell me 3 things that are red."},
    ]

# LLM function (see basic_prompt.py)
# Initialize the client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

def send_prompt(model: str = None, 
        messages: list = sample_prompt, 
        max_completion_tokens: int = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,):
    """
    This wrapper around OpenAI chat.completions.create() generates a 
    completion using the specified model and messages (required).
    We wrap this function because we will use it several times in the different API functions below.
    Other optional parameters are set to their defaults. These will be query parameters in the API.
    See https://platform.openai.com/docs/api-reference/chat/create
    

    Args:
        model (str): The model to use for generating the completion. Defaults to MODEL.
        messages (list, optional): A list of messages to use as input for the model. Defaults to None.
        max_completion_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        top_p (float, optional): Nucleus sampling probability. Defaults to 1.0.
        frequency_penalty (float, optional): Frequency penalty. Defaults to 0.0.
        presence_penalty (float, optional): Presence penalty. Defaults to 0.0.

    Returns:
        str: The content of the generated completion.
        

    Examples:
        >>> send_prompt(messages=[{"role": "user", "content": "Tell me a joke."}])
        'Why did the chicken cross the road? To get to the other side.'
    """
    if not model:
        model = MODEL
    
    completion = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content

# Create API
app = FastAPI()

# Predefined joke types 
class JokeType(str, Enum):
    generic = "random"
    knockknock = "knock-knock"
    chicken = "chicken crossing the road"

@app.post("/joke/{type}")
def get_joke(
    type: JokeType = '',
    model: str|None = None,
    about: str|None = None,
    max_completion_tokens: int|None = None,
    temperature: float|None = None,
    top_p: float|None = None,
    frequency_penalty: float|None = None,
    presence_penalty: float|None = None,
    ):

    # Query parameter- if user provides a subject, include it in the prompt
    subject = ""
    if about:
        subject = f" about {about}"
    if type == JokeType.generic:
        type = ""

    msgs = [
        {"role":"developer", "content": "You are a comedian who tells jokes. Return only the joke."},
        {"role": "user", "content": f"Tell me a {type} joke{subject}."},
    ]
    print(msgs[1]['content'])

    # Sometimes we get a blank string. Repeat until we get a joke.
    completion=''
    while completion == '':
        completion = send_prompt(model=model
            , messages=msgs
            , max_completion_tokens=max_completion_tokens
            , temperature=temperature
            , top_p=top_p
            , frequency_penalty=frequency_penalty
            , presence_penalty=presence_penalty) 
    return completion

@app.post("/limerick")
def get_limerick(
    model: str|None = None,
    about: str|None = None,
    max_completion_tokens: int|None = None,
    temperature: float|None = None,
    top_p: float|None = None,
    frequency_penalty: float|None = None,
    presence_penalty: float|None = None,
    ):

    # Query parameter- if user provides a subject, include it in the prompt
    subj = ""
    if about:
        subj = f"about {about}"

    msgs = [
    {"role": "developer", "content": "You are a poet. Just tell the limerick. Do not explain it or add extra context."},
    {"role": "user", "content": f"Tell me a limerick{subj}."},
    ]

    completion=''
    while completion == '':
        completion = send_prompt(model=model
            , messages=msgs
            , max_completion_tokens=max_completion_tokens
            , temperature=temperature
            , top_p=top_p
            , frequency_penalty=frequency_penalty
            , presence_penalty=presence_penalty) 

    return completion

@app.get("/author")
def get_quote(
    author: str|None = None,
    model: str|None = None,
    max_completion_tokens: int|None = None,
    temperature: float|None = None,
    top_p: float|None = None,
    frequency_penalty: float|None = None,
    presence_penalty: float|None = None,
    ):

    # Query parameter- if user provides an author, include it in the prompt
    if author:
        author = f"the author {author}"
    else:
        author = "a famous author"

    msgs = [
    {"role": "developer", "content": "You are a famous author. Just tell the quote. Do not explain it or add extra context."},
    {"role": "user", "content": f"Give me a quote by {author}."},
    ]


    completion = send_prompt(model=model
        , messages=msgs
        , max_completion_tokens=max_completion_tokens
        , temperature=temperature
        , top_p=top_p
        , frequency_penalty=frequency_penalty
        , presence_penalty=presence_penalty) 

    return completion


@app.put("/math/{operation}")
def math(operation: str, n: List[float|int]) -> int|float:
    """
    This function performs basic math operations on a list of numbers.
    The operation must be one of: add, subtract, multiply.
    The numbers must be passed in as a list.
    """
    
    if operation == "add":
        return sum(n)
    
    if operation == "subtract":
        val = n.pop(0)
        for i in n:
            val -= i
        return val

    if operation == "multiply":
        return np.prod(n)

