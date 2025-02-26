import os
from openai import OpenAI
import datetime
import asyncio

API_KEY = os.environ.get('OPENROUTER_API_KEY')
MODEL = "deepseek/deepseek-r1:free"

# Initialize the client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

# Enter a prompt here
sample_prompt = [
    {"role": "user", "content": "Tell me a knock knock joke."},
    ]

# Track time
start = datetime.datetime.now(datetime.timezone.utc)
print(f"Instant response\n")

# Basic prompt and completion.
completion = client.chat.completions.create(
#   extra_headers={
#     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   },
    extra_body={},
    model=MODEL,
    messages=sample_prompt,
    #n=5, # this is supposed to be the number of completions to generate, but I only get 1 completion.
)


print(completion.choices[0].message.content)

end = datetime.datetime.now(datetime.timezone.utc)
elapsed = round((end - start).total_seconds() / 60, 1)
print(f"\nElapsed time for basic prompt: {elapsed} minutes\n")
print('-'.join('' for x in range(100)))
print(f"Streaming response\n")
start = datetime.datetime.now(datetime.timezone.utc)

# NOTE: Streaming responses is probably what you should do all the time so users
# get a sense of realtime interaction / reduced latency.
# Asynchronous stream response function
async def stream_response(model: str = MODEL, messages: list = sample_prompt):
    """
    Stream responses from a chat completion model.

    Args:
        model (str): The model to use for generating responses. Defaults to MODEL.
        messages (list): A list of messages to send to the model. Defaults to sample_prompt.

    Yields:
        str: The content of each chunk received from the model's response stream.
    """
    messages: list = sample_prompt):
    async def fetch_stream():
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            yield chunk

    async for chunk in fetch_stream():
        if chunk.choices:
            print(chunk.choices[0].delta.content or "", end="")

asyncio.run(stream_response())

end = datetime.datetime.now(datetime.timezone.utc)
elapsed = round((end - start).total_seconds() / 60, 1)
print(f"\nElapsed time for streaming: {elapsed} minutes")


"""
Extra stuff:
completion
{'id': 'gen-1740371868-E87n4F0lMqNPzAIsIP2O',
 'choices': [Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Knock, knock.  \n*Who’s there?*  \nBoo.  \n*Boo who?*  \nDon’t cry, it’s just a joke! 😄  \n\n*(Alternatively, if you need a second option:)*  \nKnock, knock.  \n*Who’s there?*  \nLettuce.  \n*Lettuce who?*  \nLettuce in—it’s freezing out here! ❄️  \n\nWhich one do you prefer? 😊', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None), native_finish_reason='stop')],
 'created': 1740371868,
 'model': 'deepseek/deepseek-r1',
 'object': 'chat.completion',
 'service_tier': None,
 'system_fingerprint': None,
 'usage': CompletionUsage(completion_tokens=706, prompt_tokens=10, total_tokens=716, completion_tokens_details=None, prompt_tokens_details=None),
 '_request_id': None}

stream.__dict__
{'response': <Response [200 OK]>,
 '_cast_to': openai.types.chat.chat_completion_chunk.ChatCompletionChunk,
 '_client': <openai.OpenAI at 0x107112480>,
 '_decoder': <openai._streaming.SSEDecoder at 0x107c10bc0>,
 '_iterator': <generator object Stream.__stream__ at 0x10797ce80>,
 '__orig_class__': openai.Stream[openai.types.chat.chat_completion_chunk.ChatCompletionChunk]}

for c in stream:
    print(c)
ChatCompletionChunk(id='gen-1740371529-bbS0yrclHIgcIiMXpWkz', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None, native_finish_reason=None)], created=1740371529, model='deepseek/deepseek-r1', object='chat.completion.chunk', service_tier=None, system_fingerprint=None, usage=None, provider='Chutes')
ChatCompletionChunk(id='gen-1740371529-bbS0yrclHIgcIiMXpWkz', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None, native_finish_reason=None)], created=1740371529, model='deepseek/deepseek-r1', object='chat.completion.chunk', service_tier=None, system_fingerprint=None, usage=None, provider='Chutes')
ChatCompletionChunk(id='gen-1740371529-bbS0yrclHIgcIiMXpWkz', choices=[Choice(delta=ChoiceDelta(content='', function_call=None, refusal=None, role='assistant', tool_calls=None), finish_reason=None, index=0, logprobs=None, native_finish_reason=None)], created=1740371529, model='deepseek/deepseek-r1', object='chat.completion.chunk', service_tier=None, system_fingerprint=None, usage=None, provider='Chutes')
...



"""