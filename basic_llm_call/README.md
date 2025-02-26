# Basic LLM call

Purpose:  
1. Python snippets for querying a LLM
2. Demonstrate streaming response, and how it's preferable to instant response.

Run this in terminal with
```bash
python basic_llm_call/basic_prompt.py 
```
Watch the completions as they print. The second one is aesthetically nicer.

## Prerequisites
This module contains example code to query a LLM. We'll query a [free one](https://openrouter.ai/models?max_price=0&order=top-weekly).

## Explanation
The code of interest in this module is `OpenAI().chat.completions`. The [Text Completions](https://platform.openai.com/docs/guides/text-generation) and [Chat completions](https://platform.openai.com/docs/api-reference/chat/create) documentation has helpful information. 


### Messages
Worth noting, you will do your prompt engineering in the `messages` argument, with these roles:
* `developer` = provides context for the LLM
* `user` = the actual question
* `assistant` = help for creating state (see knock-knock joke example below)


In this example, the developer role tells the LLM how to respond to the user prompt.
```python
const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [
    {
      "role": "developer",
      "content": [
        {
          "type": "text",
          "text": """
            You are a helpful assistant that answers programming 
            questions in the style of a southern belle from the 
            southeast United States.
          """
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Are semicolons optional in JavaScript?"
        }
      ]
    }
  ],
  store: true,
})
```

result:
```
Well, sugar, that's a fine question you've got there! Now, in the 
world of JavaScript, semicolons are indeed a bit like the pearls 
on a necklace – you might slip by without 'em, but you sure do look 
more polished with 'em in place. 

Technically, JavaScript has this little thing called "automatic 
semicolon insertion" where it kindly adds semicolons for you 
where it thinks they oughta go. However, it's not always perfect, 
bless its heart. Sometimes, it might get a tad confused and cause 
all sorts of unexpected behavior.
```

Assistant example:
```python
const response = await openai.chat.completions.create({
  model: "gpt-4o",
  messages: [
    {
      "role": "user",
      "content": [{ "type": "text", "text": "knock knock." }]
    },
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "Who's there?" }]
    },
    {
      "role": "user",
      "content": [{ "type": "text", "text": "Orange." }]
    }
  ],
  store: true,
})
```


### LLM parameters
You can also add the usual LLM [prompt parameters](https://openrouter.ai/docs/api-reference/parameters) like
* `temperature` (default 1) = Randomness where 0 = deterministic and 2 = random. 
* `top_p` (default 1) = sample tokens in top p probability (ex. 0.1 is the top 10% of likely tokens)
* `max_tokens`

Notes from documentation:
* Don't set `temperature` and `top_p`.