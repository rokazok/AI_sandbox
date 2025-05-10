# Sandbox

This sandbox contains small working examples of different concepts or libraries for refreshing future-me. 
Each folder is a standalone example.

1. [basic_llm_call](basic_llm_call/README.md): Python example for querying an LLM
2. [API](api/README.md): Use FastAPI to set up an API (for querying the LLM)
3. [chatbot app](chatbot_app/README.md): Chatbot in a shiny app with UI to control outputs.
4.

# Prerequisites
## Libraries
Set up the local environment with
```bash
setup/setup_virtual_env.sh
```
This installs `pyenv`, which allows us to dictate the Python version and create a virtual environment.
The script then creates a virtual environment `sandbox` and installs our dependencies.

## API keys
This repo uses an API key from [OpenRouter](https://openrouter.ai/settings/keys). Create one there, then add it to your `.bash_profile`, `.zprofile`, or `.zshenv`.

```bash
echo 'export OPENROUTER_API_KEY=<YOUR KEY HERE>' >> ~/.zprofile
# verify with
echo $OPENROUTER_API_KEY
```
(you may need to `source ~/.zprofile` in your terminal to see changes.)
