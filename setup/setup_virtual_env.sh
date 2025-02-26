#!/bin/bash
# Run this script from the root directory.
# You may need to change permissions to run this script with: (check permissions with: ls -la setup/)
# chmod +x setup/setup_virtual_env.sh

# Install pyenv to set up virtual environment
brew update
brew install pyenv
brew install pyenv-virtualenv

# Install the python version
# See what's installed with: pyenv versions
pyenv install --skip-existing 3.12.7

# Create virtual environment 
pyenv virtualenv 3.12.7 sandbox
# Create a .python-version file in the root of the project to autoload the virtual environment
pyenv local sandbox

pip install -r setup/requirements.txt

# Set up the kernel for Jupyter
python -m ipykernel install --user --name=sandbox
