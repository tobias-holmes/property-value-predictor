#!bin/bash
# Author:    Tobias Holmes
# Created:   2025-07
#
# Description:
# Sets up a Python virtual environment for the project and installs all required dev dependencies.
# Adds src to PYTHONPATH
#
# Usage:
# Source this script to set up the environment and activate it in the current shell.:
#  $ source setup.sh
# Alternatively, run it directly to set up the environment:
#  $ bash setup.sh
##############################


# Create a virtual environment for Python
if [ -d ".venv" ]; then
    echo "⚠️  Existing virtual environment found. Removing..."
    rm -rf .venv
    echo "✅  Old virtual environment removed."
fi
echo "🐍 Setting up new Python venv..."
python -m venv .venv
echo "✅  Virtual environment created."

# Activate the virtual environment
echo "🚀 Activating the virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated."

# Install Python dependencies
echo "🔍 Looking for requirements-dev.txt..."
if [ ! -f requirements-dev.txt ]; then
    echo "❌ requirements-dev.txt not found! Please create it with your dependencies. Exiting..."
else
    echo "✅ requirements-dev.txt found."
    echo "📦 Installing Python dependencies..."
    pip install -r requirements-dev.txt
    echo "✅ Python dependencies installed."

    echo "🎉 Setup complete! Your Python virtual environment is ready to use."
fi

# Add src to PYTHONPATH
echo "Adding src to PYTHONPATH"
export PYTHONPATH=./src
echo "Done. :-)"