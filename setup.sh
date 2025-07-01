#!bin/bash
# Author:    Tobias Holmes
# Created:   06/2025
#
# Description:
# Sets up a Python virtual environment for the project and installs all required dependencies.
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
echo "🔍 Looking for requirements.txt..."
if [ ! -f requirements.txt ]; then
    echo "❌ requirements.txt not found! Please create it with your dependencies. Exiting..."
else
    echo "✅ requirements.txt found."
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
    echo "✅ Python dependencies installed."

    echo "🎉 Setup complete! Your Python virtual environment is ready to use."
fi