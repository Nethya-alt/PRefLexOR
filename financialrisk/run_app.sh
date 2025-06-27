#!/bin/bash

# Financial Risk Assessment Application Launcher
# This script sets up and runs the Streamlit application

echo "🏦 Financial Risk Assessment - PRefLexOR"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # macOS/Linux
    source venv/bin/activate
fi

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Check if Ollama is running
echo "🤖 Checking Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running"
    echo ""
    echo "Please start Ollama:"
    echo "  1. Install Ollama from https://ollama.ai"
    echo "  2. Run: ollama serve"
    echo "  3. Pull model: ollama pull llama3.1:8b-instruct-q4_K_M"
    echo ""
    read -p "Press Enter when Ollama is ready..."
fi

# Check if recommended model is available
echo "🔍 Checking for recommended model..."
if ollama list | grep -q "llama3.1:8b-instruct-q4_K_M"; then
    echo "✅ Recommended model is available"
else
    echo "⚠️  Recommended model not found"
    echo "📥 Pulling llama3.1:8b-instruct-q4_K_M..."
    ollama pull llama3.1:8b-instruct-q4_K_M
fi

# Start Streamlit application
echo "🚀 Starting Financial Risk Assessment application..."
echo "🌐 Application will open at: http://localhost:8501"
echo ""

streamlit run app.py