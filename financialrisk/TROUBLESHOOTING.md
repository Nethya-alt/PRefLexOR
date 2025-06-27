# Troubleshooting Guide - Financial Risk Assessment App

This guide helps resolve common issues when running the PRefLexOR Financial Risk Assessment application.

## Quick Diagnosis

First, run the diagnostic script to identify issues:

```bash
cd financialrisk
python test_ollama.py
```

This will test your Ollama setup and provide specific guidance.

## Common Issues & Solutions

### 1. "Assessment failed: 'NoneType' object has no attribute 'assess_mortgage_risk'"

**Cause**: Ollama connection not established properly.

**Solutions**:

#### Step 1: Check Ollama Service
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start Ollama
ollama serve
```

#### Step 2: Verify Model Installation
```bash
# List installed models
ollama list

# If empty, pull the recommended model
ollama pull llama3.1:8b-instruct-q4_K_M

# Alternative lightweight model
ollama pull llama3.2:3b-instruct-q4_K_M
```

#### Step 3: Test Model Functionality
```bash
# Test the model directly
ollama run llama3.1:8b-instruct-q4_K_M "What is 2+2?"
```

### 2. Connection Refused / Cannot Connect to Ollama

**Symptoms**:
- "Connection refused" errors
- Sidebar shows "❌ Connection failed"

**Solutions**:

#### Check Ollama Installation
```bash
# Install Ollama if not installed
# Download from https://ollama.ai

# Verify installation
ollama --version
```

#### Start Ollama Service
```bash
# Start Ollama (keeps running)
ollama serve

# Or run in background (macOS/Linux)
nohup ollama serve > ollama.log 2>&1 &

# Check if running
ps aux | grep ollama
```

#### Check Port Availability
```bash
# Check if port 11434 is in use
lsof -i :11434

# If another process is using it, kill it or change port
# To use different port:
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

### 3. Model Not Found / Model Pull Fails

**Symptoms**:
- "Model not available" errors
- Empty model list in sidebar

**Solutions**:

#### Pull Recommended Model
```bash
# Primary recommendation (6GB)
ollama pull llama3.1:8b-instruct-q4_K_M

# If you have limited RAM (2.5GB)
ollama pull llama3.2:3b-instruct-q4_K_M

# If you have lots of RAM (45GB)
ollama pull llama3.1:70b-instruct-q4_K_M
```

#### Check Disk Space
```bash
# Ollama models require significant space
df -h

# Models are stored in:
# macOS: ~/.ollama/models
# Linux: /usr/share/ollama/.ollama/models
# Windows: C:\Users\%username%\.ollama\models
```

#### Manual Model Management
```bash
# Remove unused models to free space
ollama rm <model_name>

# Show model information
ollama show llama3.1:8b-instruct-q4_K_M
```

### 4. Slow Response Times / Timeouts

**Symptoms**:
- Long loading times
- Timeout errors
- App becomes unresponsive

**Solutions**:

#### Use Lighter Model
Update `config/settings.py`:
```python
OLLAMA_CONFIG = {
    "default_model": "llama3.2:3b-instruct-q4_K_M",  # Changed from 8B to 3B
    # ... rest of config
}
```

#### Increase Timeout
```python
OLLAMA_CONFIG = {
    "timeout": 300,  # Increased from 120 to 300 seconds
    # ... rest of config
}
```

#### Check System Resources
```bash
# Check memory usage
free -h  # Linux
vm_stat  # macOS

# Check CPU usage
top

# Close other applications to free resources
```

### 5. Poor Response Quality / No Thinking Tokens

**Symptoms**:
- Responses don't include `<|thinking|>` sections
- Analysis lacks detail
- Inconsistent recommendations

**Solutions**:

#### Verify Model Version
```bash
# Ensure you're using the instruct version
ollama list | grep instruct

# The model name should end with "instruct"
# Wrong: llama3.1:8b (base model)
# Right: llama3.1:8b-instruct-q4_K_M
```

#### Check Prompt Configuration
In `config/settings.py`, verify system prompts include thinking token instructions:
```python
SYSTEM_PROMPTS = {
    "mortgage": """You are an expert mortgage underwriter...
    Provide detailed, step-by-step analysis using <|thinking|> tags to show your reasoning process.
    ..."""
}
```

#### Adjust Temperature
```python
OLLAMA_CONFIG = {
    "temperature": 0.1,  # Lower = more consistent
    # ... rest of config
}
```

### 6. Permission Denied / File Access Errors

**Symptoms**:
- Cannot write to directories
- Import errors for local modules

**Solutions**:

#### Fix File Permissions
```bash
# Make scripts executable
chmod +x run_app.sh
chmod +x test_ollama.py

# Fix directory permissions
chmod -R 755 financialrisk/
```

#### Python Path Issues
```bash
# Run from correct directory
cd financialrisk
python -c "import models.ollama_client; print('Import successful')"

# If import fails, check __init__.py files exist
ls models/__init__.py utils/__init__.py config/__init__.py
```

### 7. Streamlit-Specific Issues

**Symptoms**:
- Page won't load
- Widget errors
- State management issues

**Solutions**:

#### Clear Streamlit Cache
```bash
# Clear cache and restart
streamlit cache clear
streamlit run app.py
```

#### Check Dependencies
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Check specific versions
pip list | grep streamlit
pip list | grep ollama
```

#### Browser Issues
- Try incognito/private browsing mode
- Clear browser cache
- Try different browser
- Check console for JavaScript errors

## Advanced Debugging

### Enable Debug Logging

Add to top of `app.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Application Logs

Look for error details in the Streamlit logs or terminal output.

### Memory Monitoring

```bash
# Monitor memory usage while running
watch -n 2 'ps aux | grep -E "(ollama|streamlit|python)" | head -10'
```

### Network Debugging

```bash
# Test Ollama API directly
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b-instruct-q4_K_M",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

## Environment-Specific Issues

### macOS
- May need to allow Ollama in Security & Privacy settings
- M1/M2 Macs: Ensure you have ARM-compatible models

### Windows
- Use PowerShell or Command Prompt, not Git Bash for some commands
- May need to run as Administrator
- Check Windows Defender / antivirus isn't blocking Ollama

### Linux
- Check systemctl if running as service
- Ensure sufficient /tmp space for model loading
- Verify firewall isn't blocking port 11434

## Getting Help

### Collect Debug Information

Before asking for help, collect this information:

```bash
# System information
uname -a
python --version
pip list | grep -E "(streamlit|ollama|pandas|plotly)"

# Ollama information
ollama --version
ollama list
curl -s http://localhost:11434/api/tags | jq .

# App-specific debug
cd financialrisk
python test_ollama.py > debug_output.txt 2>&1
```

### Create Minimal Test Case

If issues persist, create a minimal test:

```python
# test_minimal.py
from models.ollama_client import OllamaClient

client = OllamaClient()
print("Connection:", client.check_connection())
print("Models:", client.list_models())

response = client.generate_response("Test prompt: What is 2+2?")
print("Response:", response)
```

## Performance Optimization

### For Low-Memory Systems (< 8GB RAM)
- Use `llama3.2:3b-instruct-q4_K_M`
- Reduce `max_tokens` to 1024
- Close other applications

### For High-Memory Systems (> 16GB RAM)
- Use `llama3.1:70b-instruct-q4_K_M` for best quality
- Increase `max_tokens` to 4096
- Enable multiple concurrent requests

### For Best Performance
- Use SSD storage for Ollama models
- Ensure sufficient RAM (model size + 2GB minimum)
- Use quantized models (q4_K_M suffix) for faster inference

Remember: The most common issue is Ollama not running or the wrong model being selected. Always start with the diagnostic script!