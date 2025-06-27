#!/usr/bin/env python3
"""
Ollama Connection Test Script
This script helps diagnose Ollama setup issues for the Financial Risk Assessment app.
"""

import requests
import json
import sys
import time

def test_ollama_connection():
    """Test basic Ollama connection"""
    print("🔍 Testing Ollama Connection...")
    print("=" * 50)
    
    base_url = "http://localhost:11434"
    
    try:
        # Test 1: Check if Ollama service is running
        print("1. Testing Ollama service...")
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            print("✅ Ollama service is running")
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            print(f"📋 Available models: {', '.join(models) if models else 'None'}")
            return models
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return []
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("💡 Make sure Ollama is running: ollama serve")
        return []
    except requests.exceptions.Timeout:
        print("❌ Connection timeout")
        print("💡 Ollama may be starting up, please wait and try again")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []

def test_model_availability(models, target_model="llama3.1:8b-instruct-q4_K_M"):
    """Test if required model is available"""
    print(f"\n2. Testing model availability...")
    
    if target_model in models:
        print(f"✅ Target model '{target_model}' is available")
        return True
    else:
        print(f"❌ Target model '{target_model}' not found")
        print(f"💡 Pull the model: ollama pull {target_model}")
        
        # Suggest alternatives
        alternatives = [
            "llama3.2:3b-instruct-q4_K_M",  # Lighter alternative
            "llama3.1:70b-instruct-q4_K_M"  # Heavier alternative
        ]
        
        available_alternatives = [m for m in alternatives if m in models]
        if available_alternatives:
            print(f"🔄 Available alternatives: {', '.join(available_alternatives)}")
        
        return False

def test_model_functionality(model_name="llama3.1:8b-instruct-q4_K_M"):
    """Test model functionality with a simple prompt"""
    print(f"\n3. Testing model functionality...")
    
    base_url = "http://localhost:11434"
    
    try:
        # Simple test prompt
        test_data = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 2+2? Answer with just the number."
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 10
            }
        }
        
        print(f"📤 Sending test prompt to {model_name}...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/api/chat",
            json=test_data,
            timeout=30
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('message', {}).get('content', 'No response')
            print(f"✅ Model responded successfully")
            print(f"📝 Response: '{answer.strip()}'")
            print(f"⏱️ Response time: {response_time:.2f} seconds")
            return True
        else:
            print(f"❌ Model request failed with status {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Model request timed out")
        print("💡 The model may be loading or your system may need more resources")
        return False
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_financial_prompt(model_name="llama3.1:8b-instruct-q4_K_M"):
    """Test with a financial analysis prompt similar to the app"""
    print(f"\n4. Testing financial analysis prompt...")
    
    base_url = "http://localhost:11434"
    
    financial_prompt = """
    Analyze this simple loan application. Use <|thinking|> to show your reasoning.
    
    Applicant: John Doe
    Income: $5000/month
    Requested loan: $20,000
    Credit score: 720
    Employment: 3 years
    
    Provide a brief risk assessment.
    """
    
    try:
        test_data = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial risk analyst. Provide clear, step-by-step analysis."
                },
                {
                    "role": "user",
                    "content": financial_prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 500
            }
        }
        
        print(f"📤 Sending financial analysis prompt...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/api/chat",
            json=test_data,
            timeout=60
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('message', {}).get('content', 'No response')
            
            print(f"✅ Financial analysis completed")
            print(f"⏱️ Response time: {response_time:.2f} seconds")
            
            # Check for thinking tokens
            if "<|thinking|>" in answer:
                print("🧠 Thinking tokens detected - reasoning structure working")
            else:
                print("⚠️ No thinking tokens found - may need better prompting")
            
            # Show preview of response
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"📝 Response preview: {preview}")
            
            return True
        else:
            print(f"❌ Financial analysis failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Financial analysis test failed: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("🏦 Financial Risk Assessment - Ollama Diagnostic Tool")
    print("=" * 55)
    
    # Test 1: Basic connection
    models = test_ollama_connection()
    if not models:
        print("\n❌ Basic connection failed. Please fix Ollama setup before continuing.")
        print("\n📋 Setup Instructions:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Start Ollama: ollama serve")
        print("3. Pull a model: ollama pull llama3.1:8b-instruct-q4_K_M")
        sys.exit(1)
    
    # Test 2: Model availability
    target_model = "llama3.1:8b-instruct-q4_K_M"
    model_available = test_model_availability(models, target_model)
    
    if not model_available:
        # Try with available models
        if models:
            print(f"\n🔄 Trying with first available model: {models[0]}")
            target_model = models[0]
        else:
            print("\n❌ No models available. Please pull a model first.")
            sys.exit(1)
    
    # Test 3: Basic functionality
    basic_test = test_model_functionality(target_model)
    if not basic_test:
        print(f"\n❌ Basic model test failed for {target_model}")
        sys.exit(1)
    
    # Test 4: Financial analysis
    financial_test = test_financial_prompt(target_model)
    
    # Summary
    print("\n" + "=" * 55)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 55)
    
    if basic_test and financial_test:
        print("✅ All tests passed! Your Ollama setup should work with the Financial Risk Assessment app.")
        print(f"🎯 Recommended model: {target_model}")
    elif basic_test:
        print("⚠️ Basic functionality works, but financial analysis had issues.")
        print("💡 The app should still work, but responses may vary in quality.")
    else:
        print("❌ Setup has issues. Please review the errors above.")
    
    print(f"\n🚀 Start the app with: streamlit run app.py")
    print(f"🌐 Then visit: http://localhost:8501")

if __name__ == "__main__":
    main()