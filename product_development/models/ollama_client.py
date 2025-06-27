"""
Ollama client for LLM interactions
"""

import ollama
import requests
import json
import time
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b-instruct-q4_K_M"):
        self.base_url = base_url
        self.model = model
        self.client = ollama.Client(host=base_url)
        
    def check_connection(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = self.client.list()
            return [model['name'] for model in response['models']]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if specific model is available"""
        available_models = self.list_models()
        return model_name in available_models
    
    def pull_model(self, model_name: str) -> bool:
        """Pull model if not available"""
        try:
            logger.info(f"Pulling model: {model_name}")
            self.client.pull(model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = "", 
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate response using Ollama"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            start_time = time.time()
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                stream=stream
            )
            
            end_time = time.time()
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if 'message' in chunk:
                        full_response += chunk['message']['content']
                content = full_response
            else:
                content = response['message']['content']
            
            return {
                "success": True,
                "content": content,
                "model": self.model,
                "response_time": end_time - start_time,
                "token_count": len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": "",
                "model": self.model,
                "response_time": 0,
                "token_count": 0
            }
    
    def generate_streaming_response(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        """Generate streaming response for real-time display"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                stream=True
            )
            
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            logger.error(f"Failed to generate streaming response: {e}")
            yield f"Error: {str(e)}"
    
    def extract_thinking_section(self, response: str) -> Dict[str, str]:
        """Extract thinking and answer sections from response"""
        thinking_start = "<|thinking|>"
        thinking_end = "<|/thinking|>"
        
        result = {
            "thinking": "",
            "answer": "",
            "has_thinking": False
        }
        
        if thinking_start in response and thinking_end in response:
            start_idx = response.find(thinking_start) + len(thinking_start)
            end_idx = response.find(thinking_end)
            
            if start_idx < end_idx:
                result["thinking"] = response[start_idx:end_idx].strip()
                result["answer"] = response[end_idx + len(thinking_end):].strip()
                result["has_thinking"] = True
            else:
                result["answer"] = response
        else:
            result["answer"] = response
            
        return result
    
    def validate_model_setup(self) -> Dict[str, Any]:
        """Validate complete model setup"""
        validation_result = {
            "ollama_running": False,
            "model_available": False,
            "model_functional": False,
            "recommended_action": "",
            "available_models": []
        }
        
        try:
            # Check Ollama connection
            if not self.check_connection():
                validation_result["recommended_action"] = "Start Ollama service: ollama serve"
                return validation_result
            
            validation_result["ollama_running"] = True
            
            # Get available models
            try:
                validation_result["available_models"] = self.list_models()
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                validation_result["recommended_action"] = f"Error listing models: {e}"
                return validation_result
            
            # Check model availability
            if not self.is_model_available(self.model):
                validation_result["recommended_action"] = f"Pull model: ollama pull {self.model}"
                return validation_result
                
            validation_result["model_available"] = True
            
            # Test model functionality
            try:
                test_response = self.generate_response(
                    "Test prompt: What is 2+2? Answer with just the number.",
                    temperature=0.1,
                    max_tokens=50
                )
                
                if test_response["success"] and test_response["content"].strip():
                    validation_result["model_functional"] = True
                    validation_result["recommended_action"] = "All systems operational"
                else:
                    error_msg = test_response.get('error', 'Model returned empty response')
                    validation_result["recommended_action"] = f"Model test failed: {error_msg}"
                    
            except Exception as e:
                logger.error(f"Model functionality test failed: {e}")
                validation_result["recommended_action"] = f"Model test error: {e}"
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_result["recommended_action"] = f"Validation error: {e}"
            
        return validation_result