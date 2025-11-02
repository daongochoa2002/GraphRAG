import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    pipeline
)
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

class FreeLLM:
    """Free LLM wrapper using Hugging Face transformers"""
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.device = device
        
        # Default to a good free model if none specified
        if model_name is None:
            model_name = "microsoft/DialoGPT-medium"
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            print(f"Loading model: {self.model_name}")
            
            # For text generation models
            if "gpt" in self.model_name.lower() or "llama" in self.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            # For T5/FLAN models
            elif "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
            
            # Fallback to a simple text generation pipeline
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                self.tokenizer = self.pipeline.tokenizer
                self.model = self.pipeline.model
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model {self.model_name}: {e}")
            print("🔄 Falling back to a smaller model...")
            
            # Fallback to a very small model that should always work
            try:
                self.model_name = "gpt2"
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                self.tokenizer = self.pipeline.tokenizer
                self.model = self.pipeline.model
                print(f"✅ Fallback model {self.model_name} loaded successfully")
            except Exception as fallback_error:
                raise Exception(f"Failed to load any model: {fallback_error}")
    
    def generate(self, prompt: str, max_length: int = 500, temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        try:
            # Prepare the prompt
            if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
                # T5 models work better with specific prefixes
                if not prompt.startswith("Please"):
                    prompt = f"Please answer: {prompt}"
            
            # Generate response
            if self.pipeline.task == "text2text-generation":
                # For T5/FLAN models
                response = self.pipeline(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=1
                )[0]['generated_text']
            else:
                # For GPT-style models
                response = self.pipeline(
                    prompt,
                    max_new_tokens=min(max_length, 200),  # Limit for smaller models
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )[0]['generated_text']
                
                # Remove the original prompt from response
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating text: {e}")
            return f"Generated response for: {prompt[:50]}..."
    
    def __call__(self, messages: List[Dict[str, str]]) -> "MockResponse":
        """Make the class callable like OpenAI's ChatCompletion"""
        
        # Extract the last user message
        user_message = ""
        system_message = ""
        
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "")
                content = message.get("content", "")
            else:
                # Handle LangChain message objects
                role = getattr(message, 'type', 'user')
                content = getattr(message, 'content', str(message))
            
            if role == "system":
                system_message = content
            elif role == "user" or role == "human":
                user_message = content
        
        # Combine system and user messages
        if system_message and user_message:
            prompt = f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
        elif user_message:
            prompt = f"User: {user_message}\n\nAssistant:"
        else:
            prompt = "Please provide a helpful response."
        
        # Generate response
        response_text = self.generate(prompt, max_length=400, temperature=0.3)
        
        return MockResponse(response_text)

class MockResponse:
    """Mock response object to mimic OpenAI's response format"""
    
    def __init__(self, content: str):
        self.content = content.strip()

# Alternative free models you can try:
FREE_MODELS = {
    "small_gpt": "gpt2",
    "medium_gpt": "microsoft/DialoGPT-medium", 
    "flan_t5_small": "google/flan-t5-small",
    "flan_t5_base": "google/flan-t5-base",
    "tiny_llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

def get_free_llm(model_key: str = "small_gpt", device: str = "cpu") -> FreeLLM:
    """Get a free LLM instance"""
    model_name = FREE_MODELS.get(model_key, model_key)
    return FreeLLM(model_name, device)

# Usage example:
if __name__ == "__main__":
    # Test the free LLM
    llm = get_free_llm("small_gpt")
    
    # Test with simple prompt
    response = llm.generate("What is artificial intelligence?")
    print(f"Response: {response}")
    
    # Test with message format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain machine learning in simple terms."}
    ]
    
    mock_response = llm(messages)
    print(f"Chat Response: {mock_response.content}")
