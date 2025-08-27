#!/usr/bin/env python
"""
Test the downloaded xLSTM-7B model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def test_model():
    print("Loading xLSTM-7B model...")
    
    # Load model and tokenizer
    model_path = "./xlstm_7b_model"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Get model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count/1e9:.1f}B parameters")
    
    # Test prompts
    prompts = [
        "The xLSTM architecture is",
        "Machine learning models are becoming",
        "The future of AI will"
    ]
    
    print("\nGenerating text...")
    print("-" * 60)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        gen_time = time.time() - start
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated}")
        print(f"Time: {gen_time:.2f}s")
    
    print("\n✓ Model test complete!")

if __name__ == "__main__":
    test_model()