#!/usr/bin/env python3
"""Configuration utility for xLSTM-Solace packages."""

import json
from xlstm_solace_torch.config_loader import list_available_configs, load_config

def show_all_configs():
    """Display all available configurations."""
    print("🔧 Available xLSTM-Solace-Torch Configurations:\n")
    
    configs = list_available_configs()
    for config_name in configs:
        try:
            config = load_config(config_name)
            print(f"📋 {config_name}")
            print(f"   Description: {config.get('description', 'No description')}")
            
            # Show kernel settings
            kernel_config = config.get('kernel_configuration', {})
            if kernel_config:
                print(f"   Kernels: {kernel_config.get('chunkwise_kernel', 'N/A')} | {kernel_config.get('step_kernel', 'N/A')}")
            
            # Show model settings
            model_settings = config.get('model_defaults', config.get('model_7b', {}))
            if model_settings:
                print(f"   Model: {model_settings.get('embedding_dim', 'N/A')}d x {model_settings.get('num_blocks', 'N/A')} blocks")
            
            print()
        except Exception as e:
            print(f"❌ {config_name}: Error loading - {e}\n")

if __name__ == "__main__":
    show_all_configs()
