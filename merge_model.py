"""
Model Merging Script - Merge LoRA adapters with base model
Creates a standalone model for faster inference
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_model():
    """Merge LoRA adapters with base model"""
    print("=" * 60)
    print("🔀 MERGING LORA MODEL")
    print("=" * 60)
    
    # Load config
    config_file = Path('models/final/training_config.json')
    if not config_file.exists():
        raise FileNotFoundError("No trained model found. Run train.py first!")
    
    with open(config_file) as f:
        config = json.load(f)
    
    base_model_name = config['base_model']
    
    print(f"\n📖 Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    
    print("🔗 Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, 'models/final')
    
    print("🔀 Merging adapters with base model...")
    merged_model = model.merge_and_unload()
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Save merged model
    output_dir = Path('models/merged')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    config['merged'] = True
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Model merged successfully!")
    print(f"📁 Merged model saved to: {output_dir}")
    print("\nBenefits of merged model:")
    print("  - Faster inference (no adapter overhead)")
    print("  - Easier deployment (single model file)")
    print("  - Compatible with standard transformers code")
    print("\nUsage:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print("=" * 60)


if __name__ == "__main__":
    merge_lora_model()
