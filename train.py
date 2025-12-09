"""
Training Script - Fine-tune Llama model with LoRA
Optimized for local execution with minimal memory usage
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import argparse


@dataclass
class ModelConfig:
    """Configuration for model and training"""
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for local training
    # Alternative models (uncomment to use):
    # base_model: str = "meta-llama/Llama-2-7b-chat-hf"  # Requires more memory
    # base_model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    
    max_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10


def setup_directories():
    """Create necessary directories"""
    dirs = ['models/checkpoints', 'models/final', 'logs', 'data']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_and_prepare_dataset(tokenizer, config):
    """Load and tokenize the dataset"""
    print("📖 Loading dataset...")
    
    data_file = "data/training_data.json"
    if not Path(data_file).exists():
        raise FileNotFoundError(
            f"{data_file} not found. Run prepare_dataset.py first!"
        )
    
    dataset = load_dataset('json', data_files=data_file, split='train')
    print(f"Loaded {len(dataset)} examples")
    
    # Format as instruction-following
    def format_instruction(example):
        text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        return {'text': text}
    
    dataset = dataset.map(format_instruction)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=config.max_length,
            padding='max_length',
        )
    
    print("🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized_dataset


def create_lora_model(base_model, config):
    """Create model with LoRA adapters"""
    print(f"🤖 Loading base model: {config.base_model}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    print("✅ LoRA model created!")
    model.print_trainable_parameters()
    
    return model


def train_model(model, tokenizer, dataset, config):
    """Train the model"""
    print("🎓 Starting training...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='models/checkpoints',
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        logging_dir='logs',
        report_to='none',  # Disable wandb for simplicity
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    start_time = datetime.now()
    print(f"⏰ Training started at {start_time.strftime('%H:%M:%S')}")
    
    trainer.train()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"✅ Training completed in {duration}")
    
    return trainer


def save_model(model, tokenizer, config):
    """Save the final model"""
    print("💾 Saving model...")
    
    output_dir = Path('models/final')
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    config_dict = {
        'base_model': config.base_model,
        'max_length': config.max_length,
        'lora_r': config.lora_r,
        'lora_alpha': config.lora_alpha,
        'training_date': datetime.now().isoformat(),
    }
    
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"✅ Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train writing style model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--base_model', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help='Base model to fine-tune')
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Create config
    config = ModelConfig(
        base_model=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    print("=" * 60)
    print("🚀 CUSTOM WRITING STYLE MODEL TRAINING")
    print("=" * 60)
    print(f"Base model: {config.base_model}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_and_prepare_dataset(tokenizer, config)
    
    # Create model
    model = create_lora_model(config.base_model, config)
    
    # Train
    trainer = train_model(model, tokenizer, dataset, config)
    
    # Save
    save_model(model, tokenizer, config)
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("  1. Evaluate: python evaluate.py")
    print("  2. Run locally: python app.py")
    print("  3. Deploy: python deploy.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
