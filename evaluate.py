"""
Evaluation Script - Test the fine-tuned model
Compares base model vs fine-tuned model performance
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_models():
    """Load both base and fine-tuned models"""
    print("🤖 Loading models...")
    
    config_file = Path('models/final/training_config.json')
    if not config_file.exists():
        raise FileNotFoundError("No trained model found. Run train.py first!")
    
    with open(config_file) as f:
        config = json.load(f)
    
    base_model_name = config['base_model']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    finetuned_model = PeftModel.from_pretrained(base_model, 'models/final')
    
    return base_model, finetuned_model, tokenizer, config


def generate_text(model, tokenizer, prompt, max_length=200):
    """Generate text from a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_models(base_model, finetuned_model, tokenizer):
    """Run evaluation on test prompts"""
    print("\n📊 Running evaluation...")
    
    test_prompts = [
        "Write in my style: Technology and innovation",
        "Continue this thought: The most important thing about learning",
        "Express this idea: Collaboration in modern work",
        "Write about: Personal growth and development",
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt}\n")
        
        # Base model generation
        print("Base model output:")
        base_output = generate_text(base_model, tokenizer, prompt)
        print(base_output[:300] + "...\n")
        
        # Fine-tuned model generation
        print("Fine-tuned model output:")
        finetuned_output = generate_text(finetuned_model, tokenizer, prompt)
        print(finetuned_output[:300] + "...\n")
        
        results.append({
            'prompt': prompt,
            'base_output': base_output,
            'finetuned_output': finetuned_output,
        })
    
    return results


def save_results(results, config):
    """Save evaluation results"""
    print("\n💾 Saving results...")
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / 'evaluation_report.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION REPORT - WRITING STYLE MODEL\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Model: {config['base_model']}\n")
        f.write(f"Training Date: {config['training_date']}\n\n")
        
        for i, result in enumerate(results, 1):
            f.write("=" * 80 + "\n")
            f.write(f"TEST {i}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"PROMPT:\n{result['prompt']}\n\n")
            f.write("-" * 80 + "\n")
            f.write("BASE MODEL OUTPUT:\n")
            f.write(result['base_output'] + "\n\n")
            f.write("-" * 80 + "\n")
            f.write("FINE-TUNED MODEL OUTPUT:\n")
            f.write(result['finetuned_output'] + "\n\n")
    
    print(f"✅ Results saved to {output_file}")


def main():
    print("=" * 60)
    print("🧪 MODEL EVALUATION")
    print("=" * 60)
    
    # Load models
    base_model, finetuned_model, tokenizer, config = load_models()
    
    # Evaluate
    results = evaluate_models(base_model, finetuned_model, tokenizer)
    
    # Save results
    save_results(results, config)
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"Check results/evaluation_report.txt for detailed comparison")
    print("=" * 60)


if __name__ == "__main__":
    main()
