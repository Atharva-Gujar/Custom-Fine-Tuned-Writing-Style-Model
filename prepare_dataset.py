"""
Dataset Preparation Script
Converts raw writing samples into a training dataset for fine-tuning
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
    return text.strip()


def split_into_samples(text: str, min_length: int = 50) -> List[str]:
    """Split text into meaningful samples"""
    # Split by double newlines (paragraphs)
    samples = [s.strip() for s in text.split('\n\n') if s.strip()]
    
    # Filter out very short samples
    samples = [s for s in samples if len(s.split()) >= min_length]
    
    return samples


def create_instruction_dataset(samples: List[str]) -> List[Dict]:
    """
    Create instruction-following dataset
    Format: {"instruction": "...", "input": "", "output": "..."}
    """
    dataset = []
    
    instructions = [
        "Write in my style:",
        "Continue this thought:",
        "Elaborate on:",
        "Express this idea:",
        "Write about:",
    ]
    
    for idx, sample in enumerate(samples):
        # Split sample into prompt and completion
        words = sample.split()
        if len(words) > 20:
            split_point = len(words) // 3
            prompt = ' '.join(words[:split_point])
            completion = ' '.join(words[split_point:])
        else:
            prompt = f"Topic {idx + 1}"
            completion = sample
        
        instruction = instructions[idx % len(instructions)]
        
        dataset.append({
            "instruction": instruction,
            "input": prompt,
            "output": completion
        })
    
    return dataset


def main():
    print("🚀 Starting dataset preparation...")
    
    # Create directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    raw_file = data_dir / "raw_samples.txt"
    output_file = data_dir / "training_data.json"
    
    # Check if raw samples exist
    if not raw_file.exists():
        print(f"\n⚠️  Creating template file at {raw_file}")
        print("Please add your writing samples to this file.\n")
        
        template = """# Add your writing samples here
# Each paragraph should be separated by a blank line
# The more diverse your samples, the better the model will learn your style

This is a sample paragraph. Replace this with your actual writing.
Write about topics you care about in your natural voice.
Include different types of content: explanations, stories, opinions, etc.

This is another sample. Each section separated by blank lines becomes
a training example. Aim for 50-200 words per sample for best results.

Add at least 20-50 samples for good results. More is better!
"""
        raw_file.write_text(template)
        print(f"✅ Template created at {raw_file}")
        print("Add your writing samples and run this script again.")
        return
    
    # Load and process samples
    print(f"📖 Loading samples from {raw_file}...")
    raw_text = raw_file.read_text(encoding='utf-8')
    
    # Remove comments
    lines = [line for line in raw_text.split('\n') if not line.strip().startswith('#')]
    raw_text = '\n'.join(lines)
    
    # Clean text
    print("🧹 Cleaning text...")
    cleaned_text = clean_text(raw_text)
    
    # Split into samples
    print("✂️  Splitting into samples...")
    samples = split_into_samples(cleaned_text)
    
    if len(samples) < 10:
        print(f"\n⚠️  Warning: Only {len(samples)} samples found.")
        print("For best results, add at least 20-50 samples to your raw_samples.txt file.")
    
    print(f"Found {len(samples)} training samples")
    
    # Create instruction dataset
    print("🎯 Creating instruction dataset...")
    dataset = create_instruction_dataset(samples)
    
    # Save dataset
    print(f"💾 Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n✅ Dataset preparation complete!")
    print(f"\n📊 Statistics:")
    print(f"  - Total examples: {len(dataset)}")
    print(f"  - Average input length: {sum(len(d['input'].split()) for d in dataset) / len(dataset):.1f} words")
    print(f"  - Average output length: {sum(len(d['output'].split()) for d in dataset) / len(dataset):.1f} words")
    print(f"\n📁 Dataset saved to: {output_file}")
    print(f"\n🎓 Ready for training! Run: python train.py")


if __name__ == "__main__":
    main()
