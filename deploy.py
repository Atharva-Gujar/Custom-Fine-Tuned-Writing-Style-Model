"""
Deployment Script - Deploy to HuggingFace Spaces
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import argparse


def create_space_files(space_dir):
    """Create necessary files for HuggingFace Spaces"""
    
    # Create README
    readme_content = """---
title: Custom Writing Style Model
emoji: ✍️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.10.0
app_file: app.py
pinned: false
---

# Custom Writing Style Model

A fine-tuned language model that learns and replicates a personal writing style.

## Features
- Fine-tuned with LoRA for efficient training
- Interactive Gradio interface
- Customizable generation parameters

## Usage
Simply enter an instruction and topic, and the model will generate text in the learned style.
"""
    
    (space_dir / "README.md").write_text(readme_content)
    
    # Create requirements.txt for Spaces
    requirements = """transformers>=4.36.0
torch>=2.1.0
peft>=0.7.0
gradio>=4.10.0
accelerate>=0.25.0
"""
    
    (space_dir / "requirements.txt").write_text(requirements)
    
    print("✅ Space files created")


def deploy_to_spaces(space_name, private=False):
    """Deploy model to HuggingFace Spaces"""
    print("=" * 60)
    print("🚀 DEPLOYING TO HUGGINGFACE SPACES")
    print("=" * 60)
    
    # Check if model exists
    model_dir = Path('models/final')
    if not model_dir.exists():
        raise FileNotFoundError("No trained model found. Run train.py first!")
    
    # Create temp directory for space
    space_dir = Path('temp_space')
    if space_dir.exists():
        shutil.rmtree(space_dir)
    space_dir.mkdir()
    
    print("\n📦 Preparing files...")
    
    # Copy model files
    print("Copying model files...")
    shutil.copytree(model_dir, space_dir / 'models/final')
    
    # Copy app.py
    shutil.copy('app.py', space_dir / 'app.py')
    
    # Create Space files
    create_space_files(space_dir)
    
    print("\n☁️  Uploading to HuggingFace...")
    
    try:
        api = HfApi()
        
        # Create space
        print(f"Creating space: {space_name}")
        create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,
        )
        
        # Upload files
        print("Uploading files...")
        api.upload_folder(
            folder_path=str(space_dir),
            repo_id=space_name,
            repo_type="space",
        )
        
        print("\n✅ Deployment successful!")
        print("=" * 60)
        print(f"🌐 Your space is available at:")
        print(f"   https://huggingface.co/spaces/{space_name}")
        print("=" * 60)
        
        # Cleanup
        shutil.rmtree(space_dir)
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("\nMake sure you've run: huggingface-cli login")
        shutil.rmtree(space_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description='Deploy model to HuggingFace Spaces')
    parser.add_argument('--space_name', type=str, required=True,
                        help='Space name (format: username/space-name)')
    parser.add_argument('--private', action='store_true',
                        help='Make space private')
    
    args = parser.parse_args()
    
    # Validate space name
    if '/' not in args.space_name:
        raise ValueError("Space name must be in format: username/space-name")
    
    deploy_to_spaces(args.space_name, args.private)


if __name__ == "__main__":
    main()
