"""
Test Notebook - Interactive testing and experimentation
Run this in Jupyter to experiment with your model
"""

# %% [markdown]
# # Custom Writing Style Model - Testing Notebook
# 
# Use this notebook to interactively test and experiment with your fine-tuned model.

# %% [markdown]
# ## Setup

# %%
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# %% [markdown]
# ## Load Model

# %%
def load_model():
    """Load the fine-tuned model"""
    config_file = Path('models/final/training_config.json')
    
    with open(config_file) as f:
        config = json.load(f)
    
    base_model_name = config['base_model']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    
    # Load fine-tuned model
    model = PeftModel.from_pretrained(base_model, 'models/final')
    model.eval()
    
    return model, tokenizer, config

model, tokenizer, config = load_model()
print(f"✅ Model loaded: {config['base_model']}")

# %% [markdown]
# ## Generation Function

# %%
def generate(instruction, input_text, max_length=200, temperature=0.7, top_p=0.9):
    """Generate text from prompt"""
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in result:
        result = result.split("### Response:")[-1].strip()
    
    return result

# %% [markdown]
# ## Quick Tests

# %%
# Test 1: Writing style
print("=== Test 1: Writing Style ===")
output = generate(
    "Write in my style:",
    "artificial intelligence and the future",
    max_length=250
)
print(output)
print()

# %%
# Test 2: Continue thought
print("=== Test 2: Continue Thought ===")
output = generate(
    "Continue this thought:",
    "The most important skill to develop is",
    max_length=200
)
print(output)
print()

# %%
# Test 3: Express idea
print("=== Test 3: Express Idea ===")
output = generate(
    "Express this idea:",
    "teamwork makes the dream work",
    max_length=200
)
print(output)
print()

# %% [markdown]
# ## Parameter Experimentation

# %%
# Compare different temperatures
instruction = "Write in my style:"
input_text = "learning and growth"

print("=== Temperature Comparison ===\n")

for temp in [0.3, 0.7, 1.0, 1.5]:
    print(f"Temperature: {temp}")
    output = generate(instruction, input_text, temperature=temp, max_length=150)
    print(output)
    print("-" * 60)
    print()

# %%
# Compare different top_p values
print("=== Top-P Comparison ===\n")

for top_p in [0.5, 0.7, 0.9, 0.95]:
    print(f"Top-P: {top_p}")
    output = generate(instruction, input_text, top_p=top_p, max_length=150)
    print(output)
    print("-" * 60)
    print()

# %% [markdown]
# ## Custom Prompts
# 
# Use this cell to test your own prompts

# %%
# Your custom test
instruction = "Write in my style:"
input_text = "YOUR TOPIC HERE"

output = generate(instruction, input_text, max_length=200, temperature=0.7)
print(output)

# %% [markdown]
# ## Batch Testing

# %%
test_cases = [
    ("Write in my style:", "technology and innovation"),
    ("Continue this thought:", "success comes from"),
    ("Express this idea:", "practice makes perfect"),
    ("Write about:", "personal development"),
]

print("=== Batch Testing ===\n")

for instruction, input_text in test_cases:
    print(f"Prompt: {instruction} '{input_text}'")
    output = generate(instruction, input_text, max_length=150)
    print(output)
    print("-" * 80)
    print()

# %% [markdown]
# ## Compare with Base Model

# %%
def compare_models(instruction, input_text):
    """Compare base model vs fine-tuned"""
    # Load base model
    base_only = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("=== BASE MODEL ===")
    with torch.no_grad():
        outputs = base_only.generate(**inputs, max_length=200, temperature=0.7)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    print("\n=== FINE-TUNED MODEL ===")
    output = generate(instruction, input_text)
    print(output)

# Test comparison
compare_models("Write in my style:", "the future of work")

# %% [markdown]
# ## Save Your Favorite Generations

# %%
generations = []

# Generate and save
output = generate("Write in my style:", "creativity and innovation")
generations.append({
    "prompt": "creativity and innovation",
    "output": output,
    "params": {"temperature": 0.7, "top_p": 0.9}
})

# Save to file
import json
with open('results/favorite_generations.json', 'w') as f:
    json.dump(generations, f, indent=2)

print("✅ Saved favorite generations!")
