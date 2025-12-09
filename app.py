"""
Gradio App - Interactive web interface for the fine-tuned model
"""

import torch
import json
import gradio as gr
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class WritingStyleModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model"""
        print("🤖 Loading model...")
        
        config_file = Path('models/final/training_config.json')
        if not config_file.exists():
            raise FileNotFoundError(
                "No trained model found! Run train.py first."
            )
        
        with open(config_file) as f:
            self.config = json.load(f)
        
        base_model_name = self.config['base_model']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, 'models/final')
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def generate(self, instruction, input_text, max_length, temperature, top_p):
        """Generate text based on input"""
        # Format prompt
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "### Response:" in result:
            result = result.split("### Response:")[-1].strip()
        
        return result


# Initialize model
model = WritingStyleModel()


def generate_wrapper(instruction, input_text, max_length, temperature, top_p):
    """Wrapper for Gradio"""
    try:
        return model.generate(instruction, input_text, max_length, temperature, top_p)
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Custom Writing Style Model") as demo:
    gr.Markdown("""
    # 🎨 Custom Writing Style Model
    
    This model has been fine-tuned on your personal writing style.
    Give it a prompt and watch it generate text in your voice!
    """)
    
    with gr.Row():
        with gr.Column():
            instruction = gr.Textbox(
                label="Instruction",
                placeholder="Write in my style:",
                value="Write in my style:",
                lines=2
            )
            
            input_text = gr.Textbox(
                label="Input/Topic",
                placeholder="What should I write about?",
                lines=3
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                max_length = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Max Length"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (creativity)"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P (diversity)"
                )
            
            generate_btn = gr.Button("Generate ✨", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Generated Text",
                lines=15,
                show_copy_button=True
            )
    
    gr.Markdown("""
    ### Example Prompts:
    - **Instruction:** "Write in my style:", **Input:** "artificial intelligence and creativity"
    - **Instruction:** "Continue this thought:", **Input:** "The best way to learn is"
    - **Instruction:** "Express this idea:", **Input:** "teamwork makes the dream work"
    """)
    
    # Connect button
    generate_btn.click(
        fn=generate_wrapper,
        inputs=[instruction, input_text, max_length, temperature, top_p],
        outputs=output
    )
    
    # Example inputs
    gr.Examples(
        examples=[
            ["Write in my style:", "The future of technology", 200, 0.7, 0.9],
            ["Continue this thought:", "What makes a great leader", 200, 0.7, 0.9],
            ["Express this idea:", "Learning from failure", 200, 0.7, 0.9],
        ],
        inputs=[instruction, input_text, max_length, temperature, top_p],
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Writing Style Model Interface")
    print("=" * 60)
    print(f"Base model: {model.config['base_model']}")
    print(f"Training date: {model.config['training_date']}")
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
