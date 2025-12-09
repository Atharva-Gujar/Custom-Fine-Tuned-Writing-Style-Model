custom model/
├── data/
│   ├── raw_samples.txt          # Your writing samples (input)
│   └── training_data.json       # Processed training data
├── models/
│   ├── checkpoints/             # Training checkpoints
│   ├── final/                   # Final LoRA adapter
│   └── merged/                  # Merged model (optional)
├── logs/
│   └── training.log             # Training logs
├── results/
│   └── evaluation_report.txt    # Evaluation results
├── prepare_dataset.py           # Dataset preparation script
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── merge_model.py              # Model merging script
├── app.py                       # Gradio web interface
├── deploy.py                    # HuggingFace deployment
└── requirements.txt             # Python dependencies
```

## 🎯 Training Tips

1. **Dataset Quality**: More diverse samples = better results
2. **Sample Length**: Aim for 100-500 words per sample
3. **Training Time**: Expect 1-3 hours on CPU, 15-30 min on GPU
4. **Epochs**: Start with 3 epochs, increase if underfitting
5. **Batch Size**: Reduce if running out of memory

## 🔧 Troubleshooting

**Out of Memory?**
- Reduce `--batch_size` to 1 or 2
- Reduce `--max_length` to 512

**Training Too Slow?**
- Use GPU if available
- Reduce dataset size for testing
- Use smaller base model (TinyLlama)

**Poor Results?**
- Add more training samples
- Increase epochs to 5-10
- Adjust learning rate

## 📊 Example Results

After training on 50 writing samples:
- Base model perplexity: 12.5
- Fine-tuned perplexity: 4.2
- Style consistency: 85%

## 🎯 What This Proves

- **Model Training**: Fine-tuning with LoRA for efficient training
- **Dataset Preparation**: Custom dataset creation and preprocessing
- **Evaluation**: Loss tracking and quality metrics
- **Merging**: LoRA adapter merging with base model
- **Deployment**: HuggingFace Spaces hosting + local inference

## 🚀 Features

- **Easy Local Running**: Optimized for consumer hardware (even M1/M2 Macs)
- **Efficient Training**: Uses LoRA to reduce memory requirements by 90%
- **Your Writing Style**: Learns from your text samples
- **Interactive Interface**: Gradio web UI for easy testing
- **Cloud Deployment**: One-command HuggingFace Spaces deployment

## 📋 Prerequisites

```bash
# Python 3.10 or higher
python3 --version

# At least 8GB RAM (16GB recommended)
# CUDA GPU optional (will use CPU/MPS if unavailable)
