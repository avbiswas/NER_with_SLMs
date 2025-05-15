# NER with Small Language Models

This project implements Named Entity Recognition (NER) and intent classification using small language models. It includes tools for dataset generation, model training, and inference for both classification and causal language modeling tasks.

## Project Structure

```
small_lm/
├── classification/          # BERT-based classification models
│   ├── train_bert.py       # Training script for classification
│   ├── infer_bert.py       # Inference script for classification
│   ├── data_loader.py      # Data loading utilities
│   └── default_bert.py     # Default BERT model configuration
├── causal_slms/            # Causal language models
│   ├── train_slm.py        # Training script for causal LM
│   ├── infer_slm.py        # Inference script for causal LM
│   ├── data_loader.py      # Data loading utilities
│   └── default_lm.py       # Default LM configuration
├── datasets/               # Generated and processed datasets
├── trained_models/         # Saved model checkpoints
├── generate_dataset.py     # Dataset generation script
└── requirements.txt        # Project dependencies
```

## Dataset Generation

The `generate_dataset.py` script creates synthetic training data for email conversation analysis. It generates examples with:
- Named entities (DATE, TIME, NAME, EMAIL) in BIO format
- Intent classification (inquiry, cancel)
- Realistic email scheduling scenarios

To generate a dataset:
```bash
python generate_dataset.py
```

The script will:
1. Generate examples using GPT-4.1-MINI
2. Save the dataset in JSON format to the `datasets/` directory
3. Include system prompt and model information in the output
4. Create a .env file and insert your OPENAI_API_KEY in there

## Classification Model

### Training
The classification model uses BERT for token classification and intent recognition.

To train the model:
```bash
python classification/train_bert.py
```

Key features:
- Token classification for NER
- Intent classification
- Configurable model parameters
- Automatic model checkpointing

### Inference
To run inference with the trained model:
```bash
python classification/infer_bert.py path/to/model"
```

## Causal Language Model

### Training
The causal language model is trained to generate and understand email-related text.

To train the model:
```bash
python causal_slms/train_slm.py
```

Features:
- Causal language modeling
- Configurable model architecture
- Training progress tracking
- Model checkpointing

### Inference
To generate text with the trained model:
```bash
python causal_slms/infer_slm.py path/to/model"
```

## Requirements

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

1. Create a `.env` file in the project root
2. Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage Examples

### Dataset Generation
```python
# Generate 50 examples
python generate_dataset.py
```


# Future work

- Generate more data covering other entity tags
- Run a local model server to do inferencing
- Currently the inference example is hardcoded in the infer files
- Classification method uses existing distill-bert-ner model, which does not support sentence classication (required for intent detection)