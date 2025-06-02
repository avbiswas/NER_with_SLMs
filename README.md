# NER with Small Language Models

This project implements Named Entity Recognition (NER) and intent classification using small language models. It includes tools for dataset generation, model training, and inference for both classification and causal language modeling tasks. The project now features a **Streamlit web interface** for easy interaction with all functionality.

## Features

✨ **Web Interface**: Interactive Streamlit application for all operations
🤖 **Dual Model Support**: Both BERT-based classification and causal language models
📊 **Dataset Generation**: Automated dataset creation using OpenAI's GPT models
🔍 **Real-time Inference**: Web-based inference for trained models
📦 **Package Structure**: Proper Python package with setup.py
🌐 **Remote LLM Support**: Integration with cloud-based language models

## Quick Start

### 1. Installation

Install the package and dependencies:
```bash
# Clone the repository
git clone <your-repo-url>
cd small_lm

# Install the package
pip install -e .
```

### 2. Environment Setup

Create a `.env` file in the project root:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

You will need:
- **OpenAI API Key**: For dataset generation (uses cost-effective gpt-4.1-mini)
- **Hugging Face Account**: For model downloads and uploads
  - Generate access tokens: [HF Security Tokens](https://huggingface.co/docs/hub/security-tokens)
  - Login via CLI: `huggingface-cli login`

### 3. Launch Web Interface

Start the Streamlit application:
```bash
streamlit run app.py
```

This opens a web interface with the following sections:
- **Home**: Project overview and dataset management
- **Dataset**: Generate and view training datasets
- **Train**: Train SLM and BERT models
- **Inference**: Run inference on trained models

## Project Structure

```
small_lm/
├── app.py                  # Main Streamlit application
├── streamlit_pages/        # Web interface components
│   ├── dataset_app.py      # Dataset generation and viewing
│   ├── train_app.py        # Model training interface
│   └── inference_app.py    # Inference interface
├── small_lm/               # Core package
│   ├── classification/     # BERT-based classification models
│   │   ├── train_bert.py   # Training script for classification
│   │   ├── infer_bert.py   # Inference script for classification
│   │   ├── data_loader.py  # Data loading utilities
│   │   └── default_bert.py # Default BERT model configuration
│   ├── causal_slms/        # Causal language models
│   │   ├── train_slm.py    # Training script for causal LM
│   │   ├── infer_slm.py    # Inference script for causal LM
│   │   ├── data_loader.py  # Data loading utilities
│   │   └── default_lm.py   # Default LM configuration
│   └── generate_dataset.py # Dataset generation script
├── remote_llms/            # Remote LLM integration
│   └── infer.py           # Remote inference utilities
├── datasets/              # Generated and processed datasets
├── trained_models/        # Saved model checkpoints
├── configs/               # Configuration files
├── docs/                  # Detailed documentation
├── setup.py              # Package configuration
└── requirements.txt      # Project dependencies
```

## Usage

### Web Interface (Recommended)

1. **Launch the app**: `streamlit run app.py`
2. **Generate Dataset**: Use the Dataset tab to create training data
3. **Train Models**: Use the Train tab for SLM or BERT training
4. **Run Inference**: Use the Inference tab to test trained models

### Command Line Interface

#### Dataset Generation
```bash
python small_lm/generate_dataset.py
```

#### Training Models
```bash
# Train BERT classification model
python small_lm/classification/train_bert.py

# Train causal language model
python small_lm/causal_slms/train_slm.py
```

#### Running Inference
```bash
# BERT inference
python small_lm/classification/infer_bert.py path/to/model

# SLM inference
python small_lm/causal_slms/infer_slm.py path/to/model
```

## Models

### Classification Model (BERT)
- **Base Model**: [dslim/distilbert-NER](https://huggingface.co/dslim/distilbert-NER)
- **Task**: Token classification and intent recognition
- **Features**: Fast inference, lightweight (60M parameters)
- **Output**: BIO tags for entities (DATE, TIME, NAME, EMAIL)

### Causal Language Model (SLM)
- **Base Model**: [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- **Task**: Text generation for entity recognition
- **Features**: Smallest available causal LM, chat-based format
- **Output**: Structured entity recognition responses

## Dataset Format

The dataset uses BIO (Begin-Inside-Outside) tagging for named entities:

```json
{
  "text": "Schedule a meeting at 2 PM on Friday with john@example.com",
  "labels": [
    {"word": "2", "label": "B-TIME"},
    {"word": "PM", "label": "I-TIME"},
    {"word": "Friday", "label": "B-DATE"},
    {"word": "john@example.com", "label": "B-EMAIL"}
  ],
  "intent": "inquiry"
}
```

**Supported Entities:**
- `DATE`: Dates and day references
- `TIME`: Time expressions
- `NAME`: Person names
- `EMAIL`: Email addresses

**Intents:**
- `inquiry`: Meeting scheduling requests
- `cancel`: Meeting cancellation requests

## Configuration

### Training Parameters
Key hyperparameters to adjust:
- `learning_rate`: Start with 2e-5 for BERT, 5e-5 for SLM
- `num_train_epochs`: 3-5 epochs typically sufficient
- `batch_size`: Adjust based on GPU memory
- `max_length`: 128 tokens for most email scenarios

### Dataset Generation
Customize in `generate_dataset.py`:
- Number of examples
- Entity types and frequency
- Intent distribution
- Complexity levels

## Documentation

For detailed information, see:
- [docs/README.md](docs/README.md) - Comprehensive technical documentation
- Model architecture details
- Training strategies and PEFT/LoRA explanations
- Dataset format specifications

## Dependencies

Core dependencies:
- `transformers==4.51.3` - Hugging Face transformers
- `torch==2.6.0` - PyTorch framework
- `peft==0.15.2` - Parameter-efficient fine-tuning
- `streamlit==1.45.1` - Web interface
- `openai==1.78.1` - Dataset generation
- `python-dotenv` - Environment management

## Development

The project is packaged as a proper Python package:
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Future Improvements

- ✅ Web interface for all operations
- ✅ Package structure and setup
- ✅ Remote LLM integration
- 🔄 Support for additional entity types
- 🔄 Local model server deployment
- 🔄 Batch inference capabilities
- 🔄 Model performance metrics dashboard
- 🔄 Export trained models to different formats

## Troubleshooting

**Common Issues:**
1. **GPU Memory**: Reduce batch size if CUDA out of memory
2. **API Limits**: Check OpenAI API quota for dataset generation
3. **HF Authentication**: Ensure `huggingface-cli login` is completed
4. **Model Loading**: Verify model paths in trained_models directory

**Performance Tips:**
- Use PEFT/LoRA for memory-efficient training
- Start with smaller datasets for prototyping
- Monitor validation loss to prevent overfitting
- Use the web interface for easier debugging