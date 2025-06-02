"""
Small Language Models Package

A package for Named Entity Recognition (NER) and intent classification using small language models.
Includes tools for dataset generation, model training, and inference for both classification and causal language modeling tasks.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main functions for easy access
from . import generate_dataset

from . import classification
from . import causal_slms

# Make main functions available at package level
from .generate_dataset import main as generate_dataset_main

__all__ = [
    "generate_dataset",
    "classification",
    "causal_slms",
    "generate_dataset_main",
]
