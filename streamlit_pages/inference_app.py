import streamlit as st
import os
import glob
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
import torch
from small_lm.classification.data_loader import LABEL_2_ID
from small_lm.causal_slms.infer_slm import infer as infer_slm
from small_lm.classification.infer_bert import infer as infer_bert


def run_slm_inference(model_path, input_text):
    """Run SLM inference on the given text"""
    output_text = infer_slm(model_path, input_text)
    return output_text


def run_bert_inference(model_path, input_text):
    results = infer_bert(model_path, input_text)
    return results


def inference_page():
    """Main inference page that handles both SLM and BERT models"""

    st.title("Inferencing")
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Selection")

        # Get available trained models
        models_dir = "trained_models"
        if os.path.exists(models_dir):
            model_dirs = [
                d
                for d in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, d))
            ]
        else:
            model_dirs = []
            st.error(f"Models directory {models_dir} not found!")

        if model_dirs:
            selected_model = st.selectbox(
                "Select Trained Model",
                model_dirs,
                help=f"📁 Choose a model from your trained models directory.",
            )
            model_path = os.path.join(models_dir, selected_model)
        else:
            st.warning("No trained models found in trained_models directory")
            selected_model = None
            model_path = None

        model_type = st.selectbox(
            "Select Model Type",
            ["slm", "bert"],
            index=0,
            help="Select the type of model you want to use for inference.",
        )

    with col2:
        st.subheader("Input Configuration")

        # Example texts based on model type
        if model_type == "slm":
            default_text = "Let's meet at 10 am on 25 May with avb@gmail.com"
            placeholder = "Enter text for entity recognition (emails, dates, times)..."
        else:
            default_text = "Let's meet at 10 am on 25 May with avb@gmail.com"
            placeholder = "Enter text for named entity recognition..."

        input_text = st.text_area(
            "Input Text",
            value=default_text,
            height=100,
            placeholder=placeholder,
            help="📝 Enter the text you want to analyze for entities.",
        )

    # Inference controls
    st.markdown("---")

    # Validation
    can_infer = bool(selected_model and input_text.strip())

    if not can_infer:
        if not selected_model:
            st.error("❌ Please select a trained model")
        if not input_text.strip():
            st.error("❌ Please enter some text for inference")

    run_inference = st.button(
        "🚀 Run Inference", disabled=not can_infer, type="primary"
    )
    # Run inference and display results
    if run_inference and can_infer:
        st.subheader("🔍 Inference Results")

        with st.spinner("Running inference..."):
            success = True
            try:
                if model_type == "slm":
                    results = run_slm_inference(model_path, input_text)
                else:
                    results = run_bert_inference(model_path, input_text)
            except Exception as e:
                success = False
                results = str(e)

        if success:
            st.success("✅ Inference completed successfully!")

            if model_type == "slm":
                st.text(results)
            else:  # BERT
                if results:
                    st.write(results)
                else:
                    st.info("No entities found in the text.")
        else:
            st.error(f"❌ Inference failed!")


def inference_slm():
    """SLM inference page"""
    inference_page(model_type="slm")


def inference_bert():
    """BERT inference page"""
    inference_page(model_type="bert")
