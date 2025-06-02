import streamlit as st
import os
import glob
import sys
import io
import contextlib
import threading
import time
from small_lm.causal_slms.train_slm import train as train_slm
from small_lm.causal_slms.train_slm import DEFAULT_BASE_MODEL as DEFAULT_SLM_BASE_MODEL
from small_lm.classification.train_bert import train as train_bert
from small_lm.classification.train_bert import (
    DEFAULT_BASE_MODEL as DEFAULT_BERT_BASE_MODEL,
)


class StreamlitLogCapture:
    """Capture logs and display them in Streamlit"""

    def __init__(self, log_container):
        self.log_container = log_container
        self.logs = []
        self.log_placeholder = log_container.empty()

    def write(self, text):
        if text.strip():  # Only add non-empty lines
            self.logs.append(text.strip())
            # Keep only last 50 lines to prevent too much output
            if len(self.logs) > 50:
                self.logs = self.logs[-50:]

            # Update the display
            log_text = "\n".join(self.logs)
            self.log_placeholder.code(log_text, language="text")

    def flush(self):
        pass


def run_training_with_logs(log_container, model_type, **train_kwargs):
    """Run training and capture logs in real-time"""
    try:
        # Create log capture
        log_capture = StreamlitLogCapture(log_container)

        # Redirect stdout and stderr to capture logs
        with contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(
            log_capture
        ):
            log_capture.write("🚀 Starting training...")
            log_capture.write(f"📋 Configuration: {train_kwargs}")
            log_capture.write("=" * 50)

            # Call the training function
            if model_type == "slm":
                train_slm(**train_kwargs)
            elif model_type == "bert":
                train_bert(**train_kwargs)

            log_capture.write("=" * 50)
            log_capture.write("✅ Training completed successfully!")

        return True, None

    except Exception as e:
        log_capture.write(f"❌ Training failed: {str(e)}")
        return False, str(e)


def training_page(model_type="slm"):
    st.title("🚀 Small Language Model Training")
    st.markdown("Train a causal language model using LoRA fine-tuning")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Configuration")

        # Base Model Selection
        base_model = st.text_input(
            "Base Model",
            value=(
                DEFAULT_SLM_BASE_MODEL
                if model_type == "slm"
                else DEFAULT_BERT_BASE_MODEL
            ),
            help="🤗 Hugging Face model identifier. This is the pre-trained model that will be fine-tuned. Default is SmolLM2-135M-Instruct.",
        )

        # LoRA Configuration
        lora_r = st.slider(
            "LoRA Rank (r)",
            min_value=8,
            max_value=128,
            value=64,
            step=8,
            help="🎯 LoRA rank parameter. Higher values mean more trainable parameters but increased memory usage. Typical range: 8-64.",
        )

        # Training Parameters
        st.subheader("Training Parameters")

        epochs = st.slider(
            "Number of Epochs",
            min_value=1,
            max_value=50,
            value=25,
            help="🔄 Number of complete passes through the training dataset. More epochs can improve performance but may lead to overfitting.",
        )

        batch_size = st.selectbox(
            "Batch Size",
            [4, 8, 16, 32, 64, 128],
            index=2,  # Default to 16
            help="📦 Number of samples processed before model weights are updated. Larger batches are more stable but require more memory.",
        )

        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
            help="📈 Step size for gradient descent optimization. Higher rates train faster but may overshoot optimal weights.",
        )

    with col2:
        st.subheader("Dataset Configuration")

        # Get available dataset files
        dataset_dir = "datasets"
        if os.path.exists(dataset_dir):
            dataset_files = glob.glob(os.path.join(dataset_dir, "*.json"))
            dataset_names = [os.path.basename(f) for f in dataset_files]
        else:
            dataset_names = []
            st.error(f"Dataset directory {dataset_dir} not found!")

        if dataset_names:
            train_files = st.multiselect(
                "Training Dataset Files",
                dataset_names,
                default=[dataset_names[0]] if dataset_names else [],
                help="📚 JSON files containing training data. You can select multiple files to combine datasets.",
            )

            test_files = st.multiselect(
                "Test Dataset Files",
                dataset_names,
                default=[dataset_names[1]] if len(dataset_names) > 1 else [],
                help="🧪 JSON files containing validation/test data. Used for model evaluation during training.",
            )
        else:
            st.warning("No dataset files found in ../datasets directory")
            train_files = []
            test_files = []

        # Output Configuration
        st.subheader("Output Configuration")

        output_dir = st.text_input(
            "Output Directory",
            value=(
                "trained_models/custom_slm"
                if model_type == "slm"
                else "trained_models/custom_bert"
            ),
            help="💾 Directory where the final trained model will be saved. Should be unique for each training run.",
        )

    # Training Status and Controls
    st.markdown("---")

    # Validation
    can_train = bool(train_files and test_files and output_dir)

    if not can_train:
        if not train_files:
            st.error("❌ Please select at least one training dataset file")
        if not test_files:
            st.error("❌ Please select at least one test dataset file")
        if not output_dir:
            st.error("❌ Please specify an output directory")

    # Initialize session state for training
    if "training_in_progress" not in st.session_state:
        st.session_state.training_in_progress = False

    col_train, col_info = st.columns([1, 2])

    with col_train:
        # Training button
        start_training = st.button(
            "🚀 Start Training",
            disabled=not can_train or st.session_state.training_in_progress,
            type="primary",
        )

        # Stop training button (if training is in progress)
        if st.session_state.training_in_progress:
            if st.button("⏹️ Stop Training", type="secondary"):
                st.session_state.training_in_progress = False
                st.rerun()

    with col_info:
        with st.expander("ℹ️ Training Tips", expanded=False):
            st.markdown(
                """
            **Training Tips:**
            - Start with smaller models and datasets for quick iteration
            - Monitor eval_loss to avoid overfitting
            - Use early stopping to prevent overtraining
            - Larger batch sizes are more stable but need more memory
            - Higher LoRA rank captures more information but uses more memory
            - Learning rate 5e-4 is a good starting point for most tasks
            """
            )

    # Training execution and log display
    if start_training and can_train:
        st.session_state.training_in_progress = True

        # Convert relative paths to full paths for train_filenames and test_filenames
        train_filepaths = [os.path.join(dataset_dir, f) for f in train_files]
        test_filepaths = [os.path.join(dataset_dir, f) for f in test_files]

        # Display training configuration
        with st.expander("📋 Training Configuration", expanded=True):
            st.json(
                {
                    "base_model": base_model,
                    "train_files": train_filepaths,
                    "test_files": test_filepaths,
                    "output_dir": output_dir,
                    "lora_r": lora_r,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                }
            )

        # Create log display area
        st.subheader("📊 Training Logs")
        log_container = st.container()

        # Status placeholder
        status_placeholder = st.empty()

        try:
            status_placeholder.info("🔄 Training in progress... Please wait.")

            # Run training with log capture
            success, error = run_training_with_logs(
                log_container,
                base_model=base_model,
                train_filenames=train_filepaths,
                test_filenames=test_filepaths,
                output_dir=output_dir,
                lora_r=lora_r,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_type=model_type,
            )

            if success:
                status_placeholder.success(
                    f"✅ Training completed! Model saved to {output_dir}"
                )
            else:
                status_placeholder.error(f"❌ Training failed: {error}")

        except Exception as e:
            status_placeholder.error(f"❌ Unexpected error: {str(e)}")

        finally:
            st.session_state.training_in_progress = False

    # Display training status
    elif st.session_state.training_in_progress:
        st.info("🔄 Training is currently in progress...")


def training_page_slm():
    training_page(model_type="slm")


def training_page_bert():
    training_page(model_type="bert")


def inference_page():
    st.write("Inference page")
