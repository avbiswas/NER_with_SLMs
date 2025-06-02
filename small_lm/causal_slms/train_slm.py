from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from .data_loader import CausalLMDataset
import os

DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
temp_dir = "temp_models/slm"
os.makedirs(temp_dir, exist_ok=True)


def train(
    base_model,
    train_filenames,
    test_filenames,
    output_dir,
    lora_r=64,
    epochs=15,
    batch_size=16,
    learning_rate=5e-4,
):
    if base_model is None:
        base_model = DEFAULT_BASE_MODEL
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_r,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = AutoModelForCausalLM.from_pretrained(base_model)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model.to(device)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_ds = CausalLMDataset(train_filenames, tokenizer, max_length=150)
    test_ds = CausalLMDataset(test_filenames, tokenizer, max_length=150)

    training_args = TrainingArguments(
        output_dir=temp_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=5,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(output_dir)  # Saves model and config
    tokenizer.save_pretrained(output_dir)  # Saves tokenizer
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train(
        base_model=DEFAULT_BASE_MODEL,
        output_dir="trained_models/slm3",
        temp_dir="temp_models/slm3",
        train_filenames=["datasets/dataset_2dae.json"],
        test_filenames=["datasets/dataset_e00a.json"],
    )
