from transformers import AutoTokenizer, AutoModelForTokenClassification
from .data_loader import TokenClassificationDataset, LABEL_2_ID
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
import torch
from transformers import EarlyStoppingCallback

DEFAULT_BASE_MODEL = "dslim/distilbert-NER"


def train(
    base_model,
    train_filenames,
    test_filenames,
    output_dir="trained_models/bert",
    temp_dir="temp_models/bert",
    lora_r=16,
    learning_rate=5e-4,
    batch_size=16,
    epochs=10,
):

    lora_config = LoraConfig(
        task_type="TOKEN_CLS",
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        bias="none",
        inference_mode=False,
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(base_model)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = TokenClassificationDataset(train_filenames, tokenizer, max_length=40)
    test_ds = TokenClassificationDataset(test_filenames, tokenizer, max_length=40)

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
        train_filenames=["datasets/dataset_e00a.json"],
        test_filenames=["datasets/dataset_2dae.json"],
        output_dir="trained_models/bert",
        temp_dir="temp_models/bert",
        lora_r=16,
        learning_rate=5e-4,
        batch_size=16,
        num_epochs=10,
    )
