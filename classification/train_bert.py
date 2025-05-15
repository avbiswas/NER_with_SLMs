from transformers import AutoTokenizer, AutoModelForTokenClassification
from data_loader import TokenClassificationDataset, LABEL_2_ID
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

lora_config = LoraConfig(
    task_type="TOKEN_CLS",
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    bias="none",
    inference_mode=False,
)

device = "mps"

bert_model = "dslim/distilbert-NER"
tokenizer = AutoTokenizer.from_pretrained(bert_model)
model = AutoModelForTokenClassification.from_pretrained(bert_model)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
output_dir = "trained_models/bert"
temp_dir = "temp_models/bert"


def main():
    train_ds = TokenClassificationDataset(tokenizer, split="train", max_length=40)
    test_ds = TokenClassificationDataset(tokenizer, split="test", max_length=40)

    training_args = TrainingArguments(
        output_dir=temp_dir,
        learning_rate=5e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()
    trainer.save_model(output_dir)  # Saves model and config
    tokenizer.save_pretrained(output_dir)  # Saves tokenizer
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
