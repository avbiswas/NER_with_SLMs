from transformers import AutoModelForCausalLM, AutoTokenizer
from data_loader import CausalLMDataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

base_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "mps"

model = AutoModelForCausalLM.from_pretrained(base_model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
tokenizer = AutoTokenizer.from_pretrained(base_model)
output_dir = "trained_models/slm2"
temp_dir = "temp_models/slm2"


def main():
    train_ds = CausalLMDataset(tokenizer, split="train", max_length=150)
    test_ds = CausalLMDataset(tokenizer, split="test", max_length=150)

    training_args = TrainingArguments(
        output_dir=temp_dir,
        learning_rate=5e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=15,
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
