import json
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer
import torch


# --- CausalLM Dataset ---
class CausalLMDataset(TorchDataset):
    def __init__(self, filenames: list[str], tokenizer, max_length=150):
        self.data = []
        for filename in filenames:
            with open(filename, "r") as f:
                self.data.extend(json.load(f)["dataset"])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format as chat messages
        system_msg = {
            "role": "system",
            "content": "You are a helpful AI assistant that can do entity recognition of emails, dates, and times.",
        }
        user_msg = {"role": "user", "content": item["text"]}
        # The target is the intent and slot labels as a string
        lines = [f"intent: {item['intent']}"]
        for label in item["labels"]:
            lines.append(f"{label['label']}: {label['word']}")
        target = "\n".join(lines) + "\n[stop]"
        assistant_msg = {"role": "assistant", "content": target}

        # Tokenize system+user only to get prefix length
        prefix_messages = [system_msg, user_msg]
        prefix_tokenized = self.tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=True,
            add_generation_prompt=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prefix_len = (prefix_tokenized[0] != self.tokenizer.pad_token_id).sum().item()

        # Tokenize full chat
        messages = [system_msg, user_msg, assistant_msg]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,  # For training, do not add generation prompt
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized[0]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        # Set labels to -100 for everything except assistant's response
        labels[: prefix_len + 1] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# --- Example usage ---
if __name__ == "__main__":
    causal_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(causal_model)
    ds = CausalLMDataset(tokenizer, split="train", max_length=150)
    for batch in ds:
        print("CausalLM batch:", batch)
        break
