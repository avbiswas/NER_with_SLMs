import json
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer
import torch

LABEL_2_ID = {
    "B-O": 0,
    "B-DATE": 1,
    "I-DATE": 2,
    "B-TIME": 3,
    "I-TIME": 4,
    "B-EMAIL": 5,
    "I-EMAIL": 6,
    "B-NAME": 7,
    "I-NAME": 8,
}


# --- BERT Token Classification Dataset ---
class TokenClassificationDataset(TorchDataset):
    def __init__(self, filenames, tokenizer, labels=LABEL_2_ID, max_length=50):
        self.data = []
        for filename in filenames:
            with open(filename, "r") as f:
                self.data.extend(json.load(f)["dataset"])

        self.tokenizer = tokenizer
        self.max_length = max_length
        # Build label2id and id2label
        self.label2id = labels
        self.id2label = {i: l for l, i in self.label2id.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        labels = item["labels"]
        # Build word-to-label mapping (start/end char -> label)
        word_spans = []
        for l in labels:
            word = l["word"]
            label = l["label"]
            # Find all occurrences of the word in text
            start = 0
            while True:
                start = text.find(word, start)
                if start == -1:
                    break
                end = start + len(word)
                word_spans.append((start, end, label))
                start = end  # move past this word

        # Sort spans by start position to ensure proper BIO tagging
        word_spans.sort(key=lambda x: x[0])

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offsets = encoding["offset_mapping"]
        label_ids = []

        # Track the current entity we're processing
        current_entity = None

        for i, (token, (start, end)) in enumerate(zip(tokens, offsets)):
            if (start, end) == (0, 0):
                label_ids.append(-100)  # Special tokens
                continue

            # Find if this token is part of any span
            token_span = None
            for span_start, span_end, label in word_spans:
                if start >= span_start and end <= span_end:
                    token_span = (span_start, span_end, label)
                    break

            if token_span is None:
                label_ids.append(self.label2id["B-O"])
                current_entity = None
            else:
                span_start, span_end, label = token_span
                entity_type = label.split("-")[-1]

                # If this is a new entity or different from current entity
                if current_entity != entity_type:
                    label_id = self.label2id["B-" + entity_type]
                    current_entity = entity_type
                else:
                    label_id = self.label2id["I-" + entity_type]

                label_ids.append(label_id)

        encoding["labels"] = label_ids
        return {
            k: torch.tensor(v) for k, v in encoding.items() if k != "offset_mapping"
        }


# --- Example usage ---
if __name__ == "__main__":
    causal_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(causal_model)
    ds = TokenClassificationDataset(tokenizer, split="train", max_length=150)
    for batch in ds:
        print("CausalLM batch:", batch)
        break
