import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from .data_loader import LABEL_2_ID
import torch
from transformers import pipeline
import sys


def collapse_entities(ner_results, original_text):
    if not ner_results:
        return []

    # Remove B-O entities
    filtered_results = [r for r in ner_results if r["entity"] != "B-O"]

    collapsed = []
    current_entity = None
    current_start = 0
    current_end = 0
    current_score = 0
    count = 0

    for result in filtered_results:
        entity_type = result["entity"].split("-")[
            1
        ]  # Get the entity type (TIME, DATE, etc.)

        if result["entity"].startswith("B-") or current_entity is None:
            # If we have a previous entity, save it
            if current_entity is not None:
                collapsed.append(
                    {
                        "entity": current_entity,
                        "text": original_text[current_start:current_end],
                        "start": current_start,
                        "end": current_end,
                        "score": current_score / count if count > 0 else 0,
                    }
                )

            # Start new entity
            current_entity = entity_type
            current_start = result["start"]
            current_end = result["end"]
            current_score = result["score"]
            count = 1
        else:
            # Continue current entity
            current_end = result["end"]
            current_score += result["score"]
            count += 1

    # Add the last entity
    if current_entity is not None:
        collapsed.append(
            {
                "entity": current_entity,
                "text": original_text[current_start:current_end],
                "start": current_start,
                "end": current_end,
                "score": current_score / count if count > 0 else 0,
            }
        )

    return collapsed


def infer(model_path, input_text):
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    id2label = {i: l for l, i in LABEL_2_ID.items()}
    model.config.id2label = id2label
    model.config.label2id = LABEL_2_ID
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(input_text)
    return ner_results


def main():
    model_path = sys.argv[1]
    example = "Let's meet at 10 am on 25 May with avb@gmail.com"
    results = infer(model_path, example)
    collapsed_results = collapse_entities(results, example)
    print(collapsed_results)


if __name__ == "__main__":
    main()
