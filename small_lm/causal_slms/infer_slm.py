from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import json

CURRENT_MODEL_PATH = None
model = None
tokenizer = None


def combine_bio_tokens(entities):
    combined = []
    current_entity = None

    for entity in entities:
        entity_type = entity["type"]
        if entity_type.startswith("B-"):
            # If we have a current entity, add it to combined
            if current_entity:
                combined.append(current_entity)
            # Start new entity
            current_entity = {
                "type": entity_type[2:],  # Remove B- prefix
                "value": entity["value"],
            }
        elif entity_type.startswith("I-"):
            # Continue current entity
            if current_entity and entity_type[2:] == current_entity["type"]:
                current_entity["value"] += " " + entity["value"]
            else:
                # If no current entity or type mismatch, treat as new B- entity
                if current_entity:
                    combined.append(current_entity)
                current_entity = {"type": entity_type[2:], "value": entity["value"]}

    # Add the last entity if exists
    if current_entity:
        combined.append(current_entity)

    return combined


def infer(model_path, input_text, end_string="[stop]"):
    global CURRENT_MODEL_PATH, model, tokenizer
    print(f"Input: {model_path}")

    if CURRENT_MODEL_PATH != model_path:
        CURRENT_MODEL_PATH = model_path
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that can do entity recognition of emails, dates, and times.",
        },
        {"role": "user", "content": input_text},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    # Get the length of input tokens to slice later
    input_length = inputs.shape[1]

    outputs = model.generate(
        inputs,
        max_new_tokens=200,
        do_sample=False,
        stop_strings=[end_string],
        tokenizer=tokenizer,
    )

    # Only decode the newly generated tokens (excluding input)
    new_tokens = outputs[0][input_length:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    output_text = output_text.split(end_string)[0]

    return output_text.strip()


def main():
    model_path = sys.argv[1]
    example = "Let's meet at 10 am on 25 May with avb@gmail.com"
    output_text = infer(model_path, example)

    print(f"-------\nRaw output: {output_text} \n--------")

    lines = output_text.split("\n")
    print(f"Lines: {lines}")
    return
    output = {
        "intent": None,
        "entities": [],
        "collapsed_entities": [],
    }
    for line in lines:
        if ":" not in line:
            continue
        key = line.split(":")[0].strip()
        value = line.split(":")[1].strip()

        if key == "intent":
            output["intent"] = value
            continue
        output["entities"].append({"type": key, "value": value})

    # Combine BIO tokens into complete entities
    output["collapsed_entities"] = combine_bio_tokens(output["entities"])

    print(json.dumps(output["collapsed_entities"], indent=4))


if __name__ == "__main__":
    main()
