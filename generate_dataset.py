import asyncio
import dotenv
import uuid
import json
import os
from openai import AsyncOpenAI

# Make sure to set the OPENAI_API_KEY environment variable in your .env file
dotenv.load_dotenv()
generating_model = "gpt-4.1-mini-2025-04-14"
total_examples = 50
examples_per_batch = 10


def extract_json(text: str) -> list[dict]:
    first_bracket = text.find("[")
    last_bracket = text.rfind("]") + 1
    json_str = text[first_bracket:last_bracket]
    return json.loads(json_str)


system_prompt = """
You are a data generator for a token classification and intent recognition task.

Generate examples of user utterances that might occur in email conversation.
The number of examples will be specified by the user.
For each example, produce a JSON object with the following fields:

1. "text": The original user utterance as a natural sentence.
2. "labels": A list of objects where each object includes:
   - "word": the exact word from the text, properly tokenized
   - "label": the entity label using BIO format
     - Use the following BIO tags:
       * B-DATE / I-DATE
       * B-TIME / I-TIME
       * B-NAME / I-NAME
       * B-EMAIL / I-EMAIL

3. "intent": A label for the overall intent of the utterance. Use one of the following:

   - "inquiry" – the user is asking about meeting details
   - "cancel" – the user is trying to cancel a meeting

4. Traditional "O" tokens are not required to generate labels for. We will assume that unlabelled tokens are "O".
Guidelines:
- Tokenize text realistically (e.g., "I'd like to..." → ["I", "'d", "like", "to", ...])
- Include only email scheduling-related examples
- Use realistic, human-like phrasing
- Ensure a balanced variety of intents

Output format:
[
  {
    "text": "Let's schedule a meeting at 7 PM next Friday with john@gmail.com.",
    "labels": [
      ...
      {"word": "john@gmail.com", "label": "B-EMAIL"}
      {"word": "7", "label": "B-TIME"}
      {"word": "PM", "label": "I-TIME"}
      {"word": "next", "label": "B-DATE"}
      {"word": "Friday", "label": "I-DATE"}
    ],
    "intent": "inquiry"
  },
  ...
]

Ensure that you don't generate the label for the "O" tokens.
Shuffle the available entities to generate a more diverse dataset.
"""

client = AsyncOpenAI()


async def generate_batch(temperature: float = 0.7):
    response = await client.chat.completions.create(
        model=generating_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Generate {examples_per_batch} examples of user utterances that might occur in email conversation.",
            },
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


async def generate_dataset(num_batches: int = 10):

    all_batches = await asyncio.gather(*[generate_batch() for _ in range(num_batches)])

    dataset = []
    for batch in all_batches:
        dataset.extend(extract_json(batch))

    return dataset


if __name__ == "__main__":
    num_batches = total_examples // examples_per_batch
    dataset = asyncio.run(generate_dataset(num_batches))
    random_id = str(uuid.uuid4().hex[-4:])
    final_dataset = {
        "system_prompt": system_prompt,
        "model": generating_model,
        "dataset": dataset,
    }
    os.makedirs("datasets", exist_ok=True)
    with open(f"datasets/dataset_{random_id}.json", "w") as f:
        json.dump(final_dataset, f, indent=4)
    print(f"Dataset saved to datasets/dataset_{random_id}.json")
