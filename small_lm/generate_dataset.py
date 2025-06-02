import asyncio
import dotenv
import uuid
import json
import os
from openai import AsyncOpenAI

# Make sure to set the OPENAI_API_KEY environment variable in your .env file
dotenv.load_dotenv()


def extract_json(text: str) -> list[dict]:
    first_bracket = text.find("[")
    last_bracket = text.rfind("]") + 1
    json_str = text[first_bracket:last_bracket]
    return json.loads(json_str)


client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


async def generate_batch(
    system_prompt, model, examples_per_batch, temperature: float = 0.7
):
    if system_prompt is None:
        raise ValueError("System prompt is not set")
    response = await client.chat.completions.create(
        model=model,
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


async def generate_dataset(
    system_prompt: str, model: str, examples_per_batch: int, num_batches: int = 10
):

    all_batches = await asyncio.gather(
        *[
            generate_batch(system_prompt, model, examples_per_batch)
            for _ in range(num_batches)
        ]
    )

    dataset = []
    for batch in all_batches:
        dataset.extend(extract_json(batch))

    return dataset


async def main(
    system_prompt: str,
    model: str = "gpt-4.1-mini-2025-04-14",
    total_examples: int = 50,
    examples_per_batch: int = 10,
):
    num_batches = max(total_examples // examples_per_batch, 1)
    dataset = await generate_dataset(system_prompt, model, num_batches)
    random_id = str(uuid.uuid4().hex[-4:])
    final_dataset = {
        "system_prompt": system_prompt,
        "model": model,
        "dataset": dataset,
    }
    print("Number of batches:", num_batches)
    print("Number of examples:", len(dataset))
    os.makedirs("datasets", exist_ok=True)
    with open(f"datasets/dataset_{random_id}.json", "w") as f:
        json.dump(final_dataset, f, indent=4)
    print(f"Dataset saved to datasets/dataset_{random_id}.json")
    return f"datasets/dataset_{random_id}.json"


if __name__ == "__main__":
    asyncio.run(main())
