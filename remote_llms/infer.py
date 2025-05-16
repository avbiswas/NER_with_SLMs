import openai
import asyncio
import dotenv

dotenv.load_dotenv()  # You need to have a .env file with OPENAI_API_KEY set.

client = openai.AsyncOpenAI()


system_prompt = """
You are a entity extraction and intent recognition model.

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

Input: 
"Let's schedule a meeting at 7 PM next Friday with john@gmail.com.",

Output format:
[
  {
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


async def generate_text(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    text = "Let's schedule a meeting at 7 PM next Friday with john@gmail.com."
    print(asyncio.run(generate_text(text)))
