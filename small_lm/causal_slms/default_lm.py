from transformers import AutoModelForCausalLM, AutoTokenizer
import time

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"

device = "mps"  # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

messages = [
    {
        "role": "system",
        "content": "Extract the following entities from the text: email, name, date, time. Output as a json object. If an entity is absent, output null.",
    },
    {
        "role": "user",
        "content": "Extract entities from this text: Email me at johndoe@gmail.com tomorrow.",
    },
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

start_time = time.time()
outputs = model.generate(
    inputs, max_new_tokens=200, temperature=0.2, top_p=0.9, do_sample=True
)
end_time = time.time()
print(tokenizer.decode(outputs[0]))
print(f"Time taken: {end_time - start_time} seconds")
