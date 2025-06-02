from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model = "dslim/bert-base-NER"
# model = "dslim/distilbert-NER"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForTokenClassification.from_pretrained(model)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "I am going to USA on 10th June."

import time

start_time = time.time()
ner_results = nlp(example)
end_time = time.time()
print(ner_results)
print(f"Time taken: {end_time - start_time} seconds")
