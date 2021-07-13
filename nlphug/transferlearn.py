import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer
import os

from transformers import pipeline

def screen_clear():
   # for mac and linux(here, os.name is 'posix')
   if os.name == 'posix':
      _ = os.system('clear')
   else:
      # for windows platfrom
      _ = os.system('cls')

screen_clear()

from datasets import load_dataset, DatasetDict, Dataset

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

screen_clear()

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
# tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

# tokenizer retirns a dictionary, we need this in a map
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

#apply the tokenizer map transformation to entire raw_dataet dictionary
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# sequences = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "This course is amazing!",
# ]
# batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")


# # This is new
# batch["labels"] = torch.tensor([1, 1])

# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# print(optimizer.step())