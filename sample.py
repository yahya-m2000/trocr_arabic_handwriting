import torch
import evaluate
import numpy as np
from tqdm.auto import tqdm 
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer, AdamW, get_scheduler

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)



def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create DataLoaders
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Number of epochs and total training steps
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

# Linear learning rate scheduler
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Detect GPU or use CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# Progress bar
progress_bar = tqdm(range(num_training_steps))

# Set model to training mode
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Move batch to device (GPU/CPU)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss  # Compute loss
        
        # Backward pass
        loss.backward()
        
        # Update weights and learning rate
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()  # Clear gradients
        
        # Update progress bar
        progress_bar.update(1)


# Load evaluation metric
metric = evaluate.load("glue", "mrpc")

# Set model to evaluation mode
model.eval()

for batch in eval_dataloader:
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass without gradients
    with torch.no_grad():
        outputs = model(**batch)
    
    # Get predictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    # Add batch results to metric
    metric.add_batch(predictions=predictions, references=batch["labels"])

# Compute final evaluation metrics
final_metrics = metric.compute()
print(final_metrics)






# def compute_metrics(eval_preds):
#     metric = evaluate.load("glue", "mrpc")
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)





# training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

# trainer = Trainer(
#     model,
#     training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     processing_class=tokenizer,
#     compute_metrics=compute_metrics,
# )

# trainer.train()







# config = BertConfig()
# model = BertModel.from_pretrained("bert-base-cased")
# model.save_pretrained("directory_on_my_computer")

# encoded_sequences = [
#     [101, 7592, 999, 102],
#     [101, 4658, 1012, 102],
#     [101, 3835, 999, 102],
# ]

# model_inputs = torch.tensor(encoded_sequences)

# output = model(model_inputs)

# print(output)

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokeniser = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)

# raw_inputs = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "I hate this so much!",
# ]
# inputs = tokeniser(raw_inputs, padding=True, truncation=True, return_tensors="pt")
# print(inputs)

# outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)
# torch.Size([2, 16, 768])


# classifier = pipeline("ner", device=-1)
# result = classifier(
#     "I'm a black man from an equitorial country. I currently live in London"
# )
# print(result)