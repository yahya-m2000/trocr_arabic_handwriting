from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
from PIL import Image
import pandas as pd
import requests
import evaluate
import torch
import json


# paths
MODEL_PATH = "microsoft/trocr-base-handwritten"
SAVE_PATH = "./fine_tuned_trocr_arabic"
VOCAB_PATH = f"{SAVE_PATH}/vocab.json"

# ahmedheakl/arabic_khatt
# "AhmedTaha012/arabic_text_yolov8_V0.1", 

# load datasets
raw_datasets = load_dataset("Fakhraddin/khatt")
raw_datasets["train"] = raw_datasets["train"].select(range(2))
print(raw_datasets["train"][0])
data = [{"file_name": example["image"], "text": example["text"]} for example in raw_datasets["train"]]
df = pd.DataFrame(data)

print(df.head())


from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


import torch
from torch.utils.data import Dataset
from PIL import Image

class ArabicDataset(Dataset):
    def __init__(self, df, processor, max_target_length=128):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]

        # prepare image (i.e. resize + normalize)
        if isinstance(file_name, str):  # If it's a file path
            print("isInstance true")
            image = Image.open(file_name).convert("RGB")
        else:  # If it's already a PIL.Image object
            # print("isInstance false")
            image = file_name
        if image.mode != "RGB":  # Convert grayscale or other modes to RGB
            image = image.convert("RGB")
        image = image.resize((224, 224))
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


from transformers import TrOCRProcessor

# load processor
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)

# initialize datasets
train_dataset = ArabicDataset(df=train_df, processor=processor)
eval_dataset = ArabicDataset(df=test_df, processor=processor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))


# get the first example from the training dataset
encoding = train_dataset[0]

# print the keys and shapes of the tensors
for k, v in encoding.items():
    print(k, v.shape)


from PIL import Image

# Display the original image
image = train_dataset.df['file_name'][0]  # First file name from DataFrame
if isinstance(image, str):  # If it's a file path, load the image
    image = Image.open(image).convert("RGB")
image.show()  # This will display the image in a viewer

# Decode the labels
labels = encoding['labels']
# Replace -100 with the padding token ID for decoding
labels[labels == -100] = processor.tokenizer.pad_token_id
# Convert the labels to text
label_str = processor.tokenizer.decode(labels, skip_special_tokens=True)
print("Decoded Text:", label_str)

def data_collator(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.stack([example["labels"] for example in batch])
    return {"pixel_values": pixel_values, "labels": labels}


from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Configure the model
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Configure beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = False
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 1.0
model.config.num_beams = 2

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,          # Enable text generation during evaluation
    evaluation_strategy="steps",         # Evaluate every few steps
    per_device_train_batch_size=4,       # Batch size for training
    per_device_eval_batch_size=2,        # Batch size for evaluation
    fp16=True,                           # Enable mixed precision for faster training
    output_dir="./trained_trocr_arabic", # Save directory
    logging_steps=10,                    # Log every 10 steps
    save_steps=1000,                     # Save model every 1000 steps
    eval_steps=200,                      # Evaluate every 200 steps
    num_train_epochs=3,                  # Number of epochs
    save_total_limit=2,                  # Keep only the last 2 checkpoints
)


# create dataloaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

# check a batch from the DataLoader
for batch in train_dataloader:
    print(batch["pixel_values"].shape)  # expected: torch.Size([batch_size, 3, H, W])
    print(batch["labels"].shape)        # expected: torch.Size([batch_size, seq_len])
    break

# optimiser
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# progress bar
progress_bar = tqdm(range(num_training_steps))

torch.cuda.empty_cache()
# set model to training mode
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss = outputs.loss
        
        # backward pass
        loss.backward()
        
        # update optimizer and scheduler
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)


# load metric (e.g., CER: Character Error Rate)
metric = evaluate.load("cer")

# set model to evaluation mode
model.eval()

batch_processed = False

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model.generate(pixel_values=batch["pixel_values"])

    labels = batch["labels"]
    labels[labels == -100] = processor.tokenizer.pad_token_id
    predictions = processor.batch_decode(outputs, skip_special_tokens=True)
    references = processor.batch_decode(batch["labels"], skip_special_tokens=True)
    
    # skip batches with all empty references
    valid_data = [(pred, ref) for pred, ref in zip(predictions, references) if ref.strip()]
    if not valid_data:
        continue
    
    predictions, references = zip(*valid_data)
    metric.add_batch(predictions=predictions, references=references)
    batch_processed = True

if not batch_processed:
    print("No valid batches were processed during evaluation.")
else:
    final_metrics = metric.compute()
    print("Validation CER:", final_metrics)


model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)


# # load model & processor
# processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
# model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# # data_collator = DataCollatorWithPadding(tokenizer=processor)

# # extend vocab for arabic
# def extend_vocab(vocab_path):
#     arabic_characters = [chr(i) for i in range(0x0600, 0x06FF + 1)]  # Arabic Unicode range
    
#     # load existing vocabulary
#     with open(vocab_path, "r", encoding="utf-8") as f:
#         vocab = json.load(f)
    
#     # add arabic characters if not present
#     for char in arabic_characters:
#         if char not in vocab:
#             vocab[char] = len(vocab)
    
#     # save updated vocabulary
#     with open(vocab_path, "w", encoding="utf-8") as f:
#         json.dump(vocab, f, ensure_ascii=False, indent=4)
#     print("vocabulary updated with arabic characters")

# # extend the vocab and resize token embeddings
# processor.tokenizer.save_pretrained(SAVE_PATH)
# extend_vocab(VOCAB_PATH)
# model.decoder.resize_token_embeddings(len(processor.tokenizer))
# processor = TrOCRProcessor.from_pretrained(SAVE_PATH) 


# def data_collator(batch):
#     pixel_values = torch.stack([example["pixel_values"] for example in batch])
#     labels = torch.stack([example["labels"] for example in batch])
#     return {"pixel_values": pixel_values, "labels": labels}


# # preprocessing function
# def preprocess_function(example):
#     # ensure the image is a PIL.Image.Image object
#     image = example["image"]
    
#     # convert image to RGB if it’s not already in RGB mode
#     if image.mode != "RGB":
#         image = image.convert("RGB")
    
#     inputs = processor(images=image, text=example["answer"], return_tensors="pt", truncation=True)
    
#     return {
#         "pixel_values": inputs["pixel_values"].squeeze(0),  # Processed image tensor
#         "labels": inputs["labels"].squeeze(0)  # Tokenized text labels
#     }

# # apply preprocessing
# processed_datasets = raw_datasets.map(preprocess_function)
# processed_datasets.set_format(type="torch", columns=["pixel_values", "labels"])

# # couldn't find validation in the dataset so here's me trying to split the dataset and make a validation column

# processed_datasets = processed_datasets["train"].train_test_split(test_size=0.2) 
# # processed_datasets = processed_datasets.rename_columns({"test": "validation"})
# # manually rename the "test" split to "validation"
# processed_datasets["validation"] = processed_datasets.pop("test")

# # print(processed_datasets)
# print(type(processed_datasets["train"][0]["pixel_values"]))  # Check the type
# print(processed_datasets["train"][0]["pixel_values"].shape) 

# # check the first example
# # preprocessed = preprocess_function(raw_datasets["train"][0])
# # example = raw_datasets["train"][0]["image"]
# # print(type(example)) 

# # create dataloaders
# train_dataloader = DataLoader(processed_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
# eval_dataloader = DataLoader(processed_datasets["validation"], batch_size=8, collate_fn=data_collator)

# # check a batch from the DataLoader
# for batch in train_dataloader:
#     print(batch["pixel_values"].shape)  # expected: torch.Size([batch_size, 3, H, W])
#     print(batch["labels"].shape)        # expected: torch.Size([batch_size, seq_len])
#     break

# # configure padding and start tokens for decoding
# model.config.pad_token_id = processor.tokenizer.pad_token_id
# model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
# model.config.max_length = 128 
# model.config.num_beams = 4

# # optimiser
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# # scheduler
# num_epochs = 3
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )

# # progress bar
# progress_bar = tqdm(range(num_training_steps))

# # set model to training mode
# model.train()

# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         # move batch to device
#         batch = {k: v.to(device) for k, v in batch.items()}
        
#         # forward pass
#         outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
#         loss = outputs.loss
        
#         # backward pass
#         loss.backward()
        
#         # update optimizer and scheduler
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()

#         progress_bar.update(1)


# # load metric (e.g., CER: Character Error Rate)
# metric = evaluate.load("cer")

# # set model to evaluation mode
# model.eval()

# batch_processed = False

# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
    
#     with torch.no_grad():
#         outputs = model.generate(pixel_values=batch["pixel_values"])
    
#     predictions = processor.batch_decode(outputs, skip_special_tokens=True)
#     references = processor.batch_decode(batch["labels"], skip_special_tokens=True)
    
#     # skip batches with all empty references
#     valid_data = [(pred, ref) for pred, ref in zip(predictions, references) if ref.strip()]
#     if not valid_data:
#         continue
    
#     predictions, references = zip(*valid_data)
#     metric.add_batch(predictions=predictions, references=references)
#     batch_processed = True

# if not batch_processed:
#     print("No valid batches were processed during evaluation.")
# else:
#     final_metrics = metric.compute()
#     print("Validation CER:", final_metrics)

# model.save_pretrained("./trocr_arabic_model")
# processor.save_pretrained("./trocr_arabic_model")











# # load image from the IAM dataset
# url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# pixel_values = processor(image, return_tensors="pt").pixel_values
# generated_ids = model.generate(pixel_values)

# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(generated_text)

