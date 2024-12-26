# This script is used to fine-tune the Nougat model on the Arabic image to markdown dataset.
# The dataset is loaded using the datasets library and split into training and testing datasets.
# The model is then fine-tuned using the Seq2SeqTrainer class.
# The model is saved after training.
# The script will benefit from a machine with multiple GPUs to run efficiently.

from transformers import (
    NougatProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrOCRProcessor
)
from datasets import load_dataset
import torch
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import warnings

# paths
MODEL_PATH = "microsoft/trocr-base-handwritten"
SAVE_PATH = "./fine_tuned_trocr_arabic"
VOCAB_PATH = f"{SAVE_PATH}/vocab.json"

warnings.filterwarnings("ignore")

# accelerate launch --multi_gpu --num_processes 4 finetune_nougat.py

processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    # attn_implementation={"decoder": "flash_attention_2", "encoder": "eager"},
)
context_length = model.decoder.config.max_position_embeddings
torch_dtype = model.dtype
print(f"Context length: {context_length}")
print(f"Model dtype: {torch_dtype}")
print(model.decoder)

# Measure model size and number of parameters
model_size = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Model size: {model_size}")

dataset = load_dataset("Fakhraddin/khatt")
# dataset["train"] = dataset["train"].select(range(500))
# print(dataset)


class DataCollatorVisionSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, samples):
        batch = {}
        pil_images = [sample["image"].convert("RGB") for sample in samples]

        # for idx, img in enumerate(pil_images):
            # print(f"Image {idx}: Mode: {img.mode}, Size: {img.size}")

        
        batch["pixel_values"] = self.processor.image_processor(
            pil_images, return_tensors="pt"
        ).pixel_values.to(torch_dtype)

        markdowns = [sample["text"] for sample in samples]
        tokenized_inputs = self.processor.tokenizer(
            markdowns,
            return_tensors="pt",
            padding="max_length",
            max_length=context_length,
            truncation=True,
        )

        labels = tokenized_inputs["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        batch["decoder_attention_mask"] = tokenized_inputs["attention_mask"]

        return batch


data_collator = DataCollatorVisionSeq2SeqWithPadding(processor)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Configure beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.num_beams = 7
model.config.no_repeat_ngram_size = 2
model.config.length_penalty = 1.5
model.config.dropout = 0.3
model.config.attention_dropout = 0.3




training_args = Seq2SeqTrainingArguments(
    output_dir=str(Path(__file__).parent / "arabic_trocr_logs"),
    eval_strategy="steps",
    save_strategy="steps",
    num_train_epochs=100,
    lr_scheduler_type="linear",
    report_to="none",
    metric_for_best_model="eval_loss",
    learning_rate=2e-5,
    warmup_steps=1000,
    save_steps=100,
    eval_steps=100,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    per_device_eval_batch_size=16,
    save_total_limit=2,
    logging_steps=5,
    dataloader_num_workers=8,  # Use multiple processes for data loading
    weight_decay = 0.01,
    predict_with_generate=True,
    gradient_checkpointing=True,  # Great for memory saving
    bf16=True,
    overwrite_output_dir=True,
    dataloader_pin_memory=True,
    load_best_model_at_end=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # To remove the warning
    ddp_find_unused_parameters=False,  # Don't know what this does
    remove_unused_columns=False,  # Gave me warnings so I set it to False
)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract valid inputs for the model
        pixel_values = inputs.get("pixel_values")
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

trainer = CustomSeq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model
trainer.train(resume_from_checkpoint=False)

trainer.state.log_history

# Save model
trainer.save_model(str(Path(__file__).parent / "arabic-base-trocr"))
processor.save_pretrained(str(Path(__file__).parent / "arabic-base-trocr"))
print("Model saved!")