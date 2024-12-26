from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image
from jiwer import wer, cer

# Load the trained model and processor
model = VisionEncoderDecoderModel.from_pretrained("./arabic-base-trocr")
processor = TrOCRProcessor.from_pretrained("./arabic-base-trocr")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Convert to RGB if needed
    return processor(images=image, return_tensors="pt").pixel_values

# Function to calculate CER and WER
def calculate_metrics(predicted_text, ground_truth):
    cer_score = cer(ground_truth, predicted_text)
    wer_score = wer(ground_truth, predicted_text)
    return cer_score, wer_score

# Example image path and ground truth text
test_image_path = "image.png"
ground_truth_text = "سوف يزرع الفلاح البطاطس في الخريف.\nهيا نأكل الآن حتى ننتهي قبل بداية البرنامج.\nلا تحاولوا أن تتسلقوا هذه الشجرة العالية."

# Preprocess image
pixel_values = preprocess_image(test_image_path).to(device)

# Generate predictions
model.eval()
with torch.no_grad():
    outputs = model.generate(pixel_values)

# Decode the predictions
predicted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("Predicted text:", predicted_text)

# Calculate CER and WER
cer_score, wer_score = calculate_metrics(predicted_text, ground_truth_text)
print(f"Character Error Rate (CER): {cer_score}")
print(f"Word Error Rate (WER): {wer_score}")

# import json

# # Path to vocab.json
# vocab_path = "./arabic-base-trocr/vocab.json"

# # Load vocabulary
# with open(vocab_path, "r", encoding="utf-8") as f:
#     vocab = json.load(f)

# # Check for Arabic characters
# arabic_chars = [char for char in vocab.keys() if "\u0600" <= char <= "\u06FF"]
# print("Arabic characters in vocab:", arabic_chars)
# print("Number of Arabic characters:", len(arabic_chars))
