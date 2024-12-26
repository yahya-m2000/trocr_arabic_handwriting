from transformers import TrOCRProcessor
import json

# paths
MODEL_PATH = "microsoft/trocr-base-handwritten"
SAVE_PATH = "./fine_tuned_trocr_arabic"
VOCAB_PATH = f"{SAVE_PATH}/vocab.json"
TOKEN_PATH = "./fine_tuned_trocr_arabic/tokenizer.json"

# processor = TrOCRProcessor.from_pretrained(MODEL_PATH)


# test_text = "هذا نص تجريبي"
# tokens = processor.tokenizer.tokenize(test_text)
# print("Tokens:", tokens)

# if not tokens or tokens[0] == "<unk>":
#     print("Tokenizer does not handle Arabic well. Training is required.")

from tokenizers import ByteLevelBPETokenizer, 

# Initialize and train the tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

tokenizer.train_from_iterator(["هذا نص تجريبي", "مثال آخر"], vocab_size=32000, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])

# # Save the tokenizer
# tokenizer.save_model(SAVE_PATH)
print("Start Token ID:", tokenizer.cls_token_id)
print("Padding Token ID:", tokenizer.pad_token_id)
print("End Token ID:", tokenizer.eos_token_id)
print("Unknown Token ID:", tokenizer.unk_token_id)
