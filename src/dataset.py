# # # import torch
# # # from torch.utils.data import Dataset
# # # import json

# # # class MultimodalDataset(Dataset):
# # #     def __init__(self, file_path, tokenizer):
# # #         self.data = [json.loads(line) for line in open(file_path)]
# # #         self.tokenizer = tokenizer

# # #     def __len__(self):
# # #         return len(self.data)

# # #     # def __getitem__(self, idx):
# # #     #     item = self.data[idx]
# # #     #     text_inputs = self.tokenizer(item["text"], truncation=True, return_tensors="pt", padding="max_length")
# # #     #     label = self.tokenizer(item["response"], truncation=True, return_tensors="pt", padding="max_length")

# # #     #     return {
# # #     #         "text_input": text_inputs["input_ids"].squeeze(),
# # #     #         "facial_emotion": torch.tensor([item["facial_emotion"]["confidence"]]),
# # #     #         "speech_emotion": torch.tensor([item["speech_emotion"]["confidence"]]),
# # #     #         "speaking": torch.tensor([1 if item["speaking"] else 0]),
# # #     #         "label": label["input_ids"].squeeze()
# # #     #     }

# # #     def __getitem__(self, idx):
# # #         item = self.data[idx]
# # #         # Ensure the input text is tokenized and converted into input_ids
# # #         tokenized = self.tokenizer(
# # #             item["text"],
# # #             truncation=True,
# # #             max_length=self.max_length,
# # #             padding="max_length",
# # #             return_tensors="pt",
# # #         )
        
# # #         # Return input_ids and attention_mask (required by transformers)
# # #         return {
# # #             "input_ids": tokenized["input_ids"].squeeze(0),
# # #             "attention_mask": tokenized["attention_mask"].squeeze(0),
# # #             "labels": tokenized["input_ids"].squeeze(0),  # Assuming causal language modeling
# # #         }
# # import torch
# # from torch.utils.data import Dataset
# # import json

# # class MultimodalDataset(Dataset):
# #     def __init__(self, file_path, tokenizer, max_length=128):
# #         self.data = [json.loads(line) for line in open(file_path)]
# #         self.tokenizer = tokenizer
# #         self.max_length = max_length  # Define max_length as an instance variable

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         item = self.data[idx]
# #         # Tokenize input text
# #         tokenized = self.tokenizer(
# #             item["text"],
# #             truncation=True,
# #             max_length=self.max_length,
# #             padding="max_length",
# #             return_tensors="pt",
# #         )

# #         # Prepare the labels
# #         labels = self.tokenizer(
# #             item["response"],
# #             truncation=True,
# #             max_length=self.max_length,
# #             padding="max_length",
# #             return_tensors="pt",
# #         )

# #         # Return tokenized input_ids and labels
# #         return {
# #             "input_ids": tokenized["input_ids"].squeeze(0),
# #             "attention_mask": tokenized["attention_mask"].squeeze(0),
# #             "labels": labels["input_ids"].squeeze(0),  # Assuming causal language modeling
# #         }
# import torch
# from torch.utils.data import Dataset
# import json

# class MultimodalDataset(Dataset):
#     def __init__(self, file_path, tokenizer, max_length=128):
#         self.data = [json.loads(line) for line in open(file_path)]
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]

#         # Prepare the full text
#         input_text = f"User: {item['text']}\nAssistant:"
#         target_text = f" {item['response']}"
#         full_text = input_text + target_text

#         # Tokenize the full text
#         tokenized = self.tokenizer(
#             full_text,
#             truncation=True,
#             max_length=self.max_length,
#             padding="max_length",
#             return_tensors="pt",
#         )

#         input_ids = tokenized['input_ids'].squeeze(0)
#         attention_mask = tokenized['attention_mask'].squeeze(0)

#         # Create labels
#         labels = input_ids.clone()

#         # Mask the input tokens
#         input_text_length = len(self.tokenizer.encode(input_text, add_special_tokens=False))
#         labels[:input_text_length] = -100  # Ignore the input tokens

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'labels': labels,
#         }
import torch
from torch.utils.data import Dataset
import json

class MultimodalDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = [json.loads(line) for line in open(file_path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract parameters
        facial_emotion = item.get('facial_emotion', {})
        speech_emotion = item.get('speech_emotion', {})
        speaking = item.get('speaking', False)
        text = item.get('text', '')
        response = item.get('response', '')

        # Prepare the input text
        input_text = (
            f"Facial Emotion: {facial_emotion.get('emotion', 'unknown')} "
            f"(confidence: {facial_emotion.get('confidence', 0.0)})\n"
            f"Speech Emotion: {speech_emotion.get('emotion', 'unknown')} "
            f"(confidence: {speech_emotion.get('confidence', 0.0)})\n"
            f"Speaking: {speaking}\n"
            f"User: {text}\n"
            f"Assistant:"
        )

        target_text = f" {response}"
        full_text = input_text + target_text

        # Tokenize the full text
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        # Create labels, masking the input tokens
        labels = input_ids.clone()
        input_text_length = len(self.tokenizer.encode(input_text, add_special_tokens=False))
        labels[:input_text_length] = -100  # Ignore input tokens in loss computation

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
