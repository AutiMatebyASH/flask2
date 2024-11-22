# # # # from transformers import AutoModelForCausalLM, AutoTokenizer

# # # # # Load fine-tuned model and tokenizer
# # # # model = AutoModelForCausalLM.from_pretrained("./models/fine_tuned_model/")
# # # # tokenizer = AutoTokenizer.from_pretrained("./models/fine_tuned_model/")

# # # # def generate_response(facial_emotion, speech_emotion, text, speaking):
# # # #     inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
# # # #     response_ids = model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)
# # # #     return tokenizer.decode(response_ids[0], skip_special_tokens=True)

# # # # # Example usage
# # # # if __name__ == "__main__":
# # # #     print(generate_response(
# # # #         {"emotion": "sad", "confidence": 0.8},
# # # #         {"emotion": "neutral", "confidence": 0.9},
# # # #         "I feel lonely.",
# # # #         True
# # # #     ))
# # # from transformers import AutoModelForCausalLM, AutoTokenizer
# # # import torch

# # # # Load fine-tuned model and tokenizer
# # # model = AutoModelForCausalLM.from_pretrained("./models/fine_tuned_model/").to("cuda" if torch.cuda.is_available() else "cpu")
# # # tokenizer = AutoTokenizer.from_pretrained("./models/fine_tuned_model/")

# # # def generate_response(facial_emotion, speech_emotion, text, speaking):
# # #     inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
# # #     response_ids = model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)
# # #     return tokenizer.decode(response_ids[0], skip_special_tokens=True)

# # # # Example usage
# # # if __name__ == "__main__":
# # #     print(generate_response(
# # #         {"emotion": "sad", "confidence": 0.8},
# # #         {"emotion": "neutral", "confidence": 0.9},
# # #         "I feel lonely.",
# # #         True
# # #     ))
# # from transformers import AutoModelForCausalLM, AutoTokenizer
# # import torch

# # # Load fine-tuned model and tokenizer
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model = AutoModelForCausalLM.from_pretrained("./models/fine_tuned_model/").to(device)
# # tokenizer = AutoTokenizer.from_pretrained("./models/fine_tuned_model/")

# # def generate_response(facial_emotion, speech_emotion, text, speaking):
# #     # Truncate input if it exceeds model's max length
# #     if len(tokenizer.encode(text)) > 1024:  # Adjust based on model's context length
# #         print("Warning: Input text is too long and will be truncated.")
# #         text = tokenizer.decode(tokenizer.encode(text)[:1024])

# #     # Tokenize input
# #     inputs = tokenizer(
# #         text,
# #         return_tensors="pt",
# #         padding="max_length",
# #         truncation=True,
# #         max_length=1024
# #     ).to(device)

# #     # Generate response
# #     response_ids = model.generate(
# #         inputs["input_ids"],
# #         attention_mask=inputs["attention_mask"],  # Ensure attention mask is passed
# #         max_new_tokens=50,  # Generate up to 50 new tokens
# #         num_beams=5,
# #         early_stopping=True,
# #         pad_token_id=tokenizer.pad_token_id
# #     )
# #     return tokenizer.decode(response_ids[0], skip_special_tokens=True)

# # # Example usage
# # if __name__ == "__main__":
# #     print(generate_response(
# #         {"emotion": "sad", "confidence": 0.8},
# #         {"emotion": "neutral", "confidence": 0.9},
# #         "I feel lonely.",
# #         True
# #     ))
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load fine-tuned model and tokenizer
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = AutoModelForCausalLM.from_pretrained("./models/fine_tuned_model/").to(device)
# tokenizer = AutoTokenizer.from_pretrained("./models/fine_tuned_model/")

# def generate_response(facial_emotion, speech_emotion, text, speaking):
#     # Prepare the input text
#     input_text = f"User: {text}\nAssistant:"

#     # Tokenize the input text
#     inputs = tokenizer(
#         input_text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=1024,
#     ).to(device)

#     # Generate response
#     response_ids = model.generate(
#         inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_new_tokens=50,
#         num_beams=5,
#         early_stopping=True,
#         pad_token_id=tokenizer.pad_token_id
#     )

#     # Extract the generated tokens (excluding the input)
#     generated_tokens = response_ids[:, inputs['input_ids'].shape[-1]:]
#     response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
#     return response.strip()

# # Example usage
# if __name__ == "__main__":
#     print(generate_response(
#         {"emotion": "sad", "confidence": 0.8},
#         {"emotion": "neutral", "confidence": 0.9},
#         "I feel lonely.",
#         True
#     ))
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained("./models/fine_tuned_model/").to(device)
tokenizer = AutoTokenizer.from_pretrained("./models/fine_tuned_model/")

def generate_response(facial_emotion, speech_emotion, text, speaking):
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

    # Tokenize the input text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    # Generate response
    response_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2  # Prevent repetition
    )

    # Extract the generated response
    generated_tokens = response_ids[:, inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return response.strip()

# Example usage
if __name__ == "__main__":
    print(generate_response(
        facial_emotion={"emotion": "sad", "confidence": 0.95},
        speech_emotion={"emotion": "sad", "confidence": 0.9},
        text="I am feeling very low today.",
        speaking=True
    ))
