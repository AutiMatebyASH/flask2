# # # from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
# # # from dataset import MultimodalDataset

# # # # Load model and tokenizer
# # # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# # # model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# # # # Load datasets
# # # train_dataset = MultimodalDataset("data/processed_data/train_data.jsonl", tokenizer)
# # # val_dataset = MultimodalDataset("data/processed_data/val_data.jsonl", tokenizer)

# # # # Define training arguments
# # # training_args = TrainingArguments(
# # #     output_dir="./models/fine_tuned_model/",
# # #     evaluation_strategy="epoch",
# # #     save_strategy="epoch",
# # #     learning_rate=1e-4,
# # #     num_train_epochs=5,
# # #     per_device_train_batch_size=4,
# # #     weight_decay=0.01,
# # # )

# # # # Trainer setup
# # # trainer = Trainer(
# # #     model=model,
# # #     args=training_args,
# # #     train_dataset=train_dataset,
# # #     eval_dataset=val_dataset,
# # #     tokenizer=tokenizer,
# # # )

# # # # Train the model
# # # if __name__ == "__main__":
# # #     trainer.train()
# # #     trainer.save_model("./models/fine_tuned_model/")
# # #     print("Training complete!")
# # from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
# # from dataset import MultimodalDataset

# # # Load model and tokenizer
# # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# # model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# # # Add padding token to the tokenizer if it doesn't exist
# # if tokenizer.pad_token is None:
# #     tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# # # Update model to account for new tokens (important when modifying tokenizer)
# # model.resize_token_embeddings(len(tokenizer))

# # # Load datasets
# # train_dataset = MultimodalDataset("data/processed_data/train_data.jsonl", tokenizer)
# # val_dataset = MultimodalDataset("data/processed_data/val_data.jsonl", tokenizer)

# # # Define training arguments
# # training_args = TrainingArguments(
# #     output_dir="./models/fine_tuned_model/",
# #     evaluation_strategy="epoch",
# #     save_strategy="epoch",
# #     learning_rate=1e-4,
# #     num_train_epochs=5,
# #     per_device_train_batch_size=4,
# #     weight_decay=0.01,
# # )

# # # Trainer setup
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=train_dataset,
# #     eval_dataset=val_dataset,
# #     tokenizer=tokenizer,
# # )

# # # Train the model
# # if __name__ == "__main__":
# #     trainer.train()
# #     trainer.save_model("./models/fine_tuned_model/")
# #     print("Training complete!")

# from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
# from dataset import MultimodalDataset

# # Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# # Add padding token to the tokenizer if it doesn't exist
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# # Update model to account for new tokens (important when modifying tokenizer)
# model.resize_token_embeddings(len(tokenizer))

# # Load datasets with a defined max_length
# max_length = 128
# train_dataset = MultimodalDataset("data/processed_data/train_data.jsonl", tokenizer, max_length=max_length)
# val_dataset = MultimodalDataset("data/processed_data/val_data.jsonl", tokenizer, max_length=max_length)

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./models/fine_tuned_model/",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=1e-4,
#     num_train_epochs=5,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     weight_decay=0.01,
# )

# # Trainer setup
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
# )

# # Train the model
# if __name__ == "__main__":
#     trainer.train()
#     trainer.save_model("./models/fine_tuned_model/")
#     print("Training complete!")
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from dataset import MultimodalDataset
import torch

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # Use smaller model
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)

# Add padding token to tokenizer if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model.resize_token_embeddings(len(tokenizer))  # Adjust for new special tokens

# Load datasets
# Load datasets with updated max_length
train_dataset = MultimodalDataset("data/processed_data/train_data.jsonl", tokenizer, max_length=512)
val_dataset = MultimodalDataset("data/processed_data/val_data.jsonl", tokenizer, max_length=512)

# Verify dataset sizes
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Define training arguments with optimizations
training_args = TrainingArguments(
    output_dir="./models/fine_tuned_model/",  # Directory to save the model
    eval_strategy="epoch",  # Evaluate at the end of every epoch
    save_strategy="epoch",  # Save the model at the end of every epoch
    logging_strategy="steps",  # Log progress every few steps
    logging_steps=10,  # Log after every 10 steps
    learning_rate=1e-4,  # Learning rate
    num_train_epochs=10,  # Number of training epochs
    per_device_train_batch_size=1,  # Reduce batch size to minimize memory usage
    per_device_eval_batch_size=1,  # Reduce evaluation batch size
    gradient_accumulation_steps=8,  # Simulate larger batch size
    fp16=True,  # Enable mixed precision training
    save_total_limit=2,  # Save only the last two checkpoints
    weight_decay=0.01,  # Weight decay for regularization
    logging_dir="./logs",  # Directory for logs
    load_best_model_at_end=True,  # Load the best model after training
)

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal language modeling doesn't use masked LM
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train the model
if __name__ == "__main__":
    try:
        trainer.train()  # Start training
        trainer.save_model("./models/fine_tuned_model/")  # Save the fine-tuned model
        tokenizer.save_pretrained("./models/fine_tuned_model/")  # Save the tokenizer
        print("Training and saving complete!")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA ran out of memory. Try reducing batch size or using gradient checkpointing.")
        else:
            raise e
