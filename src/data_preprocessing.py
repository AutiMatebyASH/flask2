import json
import os

def preprocess_data(raw_dir, processed_dir):
    # Load raw data
    with open(os.path.join(raw_dir, "facial_emotion.json")) as f:
        facial_data = json.load(f)
    with open(os.path.join(raw_dir, "speech_emotion.json")) as f:
        speech_data = json.load(f)
    with open(os.path.join(raw_dir, "transcriptions.json")) as f:
        text_data = json.load(f)

    # Combine data
    combined_data = []
    for i in range(len(text_data)):
        combined_data.append({
            "facial_emotion": facial_data[i],
            "speech_emotion": speech_data[i],
            "text": text_data[i],
            "speaking": True,  # Placeholder
            "response": "Your response here based on input"  # Add appropriate responses
        })

    # Split data into train and validation sets
    train_split = int(0.8 * len(combined_data))
    train_data = combined_data[:train_split]
    val_data = combined_data[train_split:]

    # Save processed data
    with open(os.path.join(processed_dir, "train_data.jsonl"), "w") as train_file:
        train_file.writelines([json.dumps(x) + "\n" for x in train_data])
    with open(os.path.join(processed_dir, "val_data.jsonl"), "w") as val_file:
        val_file.writelines([json.dumps(x) + "\n" for x in val_data])

if __name__ == "__main__":
    raw_dir = "data/raw_data"
    processed_dir = "data/processed_data"
    os.makedirs(processed_dir, exist_ok=True)
    preprocess_data(raw_dir, processed_dir)
    print("Data preprocessing complete!")
