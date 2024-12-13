from reasonchain.llm_models.model_manager import ModelManager
from transformers import AutoTokenizer

from datasets import Dataset

# Download the model from Hugging Face
model_name = "distilgpt2"


model_path = ModelManager.download_model(model_name=model_name, save_path="models")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Initialize ModelManager with the downloaded model
manager = ModelManager(api="custom", custom_model_path=model_path)
# Add padding token if not already present
print('Before Padding: ',tokenizer.pad_token)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
    manager.custom_model.resize_token_embeddings(len(tokenizer))
    #tokenizer.pad_token = tokenizer.eos_token
print('After Padding: ',tokenizer.pad_token)

def preprocess_data(examples):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    inputs["labels"] = inputs["input_ids"].clone()  # Labels are the same as input_ids
    return inputs


# Prepare datasets
train_data = {"text": ["SQL optimization involves indexing.", "Avoid SELECT * queries."]}
train_dataset = Dataset.from_dict({"text": train_data["text"]})
train_dataset = train_dataset.map(preprocess_data, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])



# Fine-tune the model
output_dir = "fine_tuned_model"
manager.fine_tune(train_dataset, val_dataset=None, output_dir=output_dir, num_train_epochs=5)

# Reinitialize ModelManager with the fine-tuned model
manager = ModelManager(api="custom", custom_model_path=output_dir)

# Generate a response using the fine-tuned model
response = manager.generate_response(api='custom',prompt="What are best practices for SQL optimization?")
print(f"Response:\n{response}")
