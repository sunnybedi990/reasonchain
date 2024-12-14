from reasonchain.utils.lazy_imports import transformers

def fine_tune_model(model, tokenizer, train_dataset, val_dataset=None, output_dir="fine_tuned_model", num_train_epochs=3, per_device_train_batch_size=4):
    """
    Fine-tune a Hugging Face model on a specific dataset.
    :param model: The pre-trained Hugging Face model to fine-tune.
    :param tokenizer: The tokenizer associated with the model.
    :param train_dataset: Training dataset.
    :param val_dataset: Validation dataset.
    :param output_dir: Directory to save the fine-tuned model.
    :param num_train_epochs: Number of training epochs.
    :param per_device_train_batch_size: Batch size for training.
    :return: Fine-tuned model path.
    """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    try:
        # Dynamically set evaluation strategy based on val_dataset
        evaluation_strategy = "epoch" if val_dataset is not None else "no"
        training_args = transformers.TrainingArguments(
            output_dir=output_dir,
            eval_strategy=evaluation_strategy,
            learning_rate=2e-5,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            remove_unused_columns=False
        )

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )

        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned model saved to {output_dir}")
        return output_dir
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return None
