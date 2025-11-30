from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import torch
import os
import random

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_SAVE_PATH = "models/twitter-roberta-finetuned"
SAMPLE_SIZE = 1000

def retrain_model():
    """
    Fine-tuning del modello HuggingFace su dati aggiornati del dataset tweet_eval.
    Salva il modello per predizioni future.
    """
    # Carica dataset
    dataset = load_dataset("tweet_eval", "sentiment")
    train_dataset_full = dataset["train"]

    # Campiona casualmente SAMPLE_SIZE record 
    indices = random.sample(range(len(train_dataset_full)), SAMPLE_SIZE) 
    train_dataset = train_dataset_full.select(indices)
    
    # Tokenizer e modello
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Tokenizzazione con padding fisso e truncation
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Data collator per batch dinamici
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Impostazioni trainer
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_steps=100,
        save_strategy="epoch",
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator
    )

    # Fine-tuning
    trainer.train()

    # Salvataggio modello e tokenizer
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Modello salvato in {MODEL_SAVE_PATH}")
    
if __name__ == "__main__":
    retrain_model()
