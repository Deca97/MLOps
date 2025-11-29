from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_SAVE_PATH = "models/twitter-roberta-finetuned"

def retrain_model():
    """
    Fine-tuning del modello HuggingFace su dati aggiornati del dataset tweet_eval.
    Salva il modello per predizioni future.
    """
    # Carica dataset
    dataset = load_dataset("tweet_eval", "sentiment")
    train_dataset = dataset["train"]
    
    # Tokenizer e modello
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Impostazioni trainer
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train
    )

    # Fine-tuning
    trainer.train()

    # Salvataggio modello
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Modello salvato in {MODEL_SAVE_PATH}")
    
if __name__ == "__main__":
    retrain_model()
