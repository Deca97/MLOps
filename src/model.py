from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FINE_TUNED_MODEL_PATH = "models/twitter-roberta-finetuned"

def load_model():
    """
    Carica il modello più aggiornato:
    - Se esiste il modello fine-tuned, lo carica
    - Altrimenti carica il modello pre-addestrato HuggingFace
    """
    if os.path.exists(FINE_TUNED_MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)

    model.eval()
    return tokenizer, model

def predict_sentiment(texts):
    """
    Restituisce le predizioni di sentiment ('positive', 'neutral', 'negative') per una lista di testi.
    """
    tokenizer, model = load_model()
    preds = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt")
            output = model(**encoded)
            scores = torch.nn.functional.softmax(output.logits, dim=1)
            label_id = torch.argmax(scores).item()
            if label_id == 0:
                preds.append("negative")
            elif label_id == 1:
                preds.append("neutral")
            else:
                preds.append("positive")
    return preds
