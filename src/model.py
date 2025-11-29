from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def load_model():
    """
    Carica il modello pre-addestrato HuggingFace per l'analisi del sentiment.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
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
