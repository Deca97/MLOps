import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import predict_sentiment
from src.data_loader import load_sentiment_dataset

def test_model():
    # Carica il dataset
    texts, labels = load_sentiment_dataset("test")

    # Usa solo i primi 10 esempi per test rapido
    test_texts = texts[:10]
    test_labels = labels[:10]

    # Predizione dei sentiment
    preds = predict_sentiment(test_texts)

    # Stampa risultati
    for text, true_label, pred in zip(test_texts, test_labels, preds):
        print(f"Testo: {text}\nLabel vera: {true_label}, Predizione: {pred}\n")

if __name__ == "__main__":
    test_model()
