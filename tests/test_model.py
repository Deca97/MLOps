import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import predict_sentiment
from src.data_loader import load_sentiment_dataset

def test_model():
    # Test predizione su frasi campione
    sample_texts = ["I love this!", "I hate this!", "It's okay."]
    preds = predict_sentiment(sample_texts)
    print("Predizioni di test:", preds)

    # Test caricamento dataset
    texts, labels = load_sentiment_dataset("train")
    print(f"Dataset caricato: {len(texts)} esempi")
    print("Esempio testo:", texts[0], "Label:", labels[0])

if __name__ == "__main__":
    test_model()
