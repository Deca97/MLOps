from model import predict_sentiment

if __name__ == "__main__":
    sample_texts = [
        "I love this product!",
        "I hate waiting in line.",
        "It's okay, not great."
    ]
    preds = predict_sentiment(sample_texts)
    for text, pred in zip(sample_texts, preds):
        print(f"Testo: {text} --> Sentiment: {pred}")
