from datasets import load_dataset

def load_sentiment_dataset(split="train"):
    """
    Carica il dataset TweetEval per sentiment analysis.
    """
    dataset = load_dataset("tweet_eval", "sentiment")[split]
    # Restituisce lista di testi e labels
    texts = [item["text"] for item in dataset]
    labels = [item["label"] for item in dataset]
    return texts, labels
