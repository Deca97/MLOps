import sqlite3
import os
from datetime import datetime
from src.model import predict_sentiment
from src.data_loader import load_sentiment_dataset
from sklearn.metrics import f1_score

# Percorso database

DB_DIR = "data"
DB_FILE = "sentiment.db"
DB_PATH = os.path.join(DB_DIR, DB_FILE)

# Soglia minima accettabile per F1 macro

F1_THRESHOLD = 0.7

def init_db():
    """Crea la cartella, il database e la tabella se non esistono."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    text TEXT,
    sentiment TEXT,
    score REAL,
    f1_score REAL
    );
    """)
    conn.commit()
    conn.close()

def log_sentiment(text: str, sentiment: str, score: float, f1: float = None):
    """Salva un nuovo risultato di sentiment nel database."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO sentiments (timestamp, text, sentiment, score, f1_score)
    VALUES (?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), text, sentiment, score, f1))
    conn.commit()
    conn.close()

def monitor_dataset(n_samples: int = 50):
    """Esegue il monitoraggio sui primi n_samples del dataset di test e calcola F1 macro."""
    texts, labels = load_sentiment_dataset("test")
    sample_texts = texts[:n_samples]
    sample_labels = labels[:n_samples]

    preds = predict_sentiment(sample_texts)

    # Converte le etichette predette in numerico
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    pred_nums = [label_map[p] for p in preds]

    # Calcolo F1 macro
    f1 = f1_score(sample_labels, pred_nums, average="macro")
    print(f"F1 macro sul campione: {f1:.3f}")

    if f1 < F1_THRESHOLD:
        print(f"⚠️ F1 score sotto soglia ({F1_THRESHOLD}). Attenzione!")

    # Log di ogni predizione
    for text, pred_num in zip(sample_texts, pred_nums):
        log_sentiment(text, list(label_map.keys())[pred_num], pred_num, f1)


if __name__ == "__main__":
    init_db()
    print(f"Database pronto in {os.path.abspath(DB_PATH)}. Inizio monitoraggio…")
    monitor_dataset()
    print("Monitoraggio completato.")
