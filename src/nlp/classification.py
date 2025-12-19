import os
import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------
# Fix Python path to allow src imports
# --------------------------------------------------
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.nlp.preprocessing import TextPreprocessor


class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.preprocessor = TextPreprocessor()

    def prepare_data(self, texts):
        cleaned_texts = []
        for text in texts:
            tokens = self.preprocessor.preprocess(text)
            cleaned_texts.append(" ".join(tokens))
        return cleaned_texts

    def train(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)

    def evaluate(self, X_test, y_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        predictions = self.model.predict(X_test_tfidf)

        print("Accuracy:", accuracy_score(y_test, predictions))
        print("\nClassification Report:\n")
        print(classification_report(y_test, predictions, zero_division=0))

    def save_model(self, model_path, vectorizer_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def load_model(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)


if __name__ == "__main__":

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "sample_dataset.csv")

    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(MODEL_DIR, "text_classifier.pkl")
    VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    classifier = TextClassifier()

    processed_texts = classifier.prepare_data(df["text"])

    # --------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts,
        df["label"],
        test_size=0.3,
        random_state=42,
        stratify=df["label"]
    )

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------
    classifier.train(X_train, y_train)

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------
    classifier.save_model(MODEL_PATH, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    # --------------------------------------------------
    # Evaluate Model
    # --------------------------------------------------
    classifier.evaluate(X_test, y_test)
