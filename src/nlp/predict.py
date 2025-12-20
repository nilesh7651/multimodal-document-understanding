import os
import sys
import joblib

# --------------------------------------------------
# Fix Python path
# --------------------------------------------------
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.nlp.preprocessing import TextPreprocessor


class TextPredictor:
    def __init__(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.preprocessor = TextPreprocessor()

    def predict(self, text):
        tokens = self.preprocessor.preprocess(text)
        clean_text = " ".join(tokens)

        text_tfidf = self.vectorizer.transform([clean_text])
        prediction = self.model.predict(text_tfidf)

        return prediction[0]


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    MODEL_PATH = os.path.join(MODEL_DIR, "text_classifier.pkl")
    VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    predictor = TextPredictor(MODEL_PATH, VECTORIZER_PATH)

    # -------- TEST TEXTS --------
    samples = [
        "This invoice shows the total amount due for electricity usage",
        "Resume of a machine learning engineer with Python experience",
        "Medical report indicating abnormal blood sugar levels",
        "Legal notice issued regarding breach of contract",
        "Deep learning approaches for natural language processing"
    ]

    for text in samples:
        label = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Predicted Category: {label}")
