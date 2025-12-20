import os
import sys

# --------------------------------------------------
# Fix Python path
# --------------------------------------------------
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.ocr.ocr_engine import extract_text_from_image
from src.nlp.predict import TextPredictor


def run_pipeline(image_path):
    print("\n📥 Input Image:", image_path)

    # Step 1: OCR
    extracted_text = extract_text_from_image(image_path)
    print("\n🧾 OCR Extracted Text:\n")
    print(extracted_text)

    # Step 2: NLP + Classification
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    MODEL_PATH = os.path.join(MODEL_DIR, "text_classifier.pkl")
    VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    predictor = TextPredictor(MODEL_PATH, VECTORIZER_PATH)
    predicted_label = predictor.predict(extracted_text)

    print("\n✅ Predicted Document Category:", predicted_label)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Put any image with text here
    IMAGE_PATH = os.path.join(BASE_DIR, "data", "raw", "sample.png")

    run_pipeline(IMAGE_PATH)
