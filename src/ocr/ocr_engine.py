import cv2
import pytesseract
import os

def preprocess_image(image_path):
    """
    Preprocess image for better OCR accuracy
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Failed to load image. Check file format or path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal
    gray = cv2.medianBlur(gray, 3)

    # Thresholding
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


def extract_text_from_image(image_path):
    """
    Extract text from an image using Tesseract OCR
    """
    processed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed_image)
    return text


if __name__ == "__main__":
    # Get absolute path to project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    image_path = os.path.join(BASE_DIR, "data", "raw", "sample.png")

    extracted_text = extract_text_from_image(image_path)
    print("\n📄 Extracted Text:\n")
    print(extracted_text)
