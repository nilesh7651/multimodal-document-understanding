import re
import nltk
from nltk.tokenize import word_tokenize

# Download punkt_tab if not already
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class TextPreprocessor:
    def __init__(self):
        pass

    def preprocess(self, text):
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        return tokens