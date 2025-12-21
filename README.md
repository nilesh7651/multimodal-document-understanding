# 📄 Multimodal Document Understanding System

An AI-powered web application that helps users **understand documents** by extracting text, classifying document types, and providing **confidence-aware predictions**.  
Built using **OCR + NLP + Machine Learning + Streamlit**.

---

## 🚀 Features

- 📸 Upload **Images & PDFs**
- 🔍 OCR-based text extraction
- 🧠 Hybrid document classification
  - TF-IDF + Logistic Regression (fast)
  - BERT-based classifier (accurate)
- 📊 Confidence score for predictions
- ⚡ Rule-based boosting for better accuracy
- 🔊 Read-aloud (Text-to-Speech)
- 🌐 Clean Streamlit web interface

---

## 🏗️ Architecture Overview

User
↓
Streamlit Web App
↓
OCR (Image / PDF)
↓
Hybrid NLP Classifier
├── TF-IDF (Fast)
└── BERT (Accurate)
↓
Document Type + Confidence

yaml
Copy code

---

## 📄 Supported Document Types

- Finance (Invoices, Bills, Payments)
- Legal (Notices, Court Documents)
- Medical (Reports, Prescriptions)
- Resume / CV
- Technical Documents

If confidence is low, the system safely labels the result as **“Uncertain”**.

---

## 📂 Project Structure

multimodal-document-understanding/
│
├── src/
│ ├── streamlit_app.py
│ ├── ocr/
│ │ └── ocr_engine.py
│ └── nlp/
│ └── document_classifier.py
│
├── models/
├── data/
│ ├── raw/
│ └── audio/
│
├── requirements.txt
├── .gitignore
└── README.md

yaml
Copy code

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/multimodal-document-understanding.git
cd multimodal-document-understanding
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the App
bash
Copy code
streamlit run src/streamlit_app.py
Open in browser:

arduino
Copy code
http://localhost:8501
🧪 How to Use
Upload an image or PDF

Choose classification mode:

Standard (Fast) – TF-IDF

Advanced (BERT) – More accurate

Click Analyze Document

View:

Extracted text

Document type

Confidence score

Use Read Aloud if needed

🧠 Confidence-Aware Prediction
Each prediction includes a confidence score

Low-confidence outputs are marked as Uncertain

Improves reliability and user trust

🧰 Tech Stack
Python

Streamlit

OpenCV

PyMuPDF

Scikit-learn

Transformers (BERT)

PyTorch

gTTS

⚠️ Limitations
OCR accuracy depends on document quality

BERT mode may be slow on low-memory systems

Streamlit Cloud may limit OCR support

🌍 Deployment
Designed for Streamlit Community Cloud

Can be extended to:

FastAPI backend

MERN-based frontend

SaaS deployment

