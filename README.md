# 📩 SMS Spam / Ham Classifier

A production-ready NLP pipeline that classifies SMS messages as spam or legitimate (ham). The project covers the full text classification lifecycle — from EDA and preprocessing to exhaustive experiment search, model evaluation, and deployment via an interactive Streamlit web app.

---

## 📌 Problem Statement

Given a raw SMS message, predict whether it is **spam** or a legitimate message (**ham**).  
This is a **supervised binary text classification** problem trained on ~5,500 real-world SMS records.

---

## 🗂️ Project Structure

```
spam-ham-classifier/
│
├── app.py                                      # Streamlit web application
├── data/
│   └── spam.csv                                # Raw dataset (~5.5k records)
├── model/
│   └── best_spam_model.pkl                     # Serialised model artifact (pipeline + metadata + metrics)
├── notebook/
│   └── spam-ham-classifier.ipynb               # End-to-end ML notebook
├── requirements.txt                            # Python dependencies
└── README.md
```

---

## 📊 Dataset

| Property       | Detail                                                                          |
|----------------|---------------------------------------------------------------------------------|
| Source         | [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| Rows (raw)     | 5,572                                                                           |
| Features       | 2 (`Category`, `Message`)                                                       |
| Target         | `Category` — `ham` (0) or `spam` (1)                                           |
| Class balance (raw) | ham: 4,825 · spam: 747 · imbalance ratio ≈ 6.5×                           |

After duplicate removal in preprocessing (as in the notebook), the working dataset is:
- **Rows (deduplicated):** 5,157
- **Class balance (deduplicated):** ham: 4,516 · spam: 641 · imbalance ratio ≈ 7.05×

**Feature descriptions:**

| Feature    | Type        | Description                       |
|------------|-------------|-----------------------------------|
| `Message`  | Text        | Raw SMS message body              |
| `Category` | Binary      | `ham` (legitimate) or `spam`      |

---

## 🔬 Methodology

### 1. Data Preprocessing
- Removed **duplicate rows**
- **Target encoding**: `ham → 0`, `spam → 1`
- **Text cleaning**: lowercasing, removal of special characters and digits
- Built-in preprocessing inside sklearn vectorizers for train/inference consistency
- **Stopword removal** using English stopword filtering in vectorizers

### 2. Feature Extraction
Both `CountVectorizer` and `TfidfVectorizer` were explored across multiple configurations:

| Parameter       | Values tested                          |
|-----------------|----------------------------------------|
| Analyser        | `word`, `char`                         |
| N-gram range    | `(1,1)`, `(1,2)`, `(2,4)`              |
| Max features    | 5,000 · 10,000 · None (all)            |
| Count binary    | `True` / `False` (CountVectorizer only)|
| TF-IDF sublinear| `True` / `False` (TfidfVectorizer only)|

### 3. Model Selection — Grid Search
The notebook uses **Stratified K-Fold cross-validation** and ranks candidates by **PR-AUC** (primary metric for imbalanced data).

Classifiers benchmarked:

| Variant            | Notes                                    |
|--------------------|------------------------------------------|
| `MultinomialNB`    | Count / TF-IDF word features             |
| `ComplementNB`     | Designed for imbalanced text corpora     |
| `BernoulliNB`      | Binary feature presence signals          |

Alpha values searched: `0.01`, `0.1`, `1.0`

### 4. Final Evaluation
The best pipeline (selected by CV PR-AUC) was retrained on the full training set and evaluated once on the held-out test set (80/20 stratified split):

| Metric    | Score       |
|-----------|-------------|
| PR-AUC    | **primary** |
| ROC-AUC   | reported    |
| F1        | reported    |
| Precision | reported    |
| Recall    | reported    |
| Accuracy  | reported    |

Evaluation includes a **confusion matrix**, **PR curve**, and **error analysis** of misclassified messages ranked by model confidence.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation

```bash
git clone https://github.com/amr-salah-shaheen/spam-ham-classifier.git
cd spam-ham-classifier
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Re-train the Model

Open and run all cells in the notebook:

```bash
jupyter notebook "notebook/spam-ham-classifier.ipynb"
```

Re-running the notebook regenerates the serialized model artifact in the `model/` folder.

---

## 🖥️ Web App

The Streamlit app accepts a raw SMS message and returns a real-time spam/ham prediction.

**Input:** Free-text SMS or email message  
**Output:** `SPAM` (red) or `HAM` (green) prediction label

The app loads the saved `.pkl` model artifact and uses its embedded scikit-learn `Pipeline` for inference, so training and prediction stay consistent.

---

## 🛠️ Tech Stack

| Category          | Libraries                              |
|-------------------|----------------------------------------|
| Data Processing   | `pandas`, `numpy`                      |
| NLP               | `nltk`                                 |
| Machine Learning  | `scikit-learn`                         |
| Web App           | `streamlit`                            |
| Visualisation     | `matplotlib`, `seaborn`                |

---
