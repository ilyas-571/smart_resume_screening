import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.features.tfidf_vectorizer import build_vectorizer
from src.data.text_cleaner import clean_text

# Absolute path to CSV relative to this file
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "raw",
    "Resume.csv"
)
print(f"Looking for dataset at: {DATA_PATH}")

# Keywords to detect tech resumes automatically
TECH_KEYWORDS = ["python", "data", "ml", "machine learning", "developer", "engineer", "software", "ai"]

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Dataset not found! Place your CSV at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = ["Resume_str", "Category"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ Column '{col}' missing. Required: {required_columns}")

    df["clean_resume"] = df["Resume_str"].apply(clean_text)

    # Automatic label assignment based on keywords
    def label_func(row):
        text = (row["Category"] + " " + row["Resume_str"]).lower()
        for kw in TECH_KEYWORDS:
            if kw in text:
                return 1
        return 0

    df["label"] = df.apply(label_func, axis=1)

    if len(df["label"].unique()) < 2:
        raise ValueError("❌ Dataset contains only one class after labeling. Add more tech resumes.")

    return df["clean_resume"], df["label"]

def train_model():
    print("🚀 Starting model training...")
    texts, labels = load_data()
    print(f"✅ Dataset loaded successfully: {len(texts)} resumes")
    print("Label distribution:", labels.value_counts())

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/resume_model.pkl")
    joblib.dump(vectorizer, "models/tfidf.pkl")
    print("✅ Model trained & saved successfully")