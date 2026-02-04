import pandas as pd
import sys
import os
from src.models.predict import predict
from src.data.resume_parser import extract_resume_data

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from models.train import train_model

if __name__ == "__main__":
    train_model()

def save_dataset_predictions():
    DATA_PATH = "data/raw/Resume.csv"
    OUTPUT_PATH = "data/output/Resume_with_scores.csv"
    
    # create output folder if missing
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    df = pd.read_csv(DATA_PATH)
    text_column = "Resume_str"  # update if your CSV column is different
    
    fit_scores = []
    parsed_data = []
    
    for text in df[text_column]:
        score = predict(text)
        fit_scores.append(score)
        parsed_data.append(extract_resume_data(text))
    
    df["fit_score"] = fit_scores
    df["name"] = [d["name"] for d in parsed_data]
    df["email"] = [d["email"] for d in parsed_data]
    df["phone"] = [d["phone"] for d in parsed_data]
    df["skills"] = [", ".join(d["skills"]) for d in parsed_data]
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Predictions saved to {OUTPUT_PATH}")
if __name__ == "__main__":
    save_dataset_predictions()  # <-- new
