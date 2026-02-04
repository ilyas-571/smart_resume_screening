import pandas as pd
from src.models.predict import predict
from src.data.resume_parser import extract_resume_data

# Path to your CSV
DATA_PATH = "data/raw/Resume.csv"

# Output CSV
OUTPUT_PATH = "data/output/Resume_with_scores.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Make sure the column name matches your CSV
text_column = "Resume_str"

# Lists to store results
fit_scores = []
parsed_data = []

for text in df[text_column]:
    score = predict(text)
    fit_scores.append(score)
    parsed_data.append(extract_resume_data(text))

# Create new columns
df["fit_score"] = fit_scores
df["name"] = [d["name"] for d in parsed_data]
df["email"] = [d["email"] for d in parsed_data]
df["phone"] = [d["phone"] for d in parsed_data]
df["skills"] = [", ".join(d["skills"]) for d in parsed_data]

# Save results
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Batch prediction done! Saved to {OUTPUT_PATH}")
