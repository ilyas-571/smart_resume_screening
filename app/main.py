from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import os
import uuid

from src.models.predict import predict
from src.data.resume_parser import extract_resume_data

app = FastAPI(title="Smart Resume Screening API")

# Temp folders (cloud-safe)
UPLOAD_DIR = "tmp/uploads"
OUTPUT_DIR = "tmp/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = f"{UPLOAD_DIR}/{file_id}.csv"
    output_path = f"{OUTPUT_DIR}/{file_id}_output.csv"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(input_path)

    fit_scores = []
    parsed_data = []

    for text in df["Resume_str"]:
        score = predict(text)
        fit_scores.append(score)
        parsed_data.append(extract_resume_data(text))

    df["fit_score"] = fit_scores
    df["name"] = [d["name"] for d in parsed_data]
    df["email"] = [d["email"] for d in parsed_data]
    df["phone"] = [d["phone"] for d in parsed_data]
    df["skills"] = [", ".join(d["skills"]) for d in parsed_data]

    df.to_csv(output_path, index=False)

    return FileResponse(
        output_path,
        media_type="text/csv",
        filename="resume_scores.csv"
    )
