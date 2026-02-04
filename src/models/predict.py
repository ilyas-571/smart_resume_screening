import joblib

def predict(text):
    model = joblib.load("models/resume_model.pkl")
    vectorizer = joblib.load("models/tfidf.pkl")
    X = vectorizer.transform([text])
    return round(model.predict_proba(X)[0][1] * 100, 2)