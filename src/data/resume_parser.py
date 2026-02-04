import re

def extract_resume_data(text):
    # Extract email
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    email = email[0] if email else None

    # Extract phone number (simple)
    phone = re.findall(r'\+?\d[\d -]{8,12}\d', text)
    phone = phone[0] if phone else None

    # Extract skills (example)
    skills_keywords = ["python", "java", "c++", "ml", "data", "ai", "sql", "javascript"]
    skills_found = [skill for skill in skills_keywords if skill in text.lower()]

    # Name extraction is tricky, here we just use first line
    lines = text.strip().split("\n")
    name = lines[0] if lines else None

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills_found
    }
