# 🚀 AARS - AI Automated Resume Ranking System

> An intelligent resume ranking platform built for the **India Runs Data & AI Challenge (Redrob Hackathon)**.

AARS automatically analyzes thousands of candidate profiles, detects suspicious or inconsistent resumes, and ranks applicants based on how well they match a job description using both **AI-powered semantic matching** and **multi-signal candidate scoring**.

---

## ✨ Features

- 🧠 AI Semantic Resume Matching (TF-IDF + Cosine Similarity)
- 📊 Multi-factor Candidate Scoring
- 🚨 Honeypot & Fake Resume Detection
- 💼 Consulting-only Career Detection
- 📈 Detailed Score Explanation for Every Candidate
- 📂 CSV Export Support
- ⚡ Fast Processing
- 🎨 Modern Glassmorphism Web Interface

---

## 🛠 Tech Stack

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- NumPy
- TF-IDF Vectorizer
- Cosine Similarity

### Frontend
- HTML5
- CSS3
- JavaScript

---

# Ranking Factors

Candidates are evaluated using multiple signals including:

- Skill Match
- Semantic Resume Similarity
- Experience Alignment
- Career Progression
- Technical Skills
- GitHub Activity
- Recruiter Response Rate
- Notice Period
- Preferred Location
- Resume Consistency

The final score combines all of these signals into a single ranking.

---

# Honeypot Detection

The system automatically identifies suspicious resumes such as:

- Impossible years of experience
- Technology timeline inconsistencies
- Unrealistic skill claims
- Conflicting employment history
- Education timeline errors

These candidates are flagged before ranking.

---

# Project Structure

```
AARS/
│
├── app/
│   ├── app.py
│   └── templates/
│
├── data/
│   ├── candidates.jsonl
│   ├── rank.py
│   └── sample_submission.csv
│
├── scripts/
│   ├── run.bat
│   └── run.sh
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AARS.git

cd AARS
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app/app.py
```

Open your browser

```
http://localhost:5000
```

---

# Usage

1. Upload the candidate JSONL file.
2. Paste or load a Job Description.
3. Select the number of top candidates.
4. Click **Sort & Rank**.
5. Review ranked candidates with detailed reasoning.
6. Export results as CSV.

---

# Performance

- Handles large candidate datasets efficiently
- Optimized for CPU execution
- Fast semantic matching
- Memory-efficient processing

---

# Future Improvements

- Sentence Transformer Embeddings
- LLM-based Resume Analysis
- OCR Resume Parsing
- Interview Prediction Score
- ATS Compatibility Checker
- Recruiter Dashboard
- Candidate Analytics

---

# Built For

🏆 **India Runs Data & AI Challenge**

An AI-powered recruitment solution designed to make candidate screening faster, smarter, and more reliable.

---

## 📜 License

This project is intended for educational and hackathon purposes.
