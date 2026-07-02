# 🚀 AARS - AI Automated Resume Ranking System

An intelligent resume ranking platform that automatically analyzes candidate profiles, detects fake or suspicious resumes, and ranks applicants based on their fit for a specific job description.

**Built for:** 🏆 India Runs Data & AI Challenge (Redrob Hackathon)

---

## 📖 Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [What Gets Evaluated](#what-gets-evaluated)
- [Security Features](#security-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Future Roadmap](#future-roadmap)
- [License](#license)

---

## ⚡ Quick Start

Get up and running in under 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/chandeldixit01-sketch/Smart-Rankong-System.git
cd Smart-Rankong-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the application
python app/app.py

# 4. Open in your browser
# Navigate to http://localhost:5000
```

---

## ✨ Features

**Smart Ranking**
- 🧠 AI-powered semantic resume matching using TF-IDF and Cosine Similarity
- 📊 Multi-factor scoring system that considers 10+ evaluation criteria
- 📈 Detailed explanations for every candidate's score

**Smart Security**
- 🚨 Honeypot detection to identify fake or suspicious resumes
- 💼 Consulting-only career detection
- 🔍 Automatic flagging of inconsistent candidate profiles

**User-Friendly**
- 🎨 Modern, intuitive web interface with glassmorphism design
- ⚡ Fast processing of large candidate datasets
- 📂 One-click CSV export of ranked results

---

## 🤖 How It Works

### The Ranking Process

1. **Upload your data** - Provide a JSONL file with candidate profiles and a job description
2. **Analysis** - The system analyzes resumes using AI semantic matching
3. **Scoring** - Candidates receive scores based on 10+ evaluation factors
4. **Security Check** - Suspicious resumes are automatically flagged and reviewed
5. **Results** - Get ranked candidates with detailed reasoning for each score

### Example Workflow

```
Candidate JSONL File + Job Description
         ↓
    AI Analysis
         ↓
  Multi-factor Scoring
         ↓
  Fraud Detection
         ↓
   Ranked Results with Explanations
         ↓
   Export as CSV
```

---

## 🛠 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Step-by-Step Setup

**Step 1: Clone the Repository**
```bash
git clone https://github.com/chandeldixit01-sketch/Smart-Rankong-System.git
cd Smart-Rankong-System
```

**Step 2: Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 3: Install Required Packages**
```bash
pip install -r requirements.txt
```

**Step 4: Run the Application**
```bash
python app/app.py
```

**Step 5: Access the Web Interface**
Open your browser and go to: `http://localhost:5000`

---

## 📱 Usage Guide

### Getting Started

1. **Open the application** at `http://localhost:5000`

2. **Upload candidate data**
   - Prepare a JSONL file with candidate profiles
   - Click "Upload File" and select your candidates file

3. **Enter job description**
   - Paste or upload the job description you want to match against
   - This helps the system understand role requirements

4. **Configure settings**
   - Select how many top candidates you want to see (e.g., top 10, top 20)
   - Adjust any preference filters if available

5. **Analyze and rank**
   - Click "Sort & Rank" and wait for the AI analysis to complete
   - Results appear instantly with detailed scoring

6. **Review results**
   - See ranked candidates with match scores
   - Read detailed explanations for each ranking decision
   - Identify flagged candidates with potential issues

7. **Export data**
   - Download ranked results as CSV for further analysis
   - Share with your hiring team

---

## 📊 What Gets Evaluated

Each candidate is scored based on multiple criteria:

| Factor | What It Measures |
|--------|------------------|
| **Skill Match** | How well candidate skills align with job requirements |
| **Semantic Similarity** | Resume content relevance to job description |
| **Experience Alignment** | Years and type of relevant experience |
| **Career Progression** | Growth trajectory and advancement |
| **Technical Skills** | Depth and breadth of technical expertise |
| **GitHub Activity** | Contribution history (if available) |
| **Recruiter Response** | Historical responsiveness to recruiters |
| **Notice Period** | Availability to start work |
| **Location Match** | Geographic preference alignment |
| **Resume Consistency** | No conflicting information or timeline gaps |

**Final Score** = Weighted combination of all above factors

---

## 🔒 Security Features

### Honeypot Detection

The system automatically identifies and flags suspicious resumes:

**Red Flags Detected:**
- ⚠️ Impossible years of experience (e.g., 20 years when candidate is 25)
- ⚠️ Technology used before it existed (e.g., React in 2008)
- ⚠️ Unrealistic skill claims or contradictions
- ⚠️ Conflicting employment history or gaps
- ⚠️ Education timeline errors or inconsistencies
- ⚠️ Consulting-only roles with no real employment

**What Happens:**
- Flagged candidates appear in a separate "Suspicious Profiles" section
- They're excluded from top rankings by default
- Hiring teams can review them separately if needed

---

## 🛠 Tech Stack

### Backend & Core
- **Python** - Primary programming language
- **Flask** - Lightweight web framework

### Machine Learning
- **Scikit-learn** - ML algorithms and utilities
- **NumPy** - Numerical computing
- **TF-IDF Vectorizer** - Text feature extraction
- **Cosine Similarity** - Semantic matching

### Frontend
- **HTML5** - Structure
- **CSS3** - Modern styling with glassmorphism
- **JavaScript** - Interactive features

---

## 📁 Project Structure

```
Smart-Rankong-System/
│
├── app/
│   ├── app.py                 # Main Flask application
│   └── templates/             # HTML templates
│
├── data/
│   ├── candidates.jsonl       # Sample candidate data
│   ├── rank.py               # Core ranking logic
│   └── sample_submission.csv  # Example output
│
├── scripts/
│   ├── run.bat              # Windows startup script
│   └── run.sh               # Linux/Mac startup script
│
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

---

## ⚡ Performance

**What to Expect:**
- ✅ Handles thousands of candidate profiles efficiently
- ✅ Optimized for standard CPU execution (no GPU required)
- ✅ Completes semantic matching in seconds
- ✅ Memory-efficient processing for large datasets

**Typical Performance:**
- 100 candidates: ~2-5 seconds
- 1,000 candidates: ~10-20 seconds
- 10,000+ candidates: ~1-2 minutes

---

## 🚀 Future Roadmap

We're constantly improving AARS. Here's what's coming:

- 🧠 **Advanced Embeddings** - Sentence Transformer for better semantic understanding
- 🤖 **LLM Integration** - GPT-powered resume analysis and insights
- 📸 **OCR Parsing** - Automatically parse PDF and image resumes
- 🎯 **Interview Prediction** - Estimate likelihood of interview success
- ✅ **ATS Compatibility** - Check if resumes pass Applicant Tracking Systems
- 📊 **Recruiter Dashboard** - Visual analytics and candidate pipeline management
- 📈 **Candidate Analytics** - Trends, insights, and hiring metrics

---

## 📜 License

This project is intended for **educational and hackathon purposes**.

---

## 🤝 Contributing

Found a bug? Have an idea? Feel free to open an issue or submit a pull request!

---

## 💬 Questions?

If you have questions or need help:
1. Check the Usage Guide above
2. Review the project files for examples
3. Open an issue on GitHub for technical support

**Happy ranking! 🎉**
