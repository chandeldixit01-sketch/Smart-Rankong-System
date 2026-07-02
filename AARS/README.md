# Redrob AI Ranker 🚀

An AI-powered candidate ranking system built for the **India Runs Data and AI Challenge**. This application allows recruiters to upload a large candidate pool and a job description, then intelligently scores and ranks the candidates using a combination of **TF-IDF Semantic Matching** and **Multi-Signal Rule-Based Scoring**.

## ✨ Key Features
- **Semantic Matching**: Uses TF-IDF and Cosine Similarity to find semantic relevance between the Job Description (JD) and Candidate profiles.
- **Multi-Signal Scoring**: Evaluates candidate experience, skills, GitHub activity, notice period, and recruiter response rates.
- **Honeypot Detection**: Automatically flags mathematically impossible profiles (e.g., more years of experience in a skill than total career duration, non-existent AI tools used in 1990).
- **Consulting-Firm Filter**: Identifies candidates who have only worked at consulting/IT services firms (TCS, Infosys, etc.) per JD specifications.
- **Fast & Beautiful UI**: A stunning, modern, glassmorphic UI that provides instant feedback, progress bars, and a clean results table.

---

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **Machine Learning**: `scikit-learn` (TF-IDF Vectorizer, Cosine Similarity), `numpy`
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism design system), Vanilla JavaScript (SSE-style progress tracking)

---

## 🚀 Quickstart Guide (For Judges)

We have provided simple startup scripts to get the application running instantly without manual environment configuration.

### **Windows**
1. Double-click the **`run.bat`** file inside the `scripts/` folder, OR run this from your terminal at the project root:
   ```cmd
   .\scripts\run.bat
   ```
2. The script will automatically create a virtual environment, install dependencies, and start the Flask server.
3. Open your browser and navigate to: **http://localhost:5000**

### **macOS / Linux**
1. Open your terminal in this directory.
2. Make the script executable and run it:
   ```bash
   chmod +x scripts/run.sh
   ./scripts/run.sh
   ```
3. Open your browser and navigate to: **http://localhost:5000**

*(If you prefer manual setup, run `pip install -r requirements.txt` followed by `python app/app.py`)*

---

## 📖 How to Use the App

1. **Upload Candidates:** Drag and drop the `candidates.jsonl` file (from the `data/` folder) into the drop zone on the left.
2. **Input Job Description:** Paste a job description in the right panel. Alternatively, click **"📄 Load Default JD"** to automatically load the hackathon's default job description.
3. **Rank:** Select the number of top candidates you want to view, then click **"🚀 Sort & Rank"**.
4. **Analyze Results:** Review the ranked table, which includes a detailed **Reasoning** column explaining exactly why a candidate received their score or why they were disqualified as a honeypot.
5. **Export:** Click **"⬇ Export CSV"** to save the top candidates.

---

## 📁 Directory Structure
- **`app/`**: Contains the Flask application, HTML templates, and CSS styling.
- **`data/`**: Contains the hackathon datasets (`candidates.jsonl`), CLI evaluation scripts (`rank.py`), schemas, and documentation.
- **`requirements.txt`**: Web application dependencies.
- **`scripts/`**: One-click startup scripts (`run.bat` and `run.sh`).

---
*Built for the Redrob Hackathon.* 
