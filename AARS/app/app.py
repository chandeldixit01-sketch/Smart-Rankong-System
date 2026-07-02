"""
Redrob AI Ranker — Flask Backend
Handles file upload, TF-IDF semantic matching, multi-signal scoring, and
returns ranked candidates as JSON.
"""

import json
import os
import re
import time
import multiprocessing
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'), static_folder=os.path.join(base_dir, 'static'))
app.config['MAX_CONTENT_LENGTH'] = 600 * 1024 * 1024  # 600 MB

# =========================================================================
# Constants (from rank.py)
# =========================================================================
MODERN_RETRIEVAL_SKILLS = {
    'rag', 'pinecone', 'milvus', 'qdrant', 'weaviate', 'faiss',
    'embeddings', 'vector search', 'fine-tuning llms', 'lora', 'qlora',
    'peft', 'langchain'
}

CONSULTING_FIRMS = {
    'tcs', 'tata consultancy services', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcltech', 'hcl technologies',
    'tech mahindra', 'ltimindtree', 'lti', 'l&t infotech',
    'mphasis', 'mindtree',
}

TRAP_TITLES = {
    'marketing', 'operations', 'accountant', 'hr manager',
    'human resources', 'sales', 'customer support',
    'graphic designer', 'finance', 'billing'
}

PREFERRED_LOCATIONS = {
    'noida', 'pune', 'hyderabad', 'mumbai', 'delhi', 'ncr',
    'gurgaon', 'gurugram', 'bengaluru', 'bangalore', 'chennai'
}

# Default JD text (loaded from docx at startup if available)
DEFAULT_JD = ""


def load_default_jd():
    """Try to extract the bundled job_description.docx text."""
    jd_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'job_description.docx'
    )
    if not os.path.exists(jd_path):
        return "Paste your job description here."
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        z = zipfile.ZipFile(jd_path)
        tree = ET.parse(z.open('word/document.xml'))
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text = '\n'.join(
            ''.join(node.text or '' for node in p.findall('.//w:t', ns))
            for p in tree.findall('.//w:p', ns)
        )
        return text.strip()
    except Exception:
        return "Could not load default JD."


# =========================================================================
# Candidate text builder (for TF-IDF)
# =========================================================================
def build_candidate_text(cand):
    """Build a single text string representing the candidate for TF-IDF."""
    parts = []
    profile = cand.get('profile', {})
    parts.append(profile.get('headline', ''))
    parts.append(profile.get('summary', ''))
    parts.append(profile.get('current_title', ''))

    for job in cand.get('career_history', []):
        parts.append(job.get('title', ''))
        parts.append(job.get('description', ''))

    for skill in cand.get('skills', []):
        name = skill.get('name', '')
        prof = skill.get('proficiency', '')
        parts.append(f"{name} {prof}")

    return ' '.join(parts)


# =========================================================================
# Extract skills from JD text dynamically
# =========================================================================
def extract_target_skills(jd_text):
    """
    Extract skill keywords from JD text using a comprehensive skill lexicon.
    Returns a dict of skill_name -> weight.
    """
    jd_lower = jd_text.lower()

    # Comprehensive skill lexicon with default weights
    SKILL_LEXICON = {
        # Retrieval & NLP
        'embeddings': 1.0, 'vector search': 1.0, 'retrieval': 1.0,
        'semantic search': 1.0, 'sentence-transformers': 1.0, 'rag': 1.0,
        'information retrieval': 1.0, 'nlp': 0.8, 'fine-tuning llms': 0.8,
        'learning to rank': 0.8, 'machine learning': 0.8, 'deep learning': 0.8,
        'natural language processing': 0.8, 'text classification': 0.7,
        'named entity recognition': 0.7, 'sentiment analysis': 0.7,
        # Vector DBs
        'pinecone': 1.0, 'weaviate': 1.0, 'qdrant': 1.0, 'milvus': 1.0,
        'opensearch': 0.8, 'elasticsearch': 0.8, 'faiss': 1.0, 'chromadb': 0.8,
        # Languages & Frameworks
        'python': 0.8, 'pytorch': 0.8, 'tensorflow': 0.6, 'xgboost': 0.6,
        'lightgbm': 0.6, 'scikit-learn': 0.6, 'keras': 0.5, 'jax': 0.6,
        'java': 0.5, 'scala': 0.5, 'go': 0.5, 'rust': 0.5, 'c++': 0.5,
        # LLM & Fine-tuning
        'lora': 0.6, 'qlora': 0.6, 'peft': 0.6, 'langchain': 0.5,
        'llamaindex': 0.5, 'hugging face': 0.6, 'transformers': 0.7,
        # Eval
        'evaluation': 0.8, 'ndcg': 0.8, 'mrr': 0.8, 'map': 0.8,
        'a/b testing': 0.7, 'ab testing': 0.7,
        # Infra
        'distributed systems': 0.6, 'spark': 0.5, 'pyspark': 0.5,
        'airflow': 0.5, 'kafka': 0.5, 'kubernetes': 0.5, 'docker': 0.4,
        'aws': 0.4, 'gcp': 0.4, 'azure': 0.4,
        # Data
        'sql': 0.4, 'mongodb': 0.4, 'postgresql': 0.4, 'redis': 0.4,
        # General
        'data science': 0.6, 'data engineering': 0.5, 'mlops': 0.5,
        'recommendation systems': 0.8, 'ranking': 0.8, 'search': 0.7,
    }

    found = {}
    for skill, weight in SKILL_LEXICON.items():
        if skill in jd_lower:
            found[skill] = weight

    # If very few skills found, be more lenient
    if len(found) < 5:
        for skill, weight in SKILL_LEXICON.items():
            # Check word fragments
            for word in skill.split():
                if len(word) > 3 and word in jd_lower:
                    found[skill] = weight * 0.5
                    break

    return found if found else {k: v for k, v in list(SKILL_LEXICON.items())[:10]}


# =========================================================================
# Scoring Functions
# =========================================================================
def parse_date(d_str):
    if not d_str:
        return None
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except Exception:
        return None


def parse_career_dates(career, current_date):
    parsed = []
    for job in career:
        start = parse_date(job.get('start_date'))
        end_raw = parse_date(job.get('end_date'))
        end = end_raw or current_date
        parsed.append({'start': start, 'end': end, 'end_raw': end_raw, 'job': job})
    return parsed


def check_honeypot(cand, current_date, parsed_career=None):
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])
    education = cand.get('education', [])
    profile = cand.get('profile', {})

    if parsed_career is None:
        parsed_career = parse_career_dates(career, current_date)

    expert_zero = [s for s in skills if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0]
    if len(expert_zero) >= 3:
        return True, f"Expert in {len(expert_zero)} skills with 0 months duration"

    for idx, pj in enumerate(parsed_career):
        start, end = pj['start'], pj['end']
        duration = pj['job'].get('duration_months', 0)
        if start and duration:
            elapsed = (end.year - start.year) * 12 + (end.month - start.month)
            if duration > elapsed + 2:
                return True, f"Job duration ({duration}mo) > elapsed time ({elapsed}mo)"

    parsed_jobs = sorted([(pj['start'], pj['end']) for pj in parsed_career if pj['start']], key=lambda x: x[0])
    for i in range(len(parsed_jobs) - 1):
        s1, e1 = parsed_jobs[i]
        s2, e2 = parsed_jobs[i + 1]
        if s2 < e1 and (e1 - s2).days > 90:
            return True, f"Overlapping jobs by {(e1 - s2).days} days"

    for s in skills:
        name_lower = s.get('name', '').lower().strip()
        if name_lower in MODERN_RETRIEVAL_SKILLS and s.get('duration_months', 0) > 72:
            return True, f"Modern skill '{s.get('name')}' claims {s['duration_months']} months (>6 years)"

    curr_title = profile.get('current_title', '').lower()
    is_ai_claim = any(kw in curr_title for kw in {'ml', 'machine learning', 'ai', 'retrieval', 'search', 'nlp', 'data scientist'})
    if is_ai_claim and career:
        trap_count = sum(1 for job in career if any(t in job.get('title', '').lower() for t in TRAP_TITLES))
        if len(career) > 0 and trap_count / len(career) >= 0.70:
            return True, f"Career is {trap_count / len(career) * 100:.0f}% non-technical titles"

    for idx, edu in enumerate(education):
        start, end = edu.get('start_year'), edu.get('end_year')
        if start and end and end < start:
            return True, f"Education end year < start year"

    return False, ""


def check_consulting_only(career):
    all_companies = [job.get('company', '').lower().strip() for job in career]
    if not all_companies:
        return False
    def _is(name):
        for firm in CONSULTING_FIRMS:
            if name == firm or re.search(r'\b' + re.escape(firm) + r'\b', name):
                return True
        return False
    return all(_is(c) for c in all_companies)


def score_skills(candidate_skills, target_skills, assessments=None):
    total = 0.0
    for s in candidate_skills:
        name = s.get('name', '').lower().strip()
        if name in target_skills:
            weight = target_skills[name]
            prof_w = {'expert': 1.0, 'advanced': 0.8, 'intermediate': 0.6}.get(s.get('proficiency', '').lower(), 0.3)
            dur_f = min(1.0, s.get('duration_months', 0) / 36.0)
            end_f = 1.0 + (s.get('endorsements', 0) / 20.0)
            total += weight * prof_w * dur_f * end_f
    if assessments:
        relevant = {k: v for k, v in assessments.items() if k.lower().strip() in target_skills}
        if relevant:
            avg = sum(relevant.values()) / len(relevant)
            total *= (1.0 + (avg / 100.0) * 0.15)
    return min(1.0, total / 7.0)


def score_experience(years, ideal_min=5.0, ideal_max=9.0):
    sweet_min, sweet_max = ideal_min + 1, ideal_max - 1
    if sweet_min <= years <= sweet_max:
        return 1.0
    elif ideal_min <= years <= ideal_max:
        return 0.8
    elif (ideal_min - 1) <= years <= (ideal_max + 3):
        return 0.5
    elif (ideal_min - 2) <= years <= (ideal_max + 6):
        return 0.2
    return 0.0


def score_role(cand, target_skills):
    profile = cand.get('profile', {})
    career = cand.get('career_history', [])
    title = profile.get('current_title', '').lower().strip()
    headline = profile.get('headline', '').lower().strip()
    summary = profile.get('summary', '').lower().strip()

    if any(t in title for t in TRAP_TITLES):
        return 0.0

    primary = {'machine learning engineer', 'ml engineer', 'ai engineer', 'ai research engineer',
               'nlp engineer', 'search engineer', 'retrieval engineer', 'recommendation engineer',
               'recommendation systems engineer', 'data scientist', 'applied scientist'}
    infra = {'devops engineer', 'cloud engineer', 'platform engineer', 'sre',
             'site reliability engineer', 'infrastructure engineer'}
    dev = {'software engineer', 'backend engineer', 'developer', 'full stack developer'}

    skill_terms = set(target_skills.keys())

    title_score = 0.2
    if any(pt in title for pt in primary):
        title_score = 1.0
    elif any(it in title for it in infra):
        title_score = 0.7 if any(t in summary or t in headline for t in skill_terms) else 0.4
    elif any(dt in title for dt in dev):
        title_score = 0.7 if any(t in summary or t in headline for t in skill_terms) else 0.4

    combined = " ".join(j.get('description', '').lower() + " " + j.get('title', '').lower() for j in career)
    career_score = 0.0
    career_keywords = [
        ({'retrieval', 'vector search', 'semantic search', 'faiss', 'pinecone', 'milvus', 'qdrant', 'weaviate'}, 0.3),
        ({'ranking', 'learning to rank', 'recommendation', 'recommender'}, 0.3),
        ({'production ml', 'shipped', 'deployed', 'a/b test', 'evaluation', 'ndcg'}, 0.2),
        ({'data pipeline', 'spark', 'airflow', 'kafka', 'etl', 'backend'}, 0.2),
    ]
    for kw_set, bonus in career_keywords:
        if any(kw in combined for kw in kw_set):
            career_score += bonus

    return 0.5 * title_score + 0.5 * career_score


def score_signals(signals, current_date):
    rrr = signals.get('recruiter_response_rate', 0.0)
    icr = signals.get('interview_completion_rate', 0.0)
    recency = 0.1
    last_str = signals.get('last_active_date', '')
    if last_str:
        try:
            days = (current_date - datetime.strptime(last_str, "%Y-%m-%d")).days
            recency = 1.0 if days <= 30 else 0.7 if days <= 90 else 0.4 if days <= 180 else 0.1
        except Exception:
            pass
    otw = 1.0 if signals.get('open_to_work_flag', False) else 0.5
    gh = signals.get('github_activity_score', 0.0)
    gh_s = 0.0 if gh == -1 else gh / 100.0
    notice = signals.get('notice_period_days', 90)
    notice_s = 1.0 if notice <= 30 else 0.9 if notice <= 60 else 0.8 if notice <= 90 else 0.3
    saved = min(1.0, signals.get('saved_by_recruiters_30d', 0) / 15.0)
    appear = min(1.0, signals.get('search_appearance_30d', 0) / 200.0)

    return (0.22 * rrr + 0.18 * icr + 0.18 * recency + 0.12 * otw +
            0.10 * gh_s + 0.08 * notice_s + 0.07 * saved + 0.05 * appear)


def score_consistency(cand, parsed_career=None, current_date=None):
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])
    title = profile.get('current_title', '').lower()
    headline = profile.get('headline', '').lower()
    summary = profile.get('summary', '').lower()
    score = 1.0
    desc = " ".join(j.get('description', '').lower() + " " + j.get('title', '').lower() for j in career)
    ai_kw = {'ml', 'machine learning', 'ai', 'retrieval', 'search', 'nlp', 'data scientist'}
    ai_desc = {'embeddings', 'vector search', 'rag', 'pinecone', 'milvus', 'faiss', 'machine learning', 'nlp', 'retrieval', 'ranking'}

    if any(kw in title for kw in ai_kw) and career:
        if not any(any(k in j.get('title', '').lower() for k in {'developer', 'engineer', 'scientist', 'analyst'}) for j in career):
            score -= 0.3
    for text in [headline, summary]:
        if any(kw in text for kw in ai_desc) and career and not any(kw in desc for kw in ai_desc):
            score -= 0.2
    return max(0.0, score)


def score_location(profile, signals):
    loc = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    willing = signals.get('willing_to_relocate', False)
    if any(c in loc for c in PREFERRED_LOCATIONS):
        return 1.0
    elif country == 'india':
        return 0.8 if willing else 0.6
    return 0.4 if willing else 0.3


def generate_reasoning(cand, skill_s, role_s, final_s, target_skills, is_consulting, is_honeypot, hp_reason, tfidf_score):
    if is_honeypot:
        return f"Disqualified: {hp_reason}."
    p = cand['profile']
    title, company = p.get('current_title', ''), p.get('current_company', '')
    yrs = p.get('years_of_experience', 0)
    skills_found = [s['name'] for s in cand.get('skills', []) if s['name'].lower().strip() in target_skills][:3]
    skills_str = ', '.join(skills_found) if skills_found else 'relevant skills'
    rr = cand['redrob_signals'].get('recruiter_response_rate', 0)
    gh = cand['redrob_signals'].get('github_activity_score', 0)
    career_cos = [j['company'] for j in cand.get('career_history', []) if j.get('company') != company][:2]
    prev = f" (previously at {', '.join(career_cos)})" if career_cos else ""

    if final_s >= 0.75 and role_s >= 0.7:
        base = f"Strong match: {title} at {company}{prev} with {yrs:.1f} years. Expertise in {skills_str}. Semantic similarity: {tfidf_score:.0%}."
    elif gh > 40:
        base = f"Active developer (GitHub: {gh:.0f}), {title} at {company}{prev}, {yrs:.1f} yrs. Skills: {skills_str}."
    elif role_s >= 0.4:
        base = f"Relevant background: {title} at {company}{prev}, {yrs:.1f} yrs. Career shows experience with {skills_str}."
    else:
        base = f"{title} at {company}{prev}, {yrs:.1f} yrs. Response rate: {rr:.0%}. Skills: {skills_str}."

    concerns = []
    if is_consulting:
        concerns.append("consulting-only career")
    if cand['redrob_signals'].get('notice_period_days', 0) > 90:
        concerns.append(f"{cand['redrob_signals']['notice_period_days']}-day notice")
    if concerns:
        base += " Concerns: " + "; ".join(concerns) + "."
    return base


def _score_candidate_parallel(args):
    """Worker function for parallel candidate scoring."""
    cand, tfidf, target_skills, current_date = args
    profile = cand.get('profile', {})
    signals = cand.get('redrob_signals', {})
    career = cand.get('career_history', [])

    parsed = parse_career_dates(career, current_date)
    is_hp, hp_reason = check_honeypot(cand, current_date, parsed)
    is_consulting = check_consulting_only(career)

    assessments = signals.get('skill_assessment_scores', {})
    sk = score_skills(cand.get('skills', []), target_skills, assessments)
    ex = score_experience(profile.get('years_of_experience', 0))
    ro = score_role(cand, target_skills)
    si = score_signals(signals, current_date)
    co = score_consistency(cand)
    lo = score_location(profile, signals)

    if is_hp:
        final = 0.0
    else:
        # Blend rule-based with TF-IDF semantic similarity
        rule_score = (0.30 * sk + 0.25 * ro + 0.12 * ex +
                      0.13 * si + 0.12 * co + 0.08 * lo)
        final = 0.70 * rule_score + 0.30 * tfidf
        if is_consulting:
            final *= 0.50

    # Collect JD-relevant skills for display
    all_skills = [s['name'] for s in cand.get('skills', []) if s['name'].lower().strip() in target_skills]
    core_s = [s['name'] for s in cand.get('skills', [])
              if s['name'].lower().strip() in target_skills and target_skills.get(s['name'].lower().strip(), 0) >= 0.8]

    reasoning = generate_reasoning(cand, sk, ro, final, target_skills, is_consulting, is_hp, hp_reason, tfidf)

    return {
        'candidate_id': cand['candidate_id'],
        'title': profile.get('current_title', ''),
        'company': profile.get('current_company', ''),
        'location': profile.get('location', ''),
        'years_exp': profile.get('years_of_experience', 0),
        'score': round(final, 4),
        'skills': all_skills[:6],
        'core_skills': core_s[:4],
        'reasoning': reasoning,
        'is_honeypot': is_hp,
        'is_consulting': is_consulting
    }


# =========================================================================
# Main Ranking Pipeline
# =========================================================================
def run_ranking(candidates, jd_text, top_n=100):
    current_date = datetime(2026, 6, 22)
    target_skills = extract_target_skills(jd_text)
    num_cores = multiprocessing.cpu_count()

    # ---- Phase 1: Build TF-IDF vectors ----
    # Parallel candidate text building
    with multiprocessing.Pool(processes=num_cores) as pool:
        candidate_texts = pool.map(build_candidate_text, candidates, chunksize=1000)

    corpus = [jd_text] + candidate_texts

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    jd_vector = tfidf_matrix[0:1]
    cand_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(jd_vector, cand_vectors).flatten()

    # ---- Phase 2: Score each candidate in parallel ----
    tasks = [(cand, float(similarities[i]), target_skills, current_date) for i, cand in enumerate(candidates)]

    with multiprocessing.Pool(processes=num_cores) as pool:
        scored_candidates = pool.map(_score_candidate_parallel, tasks, chunksize=1000)

    # Post-process stats and results
    results = []
    honeypots = 0
    consulting_count = 0

    for item in scored_candidates:
        if item['is_honeypot']:
            honeypots += 1
        if item['is_consulting']:
            consulting_count += 1
        results.append(item)

    results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    return results[:top_n], len(candidates), honeypots, consulting_count


# =========================================================================
# Routes
# =========================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/default-jd')
def default_jd():
    return jsonify({'jd': DEFAULT_JD})


@app.route('/api/rank', methods=['POST'])
def rank():
    if 'candidates' not in request.files:
        return jsonify({'error': 'No candidates file uploaded'}), 400

    jd_text = request.form.get('jd', '')
    if len(jd_text.strip()) < 20:
        return jsonify({'error': 'Job description too short'}), 400

    top_n = int(request.form.get('top_n', 100))

    # Parse candidates from uploaded file
    file = request.files['candidates']
    start_time = time.time()

    candidates = []
    try:
        for line in file.stream:
            line_str = line.decode('utf-8').strip()
            if line_str:
                candidates.append(json.loads(line_str))
    except Exception as e:
        return jsonify({'error': f'Failed to parse candidates: {str(e)}'}), 400

    if not candidates:
        return jsonify({'error': 'No candidates found in file'}), 400

    # Run ranking
    try:
        results, total, honeypots, consulting = run_ranking(candidates, jd_text, top_n)
    except Exception as e:
        return jsonify({'error': f'Ranking failed: {str(e)}'}), 500

    elapsed = time.time() - start_time

    return jsonify({
        'results': results,
        'total_candidates': total,
        'honeypots_found': honeypots,
        'consulting_only': consulting,
        'runtime_seconds': round(elapsed, 2),
    })


# =========================================================================
# Startup
# =========================================================================
if __name__ == '__main__':
    DEFAULT_JD = load_default_jd()
    print(f"Default JD loaded: {len(DEFAULT_JD)} chars")
    app.run(debug=True, host='0.0.0.0', port=5000)
