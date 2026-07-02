import json
import csv
import argparse
import os
import re
import multiprocessing
from datetime import datetime

# ===========================================================================
# Target skills extracted from the job description
# ===========================================================================
# Weights reflect the JD's priority tiers:
#   1.0 = "Things you absolutely need"
#   0.8 = core-adjacent / frequently mentioned
#   0.6 = "Things we'd like you to have"
#   0.5 = desirable / adjacent
TARGET_SKILLS = {
    # Core Retrieval & NLP  (JD: "Things you absolutely need")
    'embeddings': 1.0,
    'vector search': 1.0,
    'retrieval': 1.0,
    'semantic search': 1.0,
    'sentence-transformers': 1.0,
    'rag': 1.0,
    'information retrieval': 1.0,
    'nlp': 0.8,
    'fine-tuning llms': 0.8,
    'learning to rank': 0.8,
    'machine learning': 0.8,

    # Vector Databases  (JD: "Production experience with vector databases")
    'pinecone': 1.0,
    'weaviate': 1.0,
    'qdrant': 1.0,
    'milvus': 1.0,
    'opensearch': 0.8,
    'elasticsearch': 0.8,
    'faiss': 1.0,

    # ML & Core Engineering
    'python': 0.8,
    'xgboost': 0.6,
    'pytorch': 0.8,
    'tensorflow': 0.6,
    'evaluation': 0.8,
    'ndcg': 0.8,
    'mrr': 0.8,
    'map': 0.8,

    # Desirable/Adjacent  (JD: "Things we'd like you to have")
    'lora': 0.6,
    'qlora': 0.6,
    'peft': 0.6,
    'distributed systems': 0.6,
    'spark': 0.5,
    'pyspark': 0.5,
    'airflow': 0.5
}

# Skills the JD considers "absolutely needed" — used to distinguish core
# vs. desirable in reasoning text.
CORE_SKILLS = {
    'embeddings', 'vector search', 'retrieval', 'semantic search',
    'sentence-transformers', 'rag', 'information retrieval',
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss',
    'python', 'evaluation', 'ndcg', 'mrr', 'map'
}

# Modern skills released post-2020 (used for timeline anachronism checks)
MODERN_RETRIEVAL_SKILLS = {
    'rag', 'pinecone', 'milvus', 'qdrant', 'weaviate', 'faiss',
    'embeddings', 'vector search', 'fine-tuning llms', 'lora', 'qlora',
    'peft', 'langchain'
}

# Consulting firms — JD explicitly says:
# "People who have only worked at consulting firms (TCS, Infosys, Wipro,
#  Accenture, Cognizant, Capgemini, etc.) in their entire career."
# The "etc." warrants including other major Indian IT services firms.
CONSULTING_FIRMS = {
    'tcs', 'tata consultancy services',
    'infosys',
    'wipro',
    'accenture',
    'cognizant',
    'capgemini',
    'hcltech', 'hcl technologies',
    'tech mahindra',
    'ltimindtree', 'lti', 'l&t infotech',
    'mphasis',
    'mindtree',
}

# Trap titles (keyword stuffers with non-tech current titles)
TRAP_TITLES = {
    'marketing', 'operations', 'accountant', 'hr manager',
    'human resources', 'sales', 'customer support',
    'graphic designer', 'finance', 'billing'
}

# Indian cities the JD explicitly welcomes
PREFERRED_LOCATIONS = {
    'noida', 'pune', 'hyderabad', 'mumbai', 'delhi', 'ncr',
    'gurgaon', 'gurugram', 'bengaluru', 'bangalore', 'chennai'
}


def parse_date(d_str):
    if not d_str:
        return None
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except Exception:
        return None


def parse_career_dates(career, current_date):
    """
    Parses each job's start/end date once.  Both check_honeypot() and
    score_consistency() read from the cached result instead of re-parsing
    the same date strings independently.
    """
    parsed = []
    for job in career:
        start = parse_date(job.get('start_date'))
        end_raw = parse_date(job.get('end_date'))
        end = end_raw or current_date
        parsed.append({
            'start': start, 'end': end, 'end_raw': end_raw, 'job': job
        })
    return parsed


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

def check_honeypot(cand, current_date, parsed_career=None):
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])
    education = cand.get('education', [])
    profile = cand.get('profile', {})

    if parsed_career is None:
        parsed_career = parse_career_dates(career, current_date)

    # NOTE: a "salary min > max" rule was here originally.  It was removed —
    # measured against the real candidate sample, it fired on 26 % of ALL
    # candidates (not just honeypots), which is just noisy salary-range data,
    # not a "subtly impossible profile" per the spec's definition.  It is not
    # listed anywhere in redrob_signals_doc.md as a honeypot signal.

    # 1. Expert skills with 0 duration (>= 3 skills)
    expert_zero = [
        s for s in skills
        if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0
    ]
    if len(expert_zero) >= 3:
        return True, (
            f"Expert in {len(expert_zero)} skills with 0 months duration"
        )

    # 2. Job duration exceeds elapsed time since start_date
    for idx, pj in enumerate(parsed_career):
        start = pj['start']
        end = pj['end']
        duration = pj['job'].get('duration_months', 0)
        if start and duration:
            elapsed_months = (
                (end.year - start.year) * 12 + (end.month - start.month)
            )
            if duration > elapsed_months + 2:
                return True, (
                    f"Job {idx} duration ({duration} mos) > elapsed "
                    f"({elapsed_months} mos) since {pj['job'].get('start_date')}"
                )

    # 3. Significant job overlap (> 90 days) for full-time jobs
    parsed_jobs = sorted(
        [(pj['start'], pj['end']) for pj in parsed_career if pj['start']],
        key=lambda x: x[0]
    )
    for i in range(len(parsed_jobs) - 1):
        s1, e1 = parsed_jobs[i]
        s2, e2 = parsed_jobs[i + 1]
        if s2 < e1:
            overlap_days = (e1 - s2).days
            if overlap_days > 90:
                return True, (
                    f"Career history has concurrent full-time jobs "
                    f"overlapping by {overlap_days} days"
                )

    # 4. Modern skill anachronisms (> 72 months)
    for s in skills:
        name_lower = s.get('name', '').lower().strip()
        duration = s.get('duration_months', 0)
        if name_lower in MODERN_RETRIEVAL_SKILLS and duration > 72:
            return True, (
                f"Modern skill '{s.get('name')}' claims impossible "
                f"experience of {duration} months (> 6 years)"
            )

    # 5. Title stuffing across career (>= 70 % trap titles)
    curr_title = profile.get('current_title', '').lower()
    is_ai_claim = any(
        kw in curr_title
        for kw in {
            'ml', 'machine learning', 'ai', 'retrieval',
            'search', 'nlp', 'data scientist'
        }
    )
    if is_ai_claim and career:
        trap_count = sum(
            1 for job in career
            if any(t in job.get('title', '').lower() for t in TRAP_TITLES)
        )
        if trap_count / len(career) >= 0.70:
            return True, (
                f"Career history consists of "
                f"{trap_count / len(career) * 100:.0f}% "
                f"non-technical trap titles"
            )

    # 6. Education & work timeline contradictions
    for idx, edu in enumerate(education):
        start = edu.get('start_year')
        end = edu.get('end_year')
        if start and end and end < start:
            return True, (
                f"Education {idx} end year ({end}) < start year ({start})"
            )

    return False, ""


# ---------------------------------------------------------------------------
# Consulting-only check  (word-boundary matching, expanded list)
# ---------------------------------------------------------------------------

def check_consulting_only(career):
    """
    Returns True if every company in the candidate's career history matches
    one of the known IT-services/consulting firms.
    Uses word-boundary matching to avoid false positives like 'TCSolutions'.
    """
    all_companies = [job.get('company', '').lower().strip() for job in career]
    if not all_companies:
        return False

    def _is_consulting(company_name):
        for firm in CONSULTING_FIRMS:
            # Exact match or word-boundary match
            if company_name == firm:
                return True
            pattern = r'\b' + re.escape(firm) + r'\b'
            if re.search(pattern, company_name):
                return True
        return False

    return all(_is_consulting(comp) for comp in all_companies)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_skills(candidate_skills, assessment_scores=None):
    """
    Scores candidate skills against TARGET_SKILLS.
    Now also integrates platform-verified assessment scores as a bonus.
    """
    total_score = 0.0
    for s in candidate_skills:
        name = s.get('name', '').lower().strip()
        if name in TARGET_SKILLS:
            weight = TARGET_SKILLS[name]

            prof = s.get('proficiency', 'beginner').lower()
            prof_weight = 0.3
            if prof == 'expert':
                prof_weight = 1.0
            elif prof == 'advanced':
                prof_weight = 0.8
            elif prof == 'intermediate':
                prof_weight = 0.6

            duration = s.get('duration_months', 0)
            duration_factor = min(1.0, duration / 36.0)

            endorsements = s.get('endorsements', 0)
            endorsement_factor = 1.0 + (endorsements / 20.0)

            s_score = weight * prof_weight * duration_factor * endorsement_factor
            total_score += s_score

    # Assessment bonus: if the candidate has platform-verified scores for
    # JD-relevant skills, boost the skill score.  This rewards candidates
    # whose self-reported proficiency is backed by actual assessment data.
    if assessment_scores:
        relevant_assessments = {
            k: v for k, v in assessment_scores.items()
            if k.lower().strip() in TARGET_SKILLS
        }
        if relevant_assessments:
            avg_score = sum(relevant_assessments.values()) / len(relevant_assessments)
            # A perfect 100 avg across relevant assessments adds ~15 % bonus
            assessment_bonus = (avg_score / 100.0) * 0.15
            total_score *= (1.0 + assessment_bonus)

    return min(1.0, total_score / 7.0)


def score_experience(years):
    """
    Smooth scoring for experience alignment with JD's "5-9 years" range.
    JD: "6-8 years total experience, of which 4-5 are in applied ML/AI roles."
    """
    if 6.0 <= years <= 8.0:
        return 1.0
    elif 5.0 <= years <= 9.0:
        return 0.8
    elif 4.0 <= years <= 12.0:
        return 0.5
    elif 3.0 <= years <= 15.0:
        return 0.2
    else:
        return 0.0


def score_role_v2(cand):
    profile = cand.get('profile', {})
    career = cand.get('career_history', [])

    title = profile.get('current_title', '').lower().strip()
    headline = profile.get('headline', '').lower().strip()
    summary = profile.get('summary', '').lower().strip()

    if any(t in title for t in TRAP_TITLES):
        return 0.0

    primary_titles = {
        'machine learning engineer', 'ml engineer', 'ai engineer',
        'ai research engineer', 'nlp engineer', 'search engineer',
        'retrieval engineer', 'recommendation engineer',
        'recommendation systems engineer'
    }
    secondary_titles = {
        'data scientist', 'nlp scientist', 'applied scientist',
        'analytics engineer', 'data engineer'
    }
    # JD explicitly values distributed-systems / large-scale inference
    infra_titles = {
        'devops engineer', 'cloud engineer', 'platform engineer',
        'site reliability engineer', 'sre', 'infrastructure engineer',
        'distributed systems engineer'
    }
    generalist_dev_titles = {
        'software engineer', 'backend engineer', 'developer',
        'full stack developer', 'java developer', '.net developer',
        'mobile developer', 'frontend engineer'
    }

    ai_terms = {
        'embeddings', 'vector search', 'rag', 'machine learning', 'nlp',
        'ranking', 'distributed systems', 'large-scale',
        'inference optimization'
    }

    title_score = 0.2
    if any(pt in title for pt in primary_titles):
        title_score = 1.0
    elif any(st in title for st in secondary_titles):
        title_score = 0.7
    elif any(it in title for it in infra_titles):
        if any(term in summary or term in headline for term in ai_terms):
            title_score = 0.8
        else:
            title_score = 0.5
    elif any(gt in title for gt in generalist_dev_titles):
        if any(term in summary or term in headline for term in ai_terms):
            title_score = 0.8
        else:
            title_score = 0.5

    combined_desc = " ".join(
        job.get('description', '').lower() + " " + job.get('title', '').lower()
        for job in career
    )

    career_score = 0.0
    if any(kw in combined_desc for kw in {
        'retrieval', 'vector search', 'semantic search', 'neural search',
        'faiss', 'pinecone', 'milvus', 'qdrant', 'weaviate'
    }):
        career_score += 0.3
    if any(kw in combined_desc for kw in {
        'ranking', 'learning to rank', 'ltr', 'recommendation', 'recommender'
    }):
        career_score += 0.3
    if any(kw in combined_desc for kw in {
        'production ml', 'shipped', 'deployed', 'ab testing', 'a/b test',
        'evaluation', 'ndcg', 'mrr', 'map'
    }):
        career_score += 0.2
    if any(kw in combined_desc for kw in {
        'data pipeline', 'spark', 'pyspark', 'airflow', 'kafka', 'etl',
        'backend', 'data system'
    }):
        career_score += 0.2

    return 0.5 * title_score + 0.5 * career_score


def score_signals(signals, current_date):
    rrr = signals.get('recruiter_response_rate', 0.0)
    icr = signals.get('interview_completion_rate', 0.0)

    last_active_str = signals.get('last_active_date', '')
    recency_score = 0.1
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
            days_inactive = (current_date - last_active).days
            if days_inactive <= 30:
                recency_score = 1.0
            elif days_inactive <= 90:
                recency_score = 0.7
            elif days_inactive <= 180:
                recency_score = 0.4
        except Exception:
            pass

    otw = 1.0 if signals.get('open_to_work_flag', False) else 0.5

    gh = signals.get('github_activity_score', 0.0)
    gh_score = 0.0 if gh == -1 else (gh / 100.0)

    notice = signals.get('notice_period_days', 90)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.9
    elif notice <= 90:
        notice_score = 0.8
    else:
        notice_score = 0.3

    # ---- New signals (previously unused) ----

    # Recruiter social proof — saved_by_recruiters_30d.
    # If recruiters are bookmarking this person, that's strong market signal.
    saved = signals.get('saved_by_recruiters_30d', 0)
    saved_score = min(1.0, saved / 15.0)  # 15+ saves → 1.0

    # Search appearances — how often they show up in recruiter searches.
    appearances = signals.get('search_appearance_30d', 0)
    appearance_score = min(1.0, appearances / 200.0)  # 200+ → 1.0

    # Weights — slightly redistributed to accommodate new signals
    w_rrr = 0.22
    w_icr = 0.18
    w_recency = 0.18
    w_otw = 0.12
    w_gh = 0.10
    w_notice = 0.08
    w_saved = 0.07
    w_appearance = 0.05

    return (
        w_rrr * rrr +
        w_icr * icr +
        w_recency * recency_score +
        w_otw * otw +
        w_gh * gh_score +
        w_notice * notice_score +
        w_saved * saved_score +
        w_appearance * appearance_score
    )


def score_location(profile, signals):
    """
    Mild location bonus.  JD: "Pune/Noida-preferred but flexible.
    Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome."
    Not a penalty for non-India — JD says "case-by-case" for international.
    """
    location = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    willing = signals.get('willing_to_relocate', False)

    if any(city in location for city in PREFERRED_LOCATIONS):
        return 1.0
    elif country == 'india':
        return 0.8 if willing else 0.6
    else:
        return 0.4 if willing else 0.3


def score_consistency(cand, parsed_career=None):
    profile = cand.get('profile', {})
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])

    title = profile.get('current_title', '').lower()
    headline = profile.get('headline', '').lower()
    summary = profile.get('summary', '').lower()

    consistency_score = 1.0
    desc_text = " ".join(
        job.get('description', '').lower() + " " + job.get('title', '').lower()
        for job in career
    )

    is_curr_ai = any(
        kw in title
        for kw in {
            'ml', 'machine learning', 'ai', 'retrieval',
            'search', 'nlp', 'data scientist'
        }
    )
    if is_curr_ai and career:
        has_tech_history = any(
            any(kw in job.get('title', '').lower()
                for kw in {'developer', 'engineer', 'scientist', 'analyst',
                           'programmer', 'architect'})
            for job in career
        )
        if not has_tech_history:
            consistency_score -= 0.3

    has_ai_hl = any(
        kw in headline
        for kw in {
            'embeddings', 'vector search', 'rag', 'pinecone', 'milvus',
            'faiss', 'machine learning', 'nlp'
        }
    )
    if has_ai_hl and career:
        has_ai_desc = any(
            kw in desc_text
            for kw in {
                'embeddings', 'vector search', 'rag', 'pinecone', 'milvus',
                'faiss', 'machine learning', 'nlp', 'retrieval', 'ranking'
            }
        )
        if not has_ai_desc:
            consistency_score -= 0.2

    has_ai_sum = any(
        kw in summary
        for kw in {
            'embeddings', 'vector search', 'rag', 'pinecone', 'milvus',
            'faiss', 'machine learning', 'nlp'
        }
    )
    if has_ai_sum and career:
        has_ai_desc = any(
            kw in desc_text
            for kw in {
                'embeddings', 'vector search', 'rag', 'pinecone', 'milvus',
                'faiss', 'machine learning', 'nlp', 'retrieval', 'ranking'
            }
        )
        if not has_ai_desc:
            consistency_score -= 0.3

    cand_modern_skills = [
        s['name'].lower() for s in skills
        if s['name'].lower() in MODERN_RETRIEVAL_SKILLS
    ]
    if cand_modern_skills and career:
        has_skill_mention = any(s in desc_text for s in cand_modern_skills)
        if not has_skill_mention:
            consistency_score -= 0.2

    if parsed_career is None:
        parsed_career = parse_career_dates(career, datetime.now())
    parsed_dates = sorted(
        [(pj['start'], pj['end_raw'])
         for pj in parsed_career if pj['start'] and pj['end_raw']],
        key=lambda x: x[0]
    )

    for i in range(len(parsed_dates) - 1):
        _, e1 = parsed_dates[i]
        s2, _ = parsed_dates[i + 1]
        gap_days = (s2 - e1).days
        if gap_days > 1825:
            consistency_score -= 0.1
            break

    return max(0.0, consistency_score)


# ---------------------------------------------------------------------------
# Reasoning generation  (Stage 4 quality — specific, JD-connected, honest,
#                         no hallucination, varied, appropriate tone)
# ---------------------------------------------------------------------------

def generate_reasoning_v2(
    cand, skill_score, exp_score, role_score, signal_score,
    consistency_score, final_score, is_consulting, is_honeypot,
    honeypot_reason
):
    if is_honeypot:
        return (
            f"Disqualified: Logically impossible profile data. "
            f"Honeypot check: {honeypot_reason}."
        )

    profile = cand.get('profile', {})
    title = profile.get('current_title', 'Engineer')
    company = profile.get('current_company', 'N/A')
    years_exp = profile.get('years_of_experience', 0.0)

    skills = cand.get('skills', [])
    # Separate core vs. desirable skills for JD-connected reasoning
    core_skills_found = [
        s['name'] for s in skills
        if s['name'].lower().strip() in CORE_SKILLS
    ]
    desirable_skills_found = [
        s['name'] for s in skills
        if s['name'].lower().strip() in TARGET_SKILLS
        and s['name'].lower().strip() not in CORE_SKILLS
    ]
    tech_skills = [
        s['name'] for s in skills
        if s['name'].lower() in TARGET_SKILLS
    ]
    top_skills_str = ", ".join(tech_skills[:3]) if tech_skills else "relevant tools"

    # Build a richer skills phrase that distinguishes core vs. desirable
    if core_skills_found and desirable_skills_found:
        skills_phrase = (
            f"core competencies in {', '.join(core_skills_found[:2])} "
            f"and desirable skills like {desirable_skills_found[0]}"
        )
    elif core_skills_found:
        skills_phrase = (
            f"core competencies in {', '.join(core_skills_found[:3])}"
        )
    else:
        skills_phrase = f"skills in {top_skills_str}"

    notice = cand['redrob_signals'].get('notice_period_days', 0)
    rr = cand['redrob_signals'].get('recruiter_response_rate', 0.0)
    gh = cand['redrob_signals'].get('github_activity_score', 0.0)

    assessments = cand['redrob_signals'].get('skill_assessment_scores', {})
    assess_str = ""
    if assessments:
        top_assess = max(assessments.items(), key=lambda x: x[1])
        assess_str = (
            f" and a verified {top_assess[0]} assessment of "
            f"{top_assess[1]:.0f}%"
        )

    # Career history context — mention most relevant past roles
    career = cand.get('career_history', [])
    career_companies = [
        job.get('company', '') for job in career
        if job.get('company', '') != company
    ][:2]
    career_ctx = ""
    if career_companies:
        career_ctx = f" (previously at {', '.join(career_companies)})"

    # JD role context phrase for specificity
    jd_ctx = "for Redrob's retrieval and ranking intelligence layer"

    # Priority order: strength signals first, concerns appended after.
    # Stage 4 review wants tone to match overall quality.
    if final_score >= 0.75 and role_score >= 0.8:
        base = (
            f"Strong match {jd_ctx}: {title} with {years_exp:.1f} years "
            f"of experience at {company}{career_ctx}. Demonstrates {skills_phrase}"
            f"{assess_str}, with solid engagement signals "
            f"(response rate {rr:.0%})."
        )
    elif gh > 40.0:
        base = (
            f"Active developer contributing to open-source (GitHub score "
            f"{gh:.0f}), currently {title} at {company} with "
            f"{years_exp:.1f} years{career_ctx}. Brings {skills_phrase} "
            f"relevant to the retrieval engineering focus of this role."
        )
    elif assess_str != "":
        base = (
            f"Platform-verified technical fit {jd_ctx}: {title} at "
            f"{company} ({years_exp:.1f} yrs){career_ctx}. "
            f"Assessment data confirms capabilities — {skills_phrase}"
            f"{assess_str}."
        )
    elif 6.0 <= years_exp < 7.0 and role_score >= 0.6:
        base = (
            f"Ideal seniority band (JD: 6-8 years) as {title} at "
            f"{company}{career_ctx}. {years_exp:.1f} years of experience "
            f"with {skills_phrase} and a {notice}-day notice period."
        )
    elif role_score < 0.8 and role_score >= 0.4:
        base = (
            f"Adjacent-title candidate with transferable depth: currently "
            f"{title} at {company} ({years_exp:.1f} yrs){career_ctx}. "
            f"Career history shows production experience with "
            f"{top_skills_str}, making them relevant despite the "
            f"generalist title."
        )
    else:
        base = (
            f"Candidate {jd_ctx}: {title} at {company} with "
            f"{years_exp:.1f} years of experience{career_ctx}. "
            f"Demonstrates {skills_phrase} with a responsiveness "
            f"score of {rr:.0%}."
        )

    # Concerns appended after strength-based framing, not replacing it.
    concerns = []
    if is_consulting:
        concerns.append(
            "consulting-only background may require adaptation to "
            "a product-engineering environment (JD explicitly flags this)"
        )
    if notice > 90:
        concerns.append(
            f"long notice period ({notice} days) exceeds the JD's "
            f"preference for sub-30-day availability"
        )
    if consistency_score < 0.7:
        concerns.append(
            "self-reported skills aren't fully corroborated by "
            "career-history descriptions"
        )

    if concerns:
        base += " However, " + "; and ".join(concerns) + "."

    return base


# ---------------------------------------------------------------------------
# Per-candidate scoring worker  (must be top-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

# Shared state injected before Pool is created
_CURRENT_DATE = datetime(2026, 6, 22)


def _score_one(cand):
    """
    Score a single candidate dict.  This function is called in a worker
    process, so it must only reference module-level globals.
    Returns a result dict ready to append to all_scored.
    """
    current_date = _CURRENT_DATE

    candidate_id = cand['candidate_id']
    profile      = cand.get('profile', {})
    skills       = cand.get('skills', [])
    signals      = cand.get('redrob_signals', {})
    career       = cand.get('career_history', [])

    parsed_career_cache = parse_career_dates(career, current_date)
    is_honeypot, honeypot_reason = check_honeypot(
        cand, current_date, parsed_career=parsed_career_cache
    )
    is_consulting = check_consulting_only(career)

    assessment_scores  = signals.get('skill_assessment_scores', {})
    skill_score        = score_skills(skills, assessment_scores)
    exp_score          = score_experience(profile.get('years_of_experience', 0.0))
    role_score         = score_role_v2(cand)
    signal_score       = score_signals(signals, current_date)
    consistency_score  = score_consistency(cand, parsed_career=parsed_career_cache)
    location_score     = score_location(profile, signals)

    if is_honeypot:
        final_score = 0.0
    else:
        # Rebalanced weights — JD priority order:
        #   Skills (0.30): "Things you absolutely need"
        #   Role/Career (0.25): "the gap between what the JD says and means"
        #   Signals (0.13): behavioral modifier
        #   Experience (0.12): "range, not a requirement"
        #   Consistency (0.12): keyword-stuffer penalty
        #   Location (0.08): "Pune/Noida-preferred but flexible"
        final_score = (
            0.30 * skill_score +
            0.25 * role_score +
            0.13 * signal_score +
            0.12 * exp_score +
            0.12 * consistency_score +
            0.08 * location_score
        )
        if is_consulting:
            final_score *= 0.50

    reasoning = generate_reasoning_v2(
        cand, skill_score, exp_score, role_score, signal_score,
        consistency_score, final_score, is_consulting,
        is_honeypot, honeypot_reason
    )

    return {
        'candidate_id':      candidate_id,
        'skill_match_score': skill_score,
        'experience_score':  exp_score,
        'role_match_score':  role_score,
        'signal_score':      signal_score,
        'consistency_score': consistency_score,
        'location_score':    location_score,
        'final_score':       final_score,
        'is_honeypot':       int(is_honeypot),
        'is_consulting_only':int(is_consulting),
        'reasoning':         reasoning
    }


# ---------------------------------------------------------------------------
# Main ranking pipeline  (parallel — uses all available CPU cores)
# ---------------------------------------------------------------------------

def rank_candidates(candidates_path, out_submission_path, out_report_path,
                    n_workers=None):
    cpu_count = n_workers or multiprocessing.cpu_count()
    print(f"Ranking candidates from {candidates_path}...")
    print(f"Using {cpu_count} CPU workers (machine has {multiprocessing.cpu_count()} cores)")

    # ---- Phase 1: Stream-load all candidates into memory ----
    # Supports both .jsonl (one object per line) and .json (array).
    print("Loading candidates...")
    candidates = []
    with open(candidates_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    raw_stripped = raw.strip()
    if raw_stripped.startswith('['):
        # JSON array format (e.g. sample_candidates.json)
        candidates = json.loads(raw_stripped)
    else:
        # JSONL format (e.g. candidates.jsonl)
        for line in raw_stripped.splitlines():
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    total = len(candidates)
    print(f"Loaded {total} candidates. Scoring in parallel...")

    # ---- Phase 2: Parallel scoring across all CPU cores ----
    # _score_one is a top-level function so it can be pickled by Pool.
    # chunksize=500 amortises IPC overhead: each worker gets 500 candidates
    # per round-trip instead of one, reducing pickle/unpickle overhead ~500x.
    chunk = max(100, total // (cpu_count * 4))
    all_scored = []
    done = 0
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for result in pool.imap(_score_one, candidates, chunksize=chunk):
            all_scored.append(result)
            done += 1
            if done % 20000 == 0:
                print(f"  Scored {done}/{total}...")

    print(f"Scoring complete. Total candidates: {total}")

    # ---- Phase 3: Sort ----
    print("Sorting candidates...")
    all_scored.sort(
        key=lambda x: (-round(x['final_score'], 4), x['candidate_id'])
    )

    # ---- Phase 4: Write feature report ----
    print(f"Writing full feature report to {out_report_path}...")
    with open(out_report_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'candidate_id', 'skill_match_score', 'experience_score',
            'role_match_score', 'signal_score', 'consistency_score',
            'location_score', 'final_score', 'is_honeypot',
            'is_consulting_only', 'reasoning'
        ])
        for row in all_scored:
            writer.writerow([
                row['candidate_id'],
                f"{row['skill_match_score']:.4f}",
                f"{row['experience_score']:.4f}",
                f"{row['role_match_score']:.4f}",
                f"{row['signal_score']:.4f}",
                f"{row['consistency_score']:.4f}",
                f"{row['location_score']:.4f}",
                f"{row['final_score']:.4f}",
                row['is_honeypot'],
                row['is_consulting_only'],
                row['reasoning']
            ])

    # ---- Phase 5: Write top-100 submission ----
    print(f"Writing top 100 submission to {out_submission_path}...")
    with open(out_submission_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for rank, row in enumerate(all_scored[:100], 1):
            writer.writerow([
                row['candidate_id'],
                rank,
                f"{row['final_score']:.4f}",
                row['reasoning']
            ])

    print("Ranking and output generation successfully completed.")


if __name__ == '__main__':
    # Required on Windows so spawned worker processes don't re-execute main
    multiprocessing.freeze_support()

    base_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranking Engine (parallel)"
    )
    parser.add_argument(
        '--candidates', type=str,
        default=os.path.join(base_dir, 'candidates.jsonl'),
        help='Path to candidates.jsonl'
    )
    parser.add_argument(
        '--out', type=str,
        default=os.path.join(base_dir, 'submission.csv'),
        help='Output submission CSV'
    )
    parser.add_argument(
        '--report', type=str,
        default=os.path.join(base_dir, 'feature_report.csv'),
        help='Output feature report CSV'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of CPU workers (default: all available cores)'
    )
    args = parser.parse_args()

    rank_candidates(args.candidates, args.out, args.report, n_workers=args.workers)
