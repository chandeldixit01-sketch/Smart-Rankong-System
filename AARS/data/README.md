# Intelligent Candidate Discovery & Ranking — Redrob Hackathon

## Overview

Rule-based ranking engine that scores 100,000 candidate profiles against a
senior AI/ML Engineer job description for Redrob's retrieval and ranking
intelligence layer.

The system uses six scoring dimensions — skill match, role/career fit,
experience alignment, behavioral signals, profile consistency, and
location preference — combined with honeypot detection and consulting-firm
penalties to produce a top-100 shortlist with per-candidate reasoning.

## Quick Start

### Prerequisites

- Python 3.8+
- No external dependencies (stdlib only)

### Reproduce the submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This single command reads `candidates.jsonl` (100k candidates), scores and
ranks them, and writes:

- `submission.csv` — top 100 candidates with rank, score, and reasoning
- `feature_report.csv` — scores for all 100k candidates (for debugging)

### Validate the submission

```bash
python validate_submission.py submission.csv
```

## Architecture

### Phase A — Precomputation (offline, unconstrained)

Not required for this submission.  All scoring logic is rule-based and runs
directly at ranking time.  No embeddings, model weights, or pre-computed
artifacts are needed.

### Phase B — Ranking (`rank.py`, timed, sandboxed)

**Constraints**: ≤5 minutes, CPU only, no GPU, no network, ≤16GB RAM, ≤5GB disk.

The ranking pipeline:

1. **Stream-parse** `candidates.jsonl` one line at a time (constant memory)
2. **Honeypot detection** — 6 rules catching impossible profiles:
   - Expert in 3+ skills with 0 months duration
   - Job duration exceeding calendar time
   - Concurrent full-time jobs overlapping >90 days
   - Modern skill anachronisms (>6 years of experience with post-2020 tech)
   - AI title with 70%+ non-technical career history
   - Education timeline contradictions
3. **Six-dimension scoring**:
   - Skill match (0.30 weight) — JD-aligned keyword matching with
     proficiency, duration, endorsement, and platform assessment bonuses
   - Role/career fit (0.25) — title classification + career description
     keyword analysis
   - Experience alignment (0.12) — proximity to JD's 6-8 year sweet spot
   - Behavioral signals (0.13) — response rate, interview completion,
     recency, GitHub activity, recruiter saves
   - Consistency (0.12) — cross-validates claims across title, headline,
     summary, and career descriptions
   - Location (0.08) — mild bonus for JD-preferred Indian cities
4. **Consulting-firm penalty** (0.5×) — JD explicitly flags consulting-only
   careers as "we do NOT want"
5. **Sort** by score descending, candidate_id ascending for tie-breaks
6. **Write** top 100 to submission CSV with per-candidate reasoning

### Performance

~20 seconds for 100k candidates on a modern CPU.  Well within the 5-minute
budget.

## Scoring Weights Rationale

| Component   | Weight | JD Justification |
|-------------|--------|-------------------|
| Skills      | 0.30   | "Things you absolutely need" — strongest JD section |
| Role/Career | 0.25   | "The right answer involves reasoning about the gap between what the JD says and what it means" |
| Signals     | 0.13   | "A perfect-on-paper candidate who hasn't logged in for 6 months… is not available" |
| Experience  | 0.12   | "5-9 years" but "range, not a requirement" |
| Consistency | 0.12   | Keyword-stuffer trap detection (penalty, not quality signal) |
| Location    | 0.08   | "Pune/Noida-preferred but flexible" |

## Files

| File | Description |
|------|-------------|
| `rank.py` | Main ranking engine |
| `candidates.jsonl` | 100k candidate pool (input) |
| `submission.csv` | Top-100 ranking (output) |
| `feature_report.csv` | All-candidate scores (output, for debugging) |
| `validate_submission.py` | Format validator (provided by organizers) |
| `submission_metadata.yaml` | Submission metadata |
| `requirements.txt` | Dependencies (none beyond stdlib) |
