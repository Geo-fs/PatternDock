from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from mn_event_ai.nlp.features import make_text_field


IN_PATH = Path("data/processed/articles.jsonl")
OUT_DIR = Path("data/labels")
OUT_LABELS = OUT_DIR / "weak_labels.jsonl"
OUT_MAP = OUT_DIR / "label_map.json"
OUT_REPORT = OUT_DIR / "weak_label_report.json"


LABELS = [
    "weather",
    "traffic_roads",
    "crime_public_safety",
    "politics_gov",
    "education",
    "business_economy",
    "health",
    "events_culture",
    "sports",
    "other",
]


# --- Keyword rules (simple on purpose; we'll refine as data grows) ---
RULES: Dict[str, List[re.Pattern]] = {
    "weather": [
        re.compile(r"\b(blizzard|snowstorm|winter storm|ice storm|freezing rain|hail|tornado|severe storm)\b", re.I),
        re.compile(r"\b(flood|flooding|flash flood|river crest|storm surge)\b", re.I),
        re.compile(r"\b(heat advisory|excessive heat|wind chill|air quality alert|dense fog)\b", re.I),
        re.compile(r"\b(nws|national weather service|warning|watch|advisory)\b", re.I),
    ],
    "traffic_roads": [
        re.compile(r"\b(crash|pileup|collision|rollover|spin[- ]?out)\b", re.I),
        re.compile(r"\b(road closure|closed|lane closure|detour|ramp closed)\b", re.I),
        re.compile(r"\b(traffic|congestion|gridlock|commute)\b", re.I),
        re.compile(r"\b(i-?\s?35|i-?\s?94|i-?\s?494|i-?\s?694|highway|hwy|county road|bridge)\b", re.I),
    ],
    "crime_public_safety": [
        re.compile(r"\b(shooting|homicide|murder|stabbing|assault)\b", re.I),
        re.compile(r"\b(arrested|charged|indicted|sentenced|trial|court)\b", re.I),
        re.compile(r"\b(police|sheriff|state patrol|bca|investigation)\b", re.I),
        re.compile(r"\b(fire|house fire|wildfire|explosion)\b", re.I),
        re.compile(r"\b(amber alert|missing person)\b", re.I),
    ],
    "politics_gov": [
        re.compile(r"\b(governor|legislature|senate|house|mayor|city council|county board)\b", re.I),
        re.compile(r"\b(bill|law|executive order|budget|appropriation|tax|levy)\b", re.I),
        re.compile(r"\b(election|primary|ballot|campaign|vote|voter)\b", re.I),
        re.compile(r"\b(mn\.gov|department of|state of minnesota)\b", re.I),
    ],
    "education": [
        re.compile(r"\b(school|district|superintendent|students|teacher|classroom)\b", re.I),
        re.compile(r"\b(university|minnesota state|umn|college|campus)\b", re.I),
        re.compile(r"\b(board meeting|school board)\b", re.I),
    ],
    "business_economy": [
        re.compile(r"\b(economy|inflation|jobs|unemployment|hiring|layoff|strike|union)\b", re.I),
        re.compile(r"\b(earnings|revenue|profit|quarter|ipo|merger|acquisition)\b", re.I),
        re.compile(r"\b(company|startup|industry|manufacturing|retail)\b", re.I),
        re.compile(r"\b(housing market|mortgage|rent|home prices)\b", re.I),
    ],
    "health": [
        re.compile(r"\b(covid|flu|outbreak|health department|public health)\b", re.I),
        re.compile(r"\b(hospital|clinic|patient|ambulance|ems)\b", re.I),
        re.compile(r"\b(opioid|overdose|drug|substance use)\b", re.I),
        re.compile(r"\b(vaccine|vaccination)\b", re.I),
    ],
    "events_culture": [
        re.compile(r"\b(festival|fair|concert|show|tour|performance)\b", re.I),
        re.compile(r"\b(restaurant|dining|brewery|bar|chef)\b", re.I),
        re.compile(r"\b(art|arts|theater|museum|exhibit)\b", re.I),
        re.compile(r"\b(event|opening|grand opening)\b", re.I),
    ],
    "sports": [
        re.compile(r"\b(vikings|twins|timberwolves|wild|lynx|gophers)\b", re.I),
        re.compile(r"\b(nfl|nba|mlb|nhl|ncaa|playoffs|season)\b", re.I),
        re.compile(r"\b(game|match|tournament|score|win|loss)\b", re.I),
    ],
}


# --- Source priors (optional boosts) ---
SOURCE_PRIORS: Dict[str, List[str]] = {
    "mpr_politics": ["politics_gov"],
    "mpr_business": ["business_economy"],
    "mpr_education": ["education"],
    "mpr_arts": ["events_culture"],
    "mn_dhs_news": ["health", "politics_gov"],
    "mn_admin_news": ["politics_gov"],
}


def apply_rules(text: str) -> Set[str]:
    labels: Set[str] = set()
    for lab, pats in RULES.items():
        for p in pats:
            if p.search(text):
                labels.add(lab)
                break
    return labels


def apply_source_priors(source_id: str) -> Set[str]:
    return set(SOURCE_PRIORS.get(source_id, []))


def choose_final_labels(text_labels: Set[str], prior_labels: Set[str]) -> Set[str]:
    # Combine rule + prior labels
    labels = set(text_labels) | set(prior_labels)

    # If nothing matched, call it "other"
    if not labels:
        labels = {"other"}

    # If we have real labels, drop "other"
    if "other" in labels and len(labels) > 1:
        labels.remove("other")

    return labels


def load_articles(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    articles = load_articles(IN_PATH)

    label_to_idx = {lab: i for i, lab in enumerate(LABELS)}
    OUT_MAP.write_text(json.dumps(label_to_idx, indent=2), encoding="utf-8")

    counts = Counter()
    multi_label_counts = Counter()
    by_source = defaultdict(int)

    with OUT_LABELS.open("w", encoding="utf-8") as out:
        for a in articles:
            text = make_text_field(a.get("title", ""), a.get("summary", ""))
            src = a.get("source_id", "unknown")

            text_labels = apply_rules(text)
            prior_labels = apply_source_priors(src)
            final_labels = choose_final_labels(text_labels, prior_labels)

            for lab in final_labels:
                counts[lab] += 1
            multi_label_counts[len(final_labels)] += 1
            by_source[src] += 1

            row = {
                "id": a["id"],
                "source_id": src,
                "published_at": a.get("published_at"),
                "text": text,
                "labels": sorted(final_labels),
                "label_indices": [label_to_idx[l] for l in sorted(final_labels)],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "articles_in": len(articles),
        "labels": LABELS,
        "label_counts": dict(counts),
        "multi_label_histogram": {str(k): v for k, v in multi_label_counts.items()},
        "by_source_counts": dict(by_source),
        "coverage_pct_non_other": round(
            100.0 * (len(articles) - counts.get("other", 0)) / max(len(articles), 1), 2
        ),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"wrote": str(OUT_LABELS), "articles": len(articles), "coverage_non_other_pct": report["coverage_pct_non_other"]}, indent=2))


if __name__ == "__main__":
    main()
