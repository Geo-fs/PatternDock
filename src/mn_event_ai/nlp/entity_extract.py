from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Very lightweight entity extraction:
# - people/org/location-ish capitalization heuristics
# - regex for common structured entities (roads, weather terms, etc.)
# Optional: spaCy if installed.

TITLECASE_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b")
ACRONYM_RE = re.compile(r"\b([A-Z]{2,6})\b")

URLISH_RE = re.compile(r"https?://\S+", re.I)

# You can extend these patterns without changing downstream code.
PATTERNS = {
    "road": re.compile(r"\b(I-\s?\d{1,3}|US-\s?\d{1,3}|Hwy\s?\d{1,4}|Highway\s?\d{1,4})\b", re.I),
    "weather": re.compile(
        r"\b(tornado|blizzard|winter storm|ice storm|flood|flooding|heat advisory|air quality)\b",
        re.I,
    ),
    "gov": re.compile(r"\b(governor|senate|house|mayor|city council|county board)\b", re.I),
}


@dataclass
class Entities:
    people: List[str]
    orgs: List[str]
    places: List[str]
    misc: Dict[str, List[str]]


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        k = x.strip()
        if not k:
            continue
        norm = k.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(k)
    return out


def _try_spacy(text: str) -> Optional[Entities]:
    try:
        import spacy  # type: ignore
    except Exception:
        return None

    # Try loading a small model if present, otherwise bail gracefully.
    nlp = None
    for name in ("en_core_web_sm", "en_core_web_md"):
        try:
            nlp = spacy.load(name)  # type: ignore
            break
        except Exception:
            continue
    if nlp is None:
        return None

    doc = nlp(text)
    people, orgs, places = [], [], []
    for ent in doc.ents:
        if ent.label_ in ("PERSON",):
            people.append(ent.text)
        elif ent.label_ in ("ORG",):
            orgs.append(ent.text)
        elif ent.label_ in ("GPE", "LOC", "FAC"):
            places.append(ent.text)

    return Entities(
        people=_dedupe_keep_order(people),
        orgs=_dedupe_keep_order(orgs),
        places=_dedupe_keep_order(places),
        misc={},
    )


def extract_entities(title: str, summary: str = "") -> Entities:
    text = f"{title or ''}\n{summary or ''}".strip()
    text = URLISH_RE.sub("", text)

    sp = _try_spacy(text)
    if sp is not None:
        # Also add misc regex hits even when using spaCy (cheap signal).
        misc: Dict[str, List[str]] = {}
        for k, pat in PATTERNS.items():
            misc[k] = _dedupe_keep_order(pat.findall(text))
        return Entities(people=sp.people, orgs=sp.orgs, places=sp.places, misc=misc)

    # Heuristic fallback (no heavy deps).
    titlecase = TITLECASE_PHRASE_RE.findall(text)
    acronyms = ACRONYM_RE.findall(text)

    # Super rough categorization:
    # - if it contains Inc/Co/Corp/Department etc → org
    # - if it contains "City" / "County" / "Lake" etc → place
    people, orgs, places = [], [], []

    org_markers = ("Inc", "Co", "Corp", "Company", "Department", "University", "School", "Police", "Sheriff")
    place_markers = ("City", "County", "Lake", "River", "Park", "Airport", "Township", "Heights")

    for phrase in titlecase:
        toks = phrase.split()
        if any(t in org_markers for t in toks):
            orgs.append(phrase)
        elif any(t in place_markers for t in toks):
            places.append(phrase)
        else:
            # Wild guess: 2+ words that look like names are probably people.
            if len(toks) >= 2:
                people.append(phrase)

    # Acronyms are usually org-ish.
    orgs.extend(acronyms)

    misc: Dict[str, List[str]] = {}
    for k, pat in PATTERNS.items():
        misc[k] = _dedupe_keep_order(pat.findall(text))

    return Entities(
        people=_dedupe_keep_order(people),
        orgs=_dedupe_keep_order(orgs),
        places=_dedupe_keep_order(places),
        misc=misc,
    )
