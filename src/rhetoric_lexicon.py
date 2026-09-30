"""Tier 1 of the rhetoric pipeline: regex lexicons for authoritarian tropes.

Each category yields hits per 1,000 words. Tier 2 (transformer framing/polarization) is a hook:
pass `polarization_fn(sentence) -> float in [0,1]` to `score()` to re-weight only flagged sentences,
so the expensive model runs on a small, interpretable subset.
Run `python src/rhetoric_lexicon.py` for a self-test.
"""
import re

LEXICON = {
    "anti_elite": [r"enem(?:y|ies) of the people", r"corrupt (?:elites?|establishment|politicians)",
                   r"\bthe elites?\b", r"swamp", r"traitors? (?:in|within)"],
    "scapegoat": [r"illegal (?:invaders?|aliens)", r"ethnic traitors?", r"\bvermin\b", r"\binfest(?:ed|ation)\b",
                  r"foreign(?:ers)? (?:are )?(?:stealing|taking)"],
    "delegitimize": [r"fake news", r"rigged (?:elections?|system|courts?|vote)", r"stolen election",
                     r"so-called (?:judges?|courts?|media)", r"enem(?:y|ies) within", r"activist judges?"],
}
PATTERNS = {k: re.compile("|".join(v), re.I) for k, v in LEXICON.items()}
SENT = re.compile(r"(?<=[.!?])\s+")


def score(text, polarization_fn=None):
    """Return per-category hits/1000 words, total, and (optionally) polarization-weighted total."""
    words = max(len(text.split()), 1)
    out = {f"{k}_per_1k": 1000 * len(p.findall(text)) / words for k, p in PATTERNS.items()}
    out["lexicon_per_1k"] = sum(out.values())
    if polarization_fn:
        flagged = [s for s in SENT.split(text) if any(p.search(s) for p in PATTERNS.values())]
        out["weighted_per_1k"] = 1000 * sum(polarization_fn(s) for s in flagged) / words
    return out


if __name__ == "__main__":
    assert score("The rigged election was pushed by fake news and corrupt elites.")["lexicon_per_1k"] > 0
    assert score("We must protect our borders and grow the economy.")["lexicon_per_1k"] == 0
    assert score("Illegal invaders.", polarization_fn=lambda s: 0.9)["weighted_per_1k"] > 0
    print("ok")
