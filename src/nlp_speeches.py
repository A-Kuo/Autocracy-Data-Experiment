"""NLP prototype: does negative / adversarial rhetoric spike in months of economic contraction?

Country: United Kingdom, Prime Minister's Office (speeches published on GOV.UK, Open Government Licence).
Economy: ONS monthly GDP index (ECY2, CVM SA). Contraction month = month-on-month change < 0.
NLP: spaCy (sentence split + lemmas) -> VADER sentence sentiment + adversarial-framing lexicon rate.

Run:  python src/nlp_speeches.py            (scrapes with a local cache, then analyses)
      python src/nlp_speeches.py --no-scrape (analyse cached speeches only)
"""
import argparse
import html
import io
import json
import re
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "speeches_uk_pm"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output"
for d in (CACHE, PROC, OUT):
    d.mkdir(parents=True, exist_ok=True)

START = "2023-09-29"  # last ~3 years
ORG = "prime-ministers-office-10-downing-street"
PMS = {"rishi-sunak": "Sunak", "keir-starmer": "Starmer", "andy-burnham": "Burnham"}
ONS_URL = "https://www.ons.gov.uk/generator?format=csv&uri=/economy/grossdomesticproductgdp/timeseries/ecy2/mgdp"
DELAY = 0.4  # seconds between requests, be polite

# Adversarial / blame / threat framing. Matched on spaCy lemmas. Deliberately small and transparent.
FRAMING = {
    "enemy", "threat", "fail", "failure", "chaos", "crisis", "broken", "betray", "betrayal", "corrupt",
    "blame", "attack", "wreck", "destroy", "cruel", "divide", "division", "extremist", "dangerous",
    "mess", "damage", "decline", "collapse", "scandal", "shambles", "reckless", "toxic", "hostile",
}
NEG_SENT = -0.25  # VADER compound at/below this counts as a "negative sentence"


def session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1.5)))
    s.headers["User-Agent"] = "Mozilla/5.0 (compatible; AutocracyExperiment/0.1; research prototype)"
    return s


def scrape():
    s = session()
    listing, start = [], 0
    while True:
        r = s.get("https://www.gov.uk/api/search.json", timeout=30, params={
            "filter_format": "speech", "filter_organisations": ORG, "filter_public_timestamp": f"from:{START}",
            "count": 100, "start": start, "order": "-public_timestamp",
            "fields": ["title", "public_timestamp", "link", "people"]}).json()
        listing += r["results"]
        start += 100
        if start >= r["total"]:
            break
    print(f"{len(listing)} speeches listed since {START}")
    for i, item in enumerate(listing, 1):
        f = CACHE / (item["link"].strip("/").split("/")[-1][:120] + ".json")
        if f.exists():
            continue
        try:
            c = s.get("https://www.gov.uk/api/content" + item["link"], timeout=20).json()
        except Exception as e:  # noqa: BLE001 - log and move on, cached files make reruns cheap
            print("skip", item["link"], str(e)[:60])
            continue
        body = BeautifulSoup(c.get("details", {}).get("body", ""), "html.parser").get_text(" ", strip=True)
        people = [p["slug"] for p in item.get("people", [])]
        f.write_text(json.dumps({"title": html.unescape(item["title"]), "date": item["public_timestamp"],
                                 "url": "https://www.gov.uk" + item["link"], "people": people, "text": body}),
                     encoding="utf8")
        print(f"[{i}/{len(listing)}] {item['public_timestamp'][:10]} {item['title'][:60]}")
        time.sleep(DELAY)


def load_speeches():
    rows = [json.loads(f.read_text(encoding="utf8")) for f in CACHE.glob("*.json")]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.date, utc=True).dt.tz_localize(None)
    df["leader"] = df.people.map(lambda p: next((PMS[x] for x in p if x in PMS), "other"))
    return df[df.text.str.len() > 200].sort_values("date").reset_index(drop=True)


def score(df):
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    vader = SentimentIntensityAnalyzer()
    out = []
    for doc in nlp.pipe(df.text, batch_size=8):
        sents = [s for s in doc.sents if len(s) >= 4]
        comp = np.array([vader.polarity_scores(s.text)["compound"] for s in sents]) if sents else np.array([0.0])
        toks = [t for t in doc if t.is_alpha]
        out.append({
            "tokens": len(toks),
            "mean_compound": comp.mean(),
            "neg_share": (comp <= NEG_SENT).mean(),
            "framing_per_1k": 1000 * sum(t.lemma_.lower() in FRAMING for t in toks) / max(len(toks), 1),
        })
    return pd.concat([df, pd.DataFrame(out)], axis=1)


def monthly_gdp():
    f = ROOT / "data" / "raw" / "ons_monthly_gdp_ecy2.csv"
    if not f.exists():
        f.write_text(session().get(ONS_URL, timeout=30).text, encoding="utf8")
    rows = [(m.group(1), float(m.group(2))) for m in re.finditer(r'"(\d{4} [A-Z]{3})","([\d.]+)"', f.read_text(encoding="utf8"))]
    g = pd.DataFrame(rows, columns=["m", "gdp_index"])
    g["month"] = pd.to_datetime(g.m, format="%Y %b").dt.to_period("M")
    g["gdp_mom"] = g.gdp_index.pct_change() * 100
    g["contraction"] = (g.gdp_mom < 0).astype(int)
    return g[["month", "gdp_index", "gdp_mom", "contraction"]]


def perm_test(x, y, n=20000, seed=0):
    """Two-sided permutation test on difference in means (y minus x)."""
    rng = np.random.default_rng(seed)
    obs = y.mean() - x.mean()
    pool = np.concatenate([x, y])
    diffs = np.empty(n)
    for i in range(n):
        rng.shuffle(pool)
        diffs[i] = pool[len(x):].mean() - pool[:len(x)].mean()
    return obs, (np.abs(diffs) >= abs(obs)).mean()


def analyse(sp, gdp):
    sp["month"] = sp.date.dt.to_period("M")
    w = sp.assign(w=sp.tokens)
    metrics = ["neg_share", "mean_compound", "framing_per_1k"]
    m = w.groupby("month").apply(
        lambda d: pd.Series({**{k: np.average(d[k], weights=d.w) for k in metrics}, "n_speeches": len(d),
                             "leader": d.leader.mode().iat[0]}), include_groups=False).reset_index()
    m = m.merge(gdp, on="month", how="left")
    m["contraction_lag1"] = m.month.map(gdp.set_index("month").contraction.shift(1))
    for k in metrics:  # leader-demeaned version strips out the Sunak/Starmer/Burnham style differences
        m[k + "_dm"] = m[k] - m.groupby("leader")[k].transform("mean")
    m = m.dropna(subset=["contraction"])
    m.to_csv(PROC / "uk_monthly_rhetoric_vs_gdp.csv", index=False)

    rows = []
    for k in metrics:
        for col, label in [("contraction", "same month"), ("contraction_lag1", "prior month")]:
            d = m.dropna(subset=[col])
            for suffix, variant in [("", "raw"), ("_dm", "leader-demeaned")]:
                a, b = d[d[col] == 0][k + suffix].values, d[d[col] == 1][k + suffix].values
                if len(a) < 3 or len(b) < 3:
                    continue
                diff, p = perm_test(a, b)
                rows.append({"metric": k, "contraction_timing": label, "variant": variant,
                             "mean_expansion": a.mean(), "mean_contraction": b.mean(), "diff": diff,
                             "p_perm": p, "n_exp": len(a), "n_con": len(b),
                             "spearman_vs_gdp_mom": d[k + suffix].corr(d.gdp_mom, method="spearman")})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "uk_rhetoric_results.csv", index=False)
    return m, res


def plot(m):
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    x = m.month.dt.to_timestamp()
    for a, k, t in zip(ax, ["neg_share", "framing_per_1k", "gdp_mom"],
                       ["Share of negative sentences (VADER <= -0.25)", "Adversarial-framing terms per 1,000 words",
                        "UK monthly GDP, % month-on-month (ONS)"]):
        a.plot(x, m[k], marker="o", color="black" if k != "gdp_mom" else "tab:blue")
        for xi, c in zip(x, m.contraction):
            if c == 1:
                a.axvspan(xi - pd.Timedelta(days=15), xi + pd.Timedelta(days=15), color="tab:red", alpha=0.15, lw=0)
        a.set_title(t, fontsize=10)
    ax[2].axhline(0, color="grey", lw=0.8)
    for lead, d in m.groupby("leader"):  # mark first month of each leader
        for a in ax:
            a.axvline(d.month.min().to_timestamp(), color="grey", ls="-.", lw=0.8)
        ax[0].text(d.month.min().to_timestamp(), ax[0].get_ylim()[1], f" {lead}", fontsize=8, va="top")
    fig.suptitle("UK PM speeches vs. economic contraction (red = GDP fell m/m)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "uk_rhetoric_vs_gdp.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-scrape", action="store_true")
    a = ap.parse_args()
    if not a.no_scrape:
        scrape()
    sp = score(load_speeches())
    sp.drop(columns=["text"]).to_csv(PROC / "uk_speech_scores.csv", index=False)
    print(f"{len(sp)} speeches scored; {sp.date.min().date()} -> {sp.date.max().date()}; leaders: {sp.leader.value_counts().to_dict()}")
    m, res = analyse(sp, monthly_gdp())
    print(f"{len(m)} months with speeches and GDP; contraction months: {int(m.contraction.sum())}")
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    plot(m)


if __name__ == "__main__":
    main()
