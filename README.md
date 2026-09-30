# Autocracy Data Experiment: initial static heuristic index

A first-pass, deliberately simple test of one hypothesis: **democracies slide toward autocracy under economic strain.**
Everything here is a prototype with hand-picked weights. Treat outputs as illustrations, not findings.

## Layout

| Path | What it is |
|---|---|
| `src/build_index.py` | Parts 1-2: pulls V-Dem, HDI and GDP data, builds the index, writes the workbook and plots |
| `src/nlp_speeches.py` | Part 3: scrapes UK Prime Minister speeches, scores them with spaCy + VADER, compares against monthly GDP |
| `output/democratic_resilience_index.xlsx` | The spreadsheet: editable **Weights** sheet, **Regression**, **Summary**, one sheet per case (live formulas + charts) |
| `output/index_*.png` | Index plots per case; `uk_rhetoric_vs_gdp.png` for the NLP prototype |
| `data/raw`, `data/processed` | Downloaded source data (cached) and flat CSV outputs |

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/build_index.py
python src/nlp_speeches.py
```

## 1. Case studies

Weimar Germany (1919-38), Venezuela (1990-2023), Hungary (1990-2023). Data comes from Our World in Data's
grapher CSVs, which republish V-Dem (electoral and liberal democracy indices), UNDP HDI (1990+), the Escosura
historical HDI (1870-2020) and the Maddison Project GDP per capita.

## 2. Static index (0-100, higher = healthier)

`Index = 100 x (0.30 x electoral democracy + 0.30 x liberal democracy + 0.20 x HDI + 0.20 x HDI-adjusted democracy)`

- **HDI regression:** pooled OLS of liberal democracy on HDI over every country-year. The historical fit
  (Escosura HDI) has R² 0.79, n=3,030. The modern fit (UNDP HDI, 1990+) has R² 0.35, n=5,584.
- **HDI-adjusted democracy:** `0.5 + (actual liberal democracy - value predicted from HDI)`, clipped to 0-1.
  This asks whether a country is doing better or worse on democracy than its development level would predict.
- **Strain year:** GDP per capita falls 2% or more year on year (threshold editable on the Weights sheet).
- Change the yellow cells on the **Weights** sheet and each country sheet, and its chart, recalculates.
  The formulas were checked against the Python computation and match.

### What the plots show

- **Germany:** the index was flat near 50 through the 1920s. It fell with the 1930-32 slump and then dropped
  sharply in 1933. Strain led into collapse, which supports the hypothesis.
- **Hungary:** flat around 78 until 2009, when there was a strain year. It then declines steadily from 2010
  and is still falling in 2023. Strain looks like the trigger, but the slide continues through the strong-growth 2010s.
- **Venezuela:** the sharp fall starts in 1998-99, before the deepest strain (2014-2020). Most of the later
  strain comes *after* the index had already collapsed, so causation probably runs both ways.

**Verdict:** the raw data is consistent with the hypothesis for Germany and partly for Hungary. It does not
cleanly support it for Venezuela. With n=3 and hand-chosen cases this can't confirm anything.

## 3. NLP prototype (UK PM speeches vs. monthly GDP)

I used the UK because GOV.UK exposes an open, structured API with full transcripts (Open Government Licence). The
scrapers I tried for US sources were blocked, and whitehouse.gov has almost no transcripts.

- **Sample:** 138 PM's-office speeches from Oct 2023 to Sep 2026 (Sunak 31, Starmer 102, Burnham 4, other 1),
  across 34 months.
- **Economy:** ONS monthly GDP index; a contraction month is one where GDP fell month on month (9 of 34).
- **Scores per speech:** share of negative sentences (VADER on spaCy sentences), mean sentiment, and the rate of
  adversarial/blame terms per 1,000 words (spaCy lemmas; the lexicon is in the script).
- **Test:** permutation test on monthly means, contraction vs. other months, same month and one-month lag,
  raw and demeaned within each leader.

**Result: no statistically detectable link (all p > 0.2).** Negative-sentence share was 0.159 in expansion
months and 0.161 in contraction months. Adversarial-framing rate was actually *lower* in same-month contraction
(2.4 vs 3.3 per 1,000 words, p=0.21). The visible spike in spring 2026 coincides with a contraction month, but
the series is noisy and this is one episode.

## Limits to keep in mind

- **VADER is a weak fit for political speech.** It is built for social media. Swapping in a HuggingFace model
  (e.g. a transformer sentiment or stance classifier) is the obvious upgrade. I didn't install it because PyTorch on this Python 3.14 setup is heavy.
- **Topic confound:** speeches about war or security use "threat" words regardless of the economy.
- **The UK is not a backsliding case.** It tests the tooling, not the hypothesis. The next step is to run the same
  scraper on a country in the index's "slide" zone.
- **Monthly GDP is a small sample.** No NBER-style recession occurs in the window; "contraction" here means small
  month-on-month dips, which are often revised.
- **Index caveats:** Escosura HDI is sparse before 1950 (linearly interpolated between benchmark years),
  and its scale differs from UNDP's. The two HDI series are spliced at 1990 for the modern cases, which use UNDP only.
  The weights and thresholds are arbitrary starting points.
- **Reverse causation:** autocratizing governments also damage their economies (see Venezuela), so a strain-then-slide
  pattern is not proof of direction.

## Next steps

1. Add more cases, including non-slides (e.g. Spain, Portugal, South Korea), so the index can be tested for false positives.
2. Replace hand weights with a fitted model (e.g. logit on "V-Dem drops 0.1+ in 5 years") and validate out of sample.
3. Upgrade the NLP: HuggingFace stance/toxicity model, then try a leader in a backsliding country.
