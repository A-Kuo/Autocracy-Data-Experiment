"""Option A (residual modeling) + Option B (two-stage gating) on the locally cached OWID/V-Dem panel.

Step 1  Gradient-boosted baseline: liberal democracy ~ economic/development features only.
        Predictions are OUT-OF-FOLD by country (GroupKFold) so a country's own history never
        leaks into its "expected" democracy level.
Step 2  residual = actual - expected. Negative = democracy is worse than the economy explains.
Step 3  Does the residual (level, 3y change) predict a backsliding onset in the NEXT 5 years?
        Country-grouped CV AUC vs. a economics-only logistic baseline.
        Rhetoric / unrest columns are optional: if data/processed/political_features.csv exists
        (entity, year, rhetoric_score, unrest_events) they are joined and used.
Option B  Rule-based gate (vulnerability tier x rhetoric/erosion) for comparison.

Outputs: data/processed/residual_panel.csv, output/residual_model.png
Run:  python src/residual_model.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "output"
ECON = ["hdi", "hdi_d1", "hdi_d3", "log_gdp", "gdp_g1", "gdp_g3", "strain"]
HORIZON, DROP = 5, 0.10  # backsliding onset: libdem falls >= 0.10 within the next 5 years
STRAIN = -0.02


def owid(name, col, out):
    d = pd.read_csv(RAW / f"{name}.csv")
    return d.rename(columns={col: out})[["entity", "year", out]]


def build_panel():
    p = owid("liberal-democracy-index", "libdem_vdem__estimate_best", "libdem")
    for name, col, out in [
        ("electoral-democracy-index", "electdem_vdem__estimate_best", "polyarchy"),
        ("human-development-index", "hdi__sex_total", "hdi"),
        ("gdp-per-capita-maddison-project-database", "gdp_per_capita", "gdp"),
    ]:
        p = p.merge(owid(name, col, out), on=["entity", "year"], how="inner")
    p = p[p.year >= 1990].sort_values(["entity", "year"]).reset_index(drop=True)
    g = p.groupby("entity")
    p["log_gdp"] = np.log(p.gdp)
    p["gdp_g1"] = g.log_gdp.diff(1)
    p["gdp_g3"] = g.log_gdp.diff(3)
    p["hdi_d1"], p["hdi_d3"] = g.hdi.diff(1), g.hdi.diff(3)
    p["strain"] = (p.gdp_g1 <= STRAIN).astype(float)
    # label: max drop from today's libdem over the next HORIZON years (needs full lookahead)
    fut = pd.concat([g.libdem.shift(-k) for k in range(1, HORIZON + 1)], axis=1)
    p["future_drop"] = p.libdem - fut.min(axis=1, skipna=False)
    p["onset"] = (p.future_drop >= DROP).astype(float).where(p.future_drop.notna())
    return p.dropna(subset=ECON + ["libdem"]).reset_index(drop=True)


def oof_baseline(p, n_splits=5):
    """Step 1+2: out-of-fold expected libdem from economics only; returns expected, R^2."""
    exp = np.zeros(len(p))
    for tr, te in GroupKFold(n_splits).split(p, groups=p.entity):
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)
        m.fit(p.loc[tr, ECON], p.loc[tr, "libdem"])
        exp[te] = m.predict(p.loc[te, ECON])
    return exp, r2_score(p.libdem, exp)


def cv_auc(p, feats, n_splits=5):
    d = p.dropna(subset=feats + ["onset"]).reset_index(drop=True)
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits).split(d, groups=d.entity):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        m.fit(d.loc[tr, feats], d.loc[tr, "onset"])
        oof[te] = m.predict_proba(d.loc[te, feats])[:, 1]
    return roc_auc_score(d.onset, oof), len(d), d.onset.mean()


def gate(p, rhetoric_high=0.7, vuln_cut=(0.33, 0.66)):
    """Option B: vulnerability tier from strain + HDI momentum; warn when tier high AND erosion/rhetoric high."""
    vuln = (-p.gdp_g3.clip(-0.3, 0.3)).rank(pct=True) * 0.5 + (-p.hdi_d3).rank(pct=True) * 0.5
    tier = pd.cut(vuln, [-1, *vuln_cut, 2], labels=["Low", "Mid", "High"])
    erosion = (-p.res_d3).rank(pct=True)  # institutional erosion proxy: residual worsening
    if "rhetoric_score" in p:
        erosion = 0.5 * erosion + 0.5 * p.rhetoric_score.rank(pct=True)
    return tier, (tier == "High") & (erosion >= rhetoric_high)


def main():
    p = build_panel()
    extra = PROC / "political_features.csv"
    if extra.exists():
        p = p.merge(pd.read_csv(extra), on=["entity", "year"], how="left")
    p["expected"], r2 = oof_baseline(p)
    p["residual"] = p.libdem - p.expected
    g = p.groupby("entity").residual
    p["res_d3"] = g.diff(3)
    print(f"Step 1: economics-only baseline, country-grouped OOF R^2 = {r2:.3f}  (n={len(p):,})")

    base = ECON
    with_res = ECON + ["residual", "res_d3"]
    optional = [c for c in ("rhetoric_score", "unrest_events") if c in p]
    p = p.dropna(subset=with_res + ["onset"]).reset_index(drop=True)  # identical rows for every model
    print(f"Step 3 (onset = libdem drop >= {DROP} within {HORIZON}y), grouped-CV AUC:")
    for name, f in [("economics only", base), ("+ residual", with_res), ("+ residual + political", with_res + optional)]:
        auc, n, rate = cv_auc(p, f)
        print(f"  {name:<24} AUC={auc:.3f}  n={n:,}  onset rate={rate:.3f}")
    # sanity: the raw level of democracy is itself a strong predictor; report it so residual gain isn't oversold
    auc, *_ = cv_auc(p, base + ["libdem"])
    print(f"  {'economics + raw libdem':<24} AUC={auc:.3f}  (reference)")

    p["tier"], p["warning"] = gate(p)
    w = p.dropna(subset=["onset"])
    print(f"Option B gate: fires on {w.warning.mean():.1%} of country-years; "
          f"onset rate when fired {w[w.warning].onset.mean():.3f} vs {w[~w.warning].onset.mean():.3f} otherwise")

    p.to_csv(PROC / "residual_panel.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(p.expected, p.libdem, s=3, alpha=0.3)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(xlabel="expected libdem (economics only, OOF)", ylabel="actual libdem", title=f"Step 1 baseline, R²={r2:.2f}")
    for c in ["Hungary", "Venezuela", "Turkey", "Poland"]:
        d = p[p.entity == c]
        ax[1].plot(d.year, d.residual, label=c)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(title="Residual: actual - economically expected democracy", xlabel="year")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "residual_model.png", dpi=130)


if __name__ == "__main__":
    main()
