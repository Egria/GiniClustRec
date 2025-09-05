import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any
from scipy.stats import norm
import statsmodels.api as sm

# ---------- small utilities ----------
def bh_fdr(p: pd.Series) -> pd.Series:
    p = pd.Series(p, dtype=float)
    n = p.size
    order = np.argsort(p.values)
    ranks = np.empty(n, dtype=int); ranks[order] = np.arange(1, n+1)
    q_ord = (p.values[order] * n / ranks)
    # monotone
    q_ord = np.minimum.accumulate(q_ord[::-1])[::-1]
    q = np.empty_like(q_ord); q[order] = q_ord
    return pd.Series(np.clip(q, 0, 1), index=p.index)

def p_to_z_two_sided(p):
    p = np.clip(np.asarray(p, float), 1e-300, 1.0)
    return norm.isf(p/2.0)

def mean_in_log2space(x, pseudo_count=1.0):
    x = np.asarray(x, float)
    lin = np.power(2.0, x) - pseudo_count
    lin = np.clip(lin, 0.0, None)
    return float(np.log2(lin.mean() + pseudo_count))

# ---------- the "zlm-like" class ----------
@dataclass
class ZLMResult:
    genes: pd.Index
    # per-gene results
    table: pd.DataFrame          # columns defined below
    cells_meta: pd.DataFrame     # per-cell design (for reference)

    def summary(self, logFC: bool = True):
        """
        Mimic MAST summary() output a bit:
        returns an object with attribute 'datatable' (a DataFrame).
        """
        dt = self.table.copy()
        if logFC and {"log2.mean.group0","log2.mean.group1"}.issubset(dt.columns):
            dt["logFC"] = dt["log2.mean.group1"] - dt["log2.mean.group0"]
        # Reorder like MAST-ish
        cols = [c for c in [
            "log2.mean.group0","log2.mean.group1","logFC",
            "beta_logit","se_logit","p_logit",
            "beta_cont","se_cont","p_cont",
            "p_hurdle","q_hurdle"
        ] if c in dt.columns]
        dt = dt[cols]
        # wrap to mimic $datatable
        return type("Summary", (), {"datatable": dt})

    def lrTest(self, factor: str = "Population"):
        g = np.asarray(self.genes)  # shape (N,)
        n = g.shape[0]

        # Pull p-value arrays; if missing, fill with NaNs
        p_logit = self.table["p_logit"].to_numpy() if "p_logit" in self.table else np.full(n, np.nan)
        p_cont = self.table["p_cont"].to_numpy() if "p_cont" in self.table else np.full(n, np.nan)
        p_hurdle = self.table["p_hurdle"].to_numpy() if "p_hurdle" in self.table else np.full(n, np.nan)

        primerid = np.tile(g, 3)  # [g1..gN, g1..gN, g1..gN]
        test_type = np.repeat(["discrete", "continuous", "hurdle"], n)
        pvals = np.concatenate([p_logit, p_cont, p_hurdle])

        df = pd.DataFrame({
            "primerid": primerid,
            "test.type": test_type,
            "Pr(>Chisq)": pvals
        })
        return df
# ---------- the zlm-like fitter ----------
def zlm_single_cell_assay(
    data_log2: pd.DataFrame,        # genes x cells
    group: pd.Series,               # cells -> two groups
    pseudo_count: float = 1.0,
    covariates: Optional[pd.DataFrame] = None,
    add_cdr: bool = True,
    method: str = "bayesglm",
    ebayes: bool = True,
    robust_se: bool = True,
    min_pos: int = 5
) -> ZLMResult:
    """
    Simulate MAST's zlm.SingleCellAssay(~ Population, ..., method='bayesglm').
    - Discrete: Logit(detected ~ group + CDR + covars)
    - Continuous: OLS(log2 | detected>0 ~ group + CDR + covars), HC3 SEs
    - Combine p-values via Stouffer Z -> 'hurdle' p
    - If ebayes: shrink continuous SEs by a global factor (simple EB proxy)

    Returns a ZLMResult with .summary() and .lrTest() helpers.
    """
    # Align group/covariates to columns
    group = group.reindex(data_log2.columns)
    if group.isna().any():
        missing = group[group.isna()].index.tolist()
        raise ValueError(f"group missing for: {missing[:5]} ...")

    # Make design matrix per cell
    X = pd.DataFrame(index=data_log2.columns)
    # Force two-category coding 0/1 with stable order
    cats = pd.Categorical(group)
    if len(cats.categories) != 2:
        raise ValueError(f"Need exactly 2 groups; got {list(cats.categories)}")
    grp_codes = pd.Categorical(group, categories=list(cats.categories)).codes
    X["group"] = grp_codes

    # Add CDR if requested: fraction of genes detected (> log2(pseudo_count))
    if add_cdr:
        thr = np.log2(pseudo_count)
        cdr = (data_log2.gt(thr).sum(axis=0) / data_log2.shape[0]).astype(float)
        X["cdr"] = cdr.values

    # Add extra covariates
    if covariates is not None:
        covariates = covariates.reindex(X.index)
        X = pd.concat([X, covariates], axis=1)

    Xc = sm.add_constant(X, has_constant="add")

    # Precompute group means in log2 space (back/forward)
    g0 = (grp_codes == 0)
    g1 = (grp_codes == 1)
    mean0 = data_log2.loc[:, g0].apply(mean_in_log2space, axis=1, pseudo_count=pseudo_count)
    mean1 = data_log2.loc[:, g1].apply(mean_in_log2space, axis=1, pseudo_count=pseudo_count)

    # Run per-gene models
    rows = []
    # For simple EB shrink on continuous part, store raw SEs:
    se_cont_raw = []

    for gene, y in data_log2.iterrows():
        # Discrete/detected
        detected = (y.values > np.log2(pseudo_count)).astype(int)

        # --- Logistic (discrete) ---
        p_logit = np.nan; beta_logit = np.nan; se_logit = np.nan
        try:
            logit_fit = sm.Logit(detected, Xc).fit(disp=0, maxiter=200)
            beta_logit = float(logit_fit.params.get("group", np.nan))
            se_logit   = float(logit_fit.bse.get("group", np.nan))
            p_logit    = float(logit_fit.pvalues.get("group", np.nan))
        except Exception:
            pass

        # --- Continuous (positive-only) ---
        p_cont = np.nan; beta_cont = np.nan; se_cont = np.nan
        pos_idx = detected.astype(bool)
        if pos_idx.sum() >= min_pos and (~pos_idx).sum() >= 1:
            y_pos = y.values[pos_idx].astype(float)
            X_pos = sm.add_constant(X.loc[pos_idx, :], has_constant="add")
            try:
                ols = sm.OLS(y_pos, X_pos)
                fit = ols.fit(cov_type="HC3") if robust_se else ols.fit()
                beta_cont = float(fit.params.get("group", np.nan))
                se_cont   = float(fit.bse.get("group", np.nan))
                p_cont    = float(fit.pvalues.get("group", np.nan))
            except Exception:
                pass

        se_cont_raw.append(se_cont)

        # --- Combine (Stouffer Z, weighted by sqrt(n)) ---
        Zs, ws = [], []
        if np.isfinite(p_logit):
            Zs.append(p_to_z_two_sided(p_logit)); ws.append(np.sqrt(len(detected)))
        if np.isfinite(p_cont):
            Zs.append(p_to_z_two_sided(p_cont));  ws.append(np.sqrt(pos_idx.sum()))
        if len(Zs) == 0:
            p_hurdle = np.nan
        elif len(Zs) == 1:
            p_hurdle = float(2 * norm.sf(Zs[0]))
        else:
            Zc = np.dot(ws, Zs) / np.sqrt(np.sum(np.square(ws)))
            p_hurdle = float(2 * norm.sf(Zc))

        rows.append({
            "gene": gene,
            "log2.mean.group0": mean0.loc[gene],
            "log2.mean.group1": mean1.loc[gene],
            "beta_logit": beta_logit, "se_logit": se_logit, "p_logit": p_logit,
            "beta_cont": beta_cont,   "se_cont": se_cont,   "p_cont": p_cont,
            "p_hurdle": p_hurdle
        })

    out = pd.DataFrame(rows).set_index("gene")

    # Optional: simple EB variance moderation for continuous part
    if ebayes:
        se = pd.Series(se_cont_raw, index=out.index, dtype="float64")
        # Global prior: shrink extreme SEs toward median (very simple proxy)
        med = np.nanmedian(se.values)
        if np.isfinite(med) and med > 0:
            shrink = 0.25  # 0=no shrink; 1=full shrink to median
            se_mod = np.where(np.isfinite(se), (1-shrink)*se + shrink*med, se)
            # Recompute p_cont using moderated SE if beta exists
            with np.errstate(invalid="ignore", divide="ignore"):
                tvals = out["beta_cont"].values / se_mod
                p_cont_mod = 2*norm.sf(np.abs(tvals))
            out["se_cont"] = se_mod
            out["p_cont"]  = np.where(np.isfinite(p_cont_mod), p_cont_mod, out["p_cont"])

            # Rebuild hurdle p with moderated continuous p
            p_logit = out["p_logit"].values
            p_cont  = out["p_cont"].values
            Zs = np.c_[p_to_z_two_sided(p_logit), p_to_z_two_sided(p_cont)]
            # weights: sqrt(n_all), sqrt(n_pos) (approximate by med across genes)
            n_all = data_log2.shape[1]
            # crude per-gene n_pos:
            n_pos = (data_log2.gt(np.log2(pseudo_count)).sum(axis=1).clip(lower=1)).values
            w = np.c_[np.sqrt(n_all) * np.ones_like(n_pos), np.sqrt(n_pos)]
            Zc = (Zs * w).sum(1) / np.sqrt((w**2).sum(1))
            out["p_hurdle"] = 2*norm.sf(np.abs(Zc))

    # FDR for hurdle p
    out["q_hurdle"] = bh_fdr(out["p_hurdle"])

    return ZLMResult(
        genes=out.index,
        table=out,
        cells_meta=X.assign(const=1.0)[["const"] + [c for c in X.columns]],  # show design used
    )
