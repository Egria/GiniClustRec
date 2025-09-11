import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.api import add_constant, OLS, Logit
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from sklearn.metrics import roc_auc_score

# ---------- helpers ----------
def mean_in_log2space(x, pseudo_count=1.0):
    x = np.asarray(x, dtype=float)
    lin = np.power(2.0, x) - pseudo_count
    lin = np.clip(lin, 0.0, None)
    return np.log2(lin.mean() + pseudo_count)

def m_auc(data_m: pd.DataFrame, group: pd.Series) -> pd.Series:
    group = group.reindex(data_m.columns)
    def _auc(row):
        y = group.values
        x = row.values.astype(float)
        # make labels 0/1
        lab = pd.Series(y).astype('category').cat.codes.to_numpy()
        try:
            return float(roc_auc_score(lab, x))
        except Exception:
            return 0.5
    return data_m.apply(_auc, axis=1).fillna(0.5)

def bh_fdr(p):
    p = pd.Series(p, dtype=float)
    n = p.size
    order = p.argsort()
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n+1)
    q = (p.iloc[order] * n / ranks).cummin()[::-1].cummin()[::-1]  # safe double-cummin
    q_full = pd.Series(index=p.index, dtype=float)
    q_full.iloc[order] = q.values
    return q_full.clip(upper=1.0)

# two-sided p → Z (absolute); we’ll restore direction from continuous beta
def p_to_z_two_sided(p):
    p = np.clip(p, 1e-300, 1.0)  # avoid inf
    return norm.isf(p/2.0)

# ---------- core: MAST-like hurdle ----------
def mast_like(
    data_log2: pd.DataFrame,
    group: pd.Series,
    pseudo_count: float = 1.0,
    covariates: pd.DataFrame | None = None,
    add_cdr: bool = True,
    robust_se: bool = True
) -> pd.DataFrame:
    """
    Returns a per-gene table with:
        log2.mean.group0, log2.mean.group1, log2_fc,
        n0_detect, n1_detect,
        p_logit, p_cont, p_comb, q_comb,
        beta_logit, beta_cont, auc
    """
    # Align group & covariates to columns
    group = group.reindex(data_log2.columns)
    if group.isna().any():
        missing = group[group.isna()].index.tolist()
        raise ValueError(f"group has missing labels for columns: {missing[:5]} ...")

    # Build design matrix columns
    X_cells = pd.DataFrame(index=data_log2.columns)
    # binary group: 0/1, keep a stable order (map first category -> 0, second -> 1)
    cats = pd.Categorical(group)
    if len(cats.categories) != 2:
        raise ValueError(f"mast_like expects exactly 2 groups; got {list(cats.categories)}")
    group01 = pd.Series(pd.Categorical(group, categories=cats.categories).codes, index=group.index)
    X_cells["group"] = group01

    # cellular detection rate (fraction > 0 raw) – approximate using log2(raw + pseudo) > log2(pseudo)
    if add_cdr:
        thresh = np.log2(pseudo_count)  # cells with value > thresh are detected
        cdr = (data_log2.gt(thresh).sum(axis=0) / data_log2.shape[0]).astype(float)
        X_cells["cdr"] = cdr

    # extra covariates
    if covariates is not None:
        covariates = covariates.reindex(X_cells.index)
        X_cells = pd.concat([X_cells, covariates], axis=1)

    # Precompute group-wise means (in log2 space, on the original scale, then back)
    levels = list(cats.categories)
    g0_mask = (group == levels[0])
    g1_mask = (group == levels[1])
    mean0 = data_log2.loc[:, g0_mask].apply(mean_in_log2space, axis=1, pseudo_count=pseudo_count)
    mean1 = data_log2.loc[:, g1_mask].apply(mean_in_log2space, axis=1, pseudo_count=pseudo_count)
    log2_fc = mean1 - mean0

    # AUC per gene (optional but useful and in your R flow)
    auc_series = m_auc(data_log2, group)

    # storage
    results = []

    # Fit per-gene models
    for gene, y_log2 in data_log2.iterrows():
        # detection vector from log2(raw + pseudo) → raw > 0 iff y_log2 > log2(pseudo)
        detect = (y_log2.values > np.log2(pseudo_count)).astype(int)
        X = X_cells.copy()

        # --- logistic part (detection) ---
        # model: detect ~ group + cdr + covariates
        try:
            X_logit = add_constant(X, has_constant="add")
            logit_fit = Logit(detect, X_logit, missing="drop").fit(disp=0, maxiter=100)
            # p for group coefficient
            p_logit = float(logit_fit.pvalues.get("group", np.nan))
            beta_logit = float(logit_fit.params.get("group", np.nan))
        except PerfectSeparationError:
            p_logit, beta_logit = np.nan, np.nan
        except Exception:
            p_logit, beta_logit = np.nan, np.nan

        # --- continuous part (only positive cells) ---
        pos_idx = detect.astype(bool)
        p_cont, beta_cont = np.nan, np.nan
        if pos_idx.sum() >= 5 and (~pos_idx).sum() >= 1:
            y_pos = y_log2.values[pos_idx].astype(float)
            X_pos = X.loc[pos_idx, :]
            X_pos = add_constant(X_pos, has_constant="add")
            try:
                ols = OLS(y_pos, X_pos, missing="drop")
                if robust_se:
                    fit = ols.fit(cov_type="HC3")
                else:
                    fit = ols.fit()
                p_cont  = float(fit.pvalues.get("group", np.nan))
                beta_cont = float(fit.params.get("group", np.nan))
            except Exception:
                p_cont, beta_cont = np.nan, np.nan

        # --- combine two p-values (Stouffer), weighting by sqrt(n) ---
        # if one part missing, fall back to the other
        Zs, ws = [], []
        if np.isfinite(p_logit):
            Zs.append(p_to_z_two_sided(p_logit))
            ws.append(np.sqrt(len(detect)))
        if np.isfinite(p_cont):
            Zs.append(p_to_z_two_sided(p_cont))
            ws.append(np.sqrt(pos_idx.sum()))
        if len(Zs) == 0:
            p_comb = np.nan
        elif len(Zs) == 1:
            p_comb = 2*norm.sf(Zs[0])
        else:
            Z_comb = np.dot(ws, Zs) / (np.sqrt(np.sum(np.square(ws))))
            p_comb = 2*norm.sf(Z_comb)

        results.append({
            "gene": gene,
            "log2.mean.group0": mean0.loc[gene],
            "log2.mean.group1": mean1.loc[gene],
            "log2_fc": log2_fc.loc[gene],
            "n0_detect": int(detect[g0_mask.values].sum()),
            "n1_detect": int(detect[g1_mask.values].sum()),
            "p_logit": p_logit,
            "p_cont": p_cont,
            "p_comb": p_comb,
            "beta_logit": beta_logit,
            "beta_cont": beta_cont,
            "Auc": auc_series.loc[gene],
        })

    out = pd.DataFrame(results).set_index("gene")
    out["q_comb"] = bh_fdr(out["p_comb"])
    # optional: cap ultra-small p like you did in R
    out["p_logit"] = out["p_logit"].clip(lower=1e-300)
    out["p_cont"]  = out["p_cont"].clip(lower=1e-300)
    out["p_comb"]  = out["p_comb"].clip(lower=1e-300)
    return out
