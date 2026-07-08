"""
eda_utils.py

Reusable EDA utilities for the datasets

Notes:
- Every function takes a DataFrame (or list of paths) in, returns a DataFrame
  or dict out. Nothing prints by default except profile_dataframe, which is
  meant for interactive use, everything else is meant to be logged or saved.
- Functions are side-effect free where possible, no in-place mutation of
  input dataframes.
- Plotting functions return the matplotlib Figure object instead of calling
  plt.show(), so you is is possible tosave them to disk in a pipeline or display them
  inline in a notebook.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------
# 1. Core dataframe profiling
# --------------------------------------------------------------------------

@dataclass
class DataFrameProfile:
    name: str
    shape: tuple
    dtypes: pd.Series
    n_duplicate_rows: int
    memory_mb: float
    head: pd.DataFrame
    describe: pd.DataFrame


def profile_dataframe(df: pd.DataFrame, name: str = "df", verbose: bool = True) -> DataFrameProfile:
    """
    Single entry point EDA summary, replaces manually calling
    df.shape, df.head(), df.info(), df.describe() separately.

    Parameters
    ----------
    df : DataFrame to profile
    name : label used in printed output, useful when profiling multiple
        dataframes in a loop
    verbose : if True, prints a human readable summary to stdout

    Returns
    -------
    DataFrameProfile dataclass, so the summary can also be stored and
    compared programmatically later (e.g. before/after cleaning).
    """
    n_dupes = int(df.duplicated().sum())
    mem_mb = df.memory_usage(deep=True).sum() / 1e6

    profile = DataFrameProfile(
        name=name,
        shape=df.shape,
        dtypes=df.dtypes,
        n_duplicate_rows=n_dupes,
        memory_mb=mem_mb,
        head=df.head(),
        describe=df.describe(include="all").transpose(),
    )

    if verbose:
        print(f"\n{'=' * 60}\nPROFILE: {name}\n{'=' * 60}")
        print(f"shape: {profile.shape}")
        print(f"memory: {profile.memory_mb:.2f} MB")
        print(f"duplicate rows: {profile.n_duplicate_rows}")
        print("\ndtypes:")
        print(profile.dtypes)
        print("\nhead:")
        print(profile.head)

    return profile


def missingness_report(
    df: pd.DataFrame,
    flag_threshold: float = 0.4,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Percent missing per column, sorted descending, with a boolean flag for
    columns above flag_threshold (default 40 percent missing).

    Returns a DataFrame with columns: column, n_missing, pct_missing, flagged
    """
    n_missing = df.isna().sum()
    pct_missing = (n_missing / len(df)).round(4)

    report = pd.DataFrame({
        "column": df.columns,
        "n_missing": n_missing.values,
        "pct_missing": pct_missing.values,
    })
    report["flagged"] = report["pct_missing"] > flag_threshold
    report = report.sort_values("pct_missing", ascending=False).reset_index(drop=True)

    if plot and report["n_missing"].sum() > 0:
        fig, ax = plt.subplots(figsize=(8, max(4, len(report) * 0.25)))
        subset = report[report["n_missing"] > 0]
        sns.barplot(data=subset, y="column", x="pct_missing", ax=ax, color="steelblue")
        ax.axvline(flag_threshold, color="red", linestyle="--", label=f"{flag_threshold:.0%} threshold")
        ax.set_xlabel("fraction missing")
        ax.legend()
        fig.tight_layout()
        report.attrs["figure"] = fig

    return report


def dtype_audit(df: pd.DataFrame, numeric_cast_sample: int = 500) -> pd.DataFrame:
    """
    Flags columns that are stored as object but actually contain numeric or
    boolean-like values, common after pd.read_csv(low_memory=False) on
    messy mixed type columns.

    Returns a DataFrame with columns: column, current_dtype, suspected_type,
    n_unique, sample_values
    """
    rows = []
    for col in df.columns:
        series = df[col]
        current_dtype = str(series.dtype)
        n_unique = series.nunique(dropna=True)
        sample_vals = series.dropna().unique()[:5]

        suspected = current_dtype
        if current_dtype == "object":
            sample = series.dropna().head(numeric_cast_sample)
            numeric_coerced = pd.to_numeric(sample, errors="coerce")
            frac_numeric = numeric_coerced.notna().mean() if len(sample) else 0

            unique_lower = set(str(v).strip().lower() for v in sample_vals)
            bool_like = unique_lower.issubset({"true", "false", "0", "1", "yes", "no", "nan", ""})

            if frac_numeric > 0.95:
                suspected = "numeric (object-cast)"
            elif bool_like and n_unique <= 2:
                suspected = "boolean (object-cast)"
            elif n_unique <= 20:
                suspected = "categorical (low cardinality object)"

        rows.append({
            "column": col,
            "current_dtype": current_dtype,
            "suspected_type": suspected,
            "n_unique": n_unique,
            "sample_values": list(sample_vals),
        })

    return pd.DataFrame(rows)


def cardinality_report(df: pd.DataFrame, near_constant_threshold: float = 0.99) -> pd.DataFrame:
    """
    Unique value count and ratio per column. Flags likely ID columns
    (cardinality ~= n_rows) and near constant columns (one value dominates
    above near_constant_threshold), both of which are candidates to drop
    before modeling.
    """
    n = len(df)
    rows = []
    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / n if n else 0

        top_value_ratio = 0.0
        if series.notna().any():
            top_value_ratio = series.value_counts(normalize=True, dropna=True).iloc[0]

        rows.append({
            "column": col,
            "n_unique": n_unique,
            "unique_ratio": round(unique_ratio, 4),
            "top_value_ratio": round(float(top_value_ratio), 4),
            "likely_id_column": unique_ratio > 0.98,
            "near_constant": top_value_ratio > near_constant_threshold,
        })

    return pd.DataFrame(rows).sort_values("unique_ratio", ascending=False).reset_index(drop=True)


def outlier_summary(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, method: str = "iqr") -> pd.DataFrame:
    """
    IQR based (default) or z-score based outlier flags for numeric columns.

    method : "iqr" uses 1.5 * IQR fences, "zscore" uses |z| > 3

    Returns a DataFrame with columns: column, n_outliers, pct_outliers,
    lower_bound, upper_bound
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    rows = []
    for col in cols:
        series = df[col].dropna()
        if series.empty:
            continue

        if method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = int(((series < lower) | (series > upper)).sum())
        elif method == "zscore":
            z = (series - series.mean()) / series.std(ddof=0)
            lower, upper = series.mean() - 3 * series.std(), series.mean() + 3 * series.std()
            n_outliers = int((z.abs() > 3).sum())
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        rows.append({
            "column": col,
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(series), 4),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
        })

    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True)


def class_balance_report(df: pd.DataFrame, label_col: str, plot: bool = True) -> pd.DataFrame:
    """
    Counts and proportions per class, plus an imbalance ratio
    (majority class count / minority class count). Screening datasets
    are almost always imbalanced, this makes it explicit early.
    """
    counts = df[label_col].value_counts(dropna=False)
    props = df[label_col].value_counts(normalize=True, dropna=False)

    report = pd.DataFrame({
        "count": counts,
        "proportion": props.round(4),
    }).reset_index().rename(columns={"index": label_col})

    if len(counts) >= 2:
        imbalance_ratio = counts.iloc[0] / counts.iloc[-1]
        report.attrs["imbalance_ratio"] = round(float(imbalance_ratio), 2)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=report, x=label_col, y="count", ax=ax, color="steelblue")
        ax.set_title(f"class balance: {label_col}")
        fig.tight_layout()
        report.attrs["figure"] = fig

    return report


def duplicate_check(df: pd.DataFrame, subset_keys: Optional[Sequence[str]] = None) -> dict:
    """
    Reports exact full-row duplicates, plus duplicates on a subset of key
    columns (e.g. patient_id + recording_date), which catches near
    duplicates that a full row check would miss.
    """
    exact_dupes = df[df.duplicated(keep=False)]
    result = {
        "n_exact_duplicates": int(df.duplicated().sum()),
        "exact_duplicate_rows": exact_dupes,
    }

    if subset_keys:
        subset_dupes = df[df.duplicated(subset=subset_keys, keep=False)]
        result["n_subset_duplicates"] = int(df.duplicated(subset=subset_keys).sum())
        result["subset_duplicate_rows"] = subset_dupes.sort_values(list(subset_keys))

    return result


def save_profile_report(df: pd.DataFrame, path: Path, minimal: bool = True) -> Path:
    """
    Wraps ydata-profiling to generate a baseline HTML EDA report.
    minimal=True keeps generation fast on larger datasets, set False for a
    small dataset where you want the full correlation/interaction analysis.

    Requires: pip install ydata-profiling
    """
    from ydata_profiling import ProfileReport

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = ProfileReport(df, title=path.stem, minimal=minimal)
    profile.to_file(path)
    logger.info("Saved profile report to %s", path)
    return path


# --------------------------------------------------------------------------
# 2. Audio specific utilities (COUGHVID, TB_screen, West China all need these)
# --------------------------------------------------------------------------

def audio_file_integrity_check(paths: Iterable[str | Path]) -> pd.DataFrame:
    """
    Confirms every referenced audio file actually exists on disk and is
    non-empty. Run this before any training job, a single broken path can
    crash a batch job hours in.

    Returns a DataFrame with columns: path, exists, size_bytes, readable
    """
    rows = []
    for p in paths:
        p = Path(p)
        exists = p.exists()
        size_bytes = p.stat().st_size if exists else 0
        readable = False
        if exists and size_bytes > 0:
            try:
                with open(p, "rb") as f:
                    f.read(64)
                readable = True
            except Exception:
                readable = False

        rows.append({
            "path": str(p),
            "exists": exists,
            "size_bytes": size_bytes,
            "readable": readable,
        })

    report = pd.DataFrame(rows)
    n_bad = (~report["exists"] | ~report["readable"] | (report["size_bytes"] == 0)).sum()
    logger.info("audio_file_integrity_check: %d / %d files failed integrity check", n_bad, len(report))
    return report


def audio_metadata_extract(
    paths: Iterable[str | Path],
    cache_path: Optional[Path] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Extracts duration, sample rate, channels, and bit depth for each audio
    file using soundfile (fast, header-only read, no full decode).

    Results are cached to cache_path as parquet if provided, since
    re-reading headers for thousands of files on every EDA run is wasteful.

    Returns a DataFrame with columns: path, duration_sec, sample_rate,
    channels, subtype, error
    """
    import soundfile as sf

    if cache_path is not None and Path(cache_path).exists():
        logger.info("Loading cached audio metadata from %s", cache_path)
        return pd.read_parquet(cache_path)

    def _extract_one(p: str | Path) -> dict:
        p = Path(p)
        try:
            info = sf.info(str(p))
            return {
                "path": str(p),
                "duration_sec": info.frames / info.samplerate if info.samplerate else None,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "subtype": info.subtype,
                "error": None,
            }
        except Exception as e:
            return {
                "path": str(p),
                "duration_sec": None,
                "sample_rate": None,
                "channels": None,
                "subtype": None,
                "error": str(e),
            }

    paths = list(paths)
    if n_jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_extract_one, paths))
    else:
        rows = [_extract_one(p) for p in paths]

    result = pd.DataFrame(rows)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path)
        logger.info("Cached audio metadata to %s", cache_path)

    return result


def duration_distribution_report(
    audio_meta_df: pd.DataFrame,
    duration_col: str = "duration_sec",
    short_thresh: float = 0.5,
    long_thresh: float = 30.0,
    plot: bool = True,
) -> dict:
    """
    Summarizes the duration distribution and flags suspiciously short
    (< short_thresh sec) or long (> long_thresh sec) clips, both are common
    signs of a bad recording or a mislabeled/truncated file.

    Returns dict with keys: stats (Series), flagged_short (DataFrame),
    flagged_long (DataFrame), and optionally figure
    """
    durations = audio_meta_df[duration_col].dropna()
    stats = durations.describe()

    flagged_short = audio_meta_df[audio_meta_df[duration_col] < short_thresh]
    flagged_long = audio_meta_df[audio_meta_df[duration_col] > long_thresh]

    result = {
        "stats": stats,
        "flagged_short": flagged_short,
        "flagged_long": flagged_long,
        "n_flagged_short": len(flagged_short),
        "n_flagged_long": len(flagged_long),
    }

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(durations, bins=50, ax=ax)
        ax.axvline(short_thresh, color="orange", linestyle="--", label=f"short thresh ({short_thresh}s)")
        ax.axvline(long_thresh, color="red", linestyle="--", label=f"long thresh ({long_thresh}s)")
        ax.set_xlabel("duration (sec)")
        ax.legend()
        fig.tight_layout()
        result["figure"] = fig

    return result


def silence_and_clipping_check(
    paths: Iterable[str | Path],
    silence_thresh_db: float = -40.0,
    clip_thresh: float = 0.99,
) -> pd.DataFrame:
    """
    For each file, estimates percent silence (below silence_thresh_db) and
    percent clipped samples (abs amplitude above clip_thresh, on a
    normalized -1..1 scale). Both are common quality issues in
    crowdsourced or field recorded cough audio.

    Returns a DataFrame with columns: path, pct_silence, pct_clipped, error
    """
    import librosa

    rows = []
    for p in paths:
        p = Path(p)
        try:
            y, sr = librosa.load(str(p), sr=None, mono=True)
            if len(y) == 0:
                rows.append({"path": str(p), "pct_silence": None, "pct_clipped": None, "error": "empty file"})
                continue

            rms = librosa.feature.rms(y=y)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            pct_silence = float((rms_db < silence_thresh_db).mean())
            pct_clipped = float((np.abs(y) > clip_thresh).mean())

            rows.append({
                "path": str(p),
                "pct_silence": round(pct_silence, 4),
                "pct_clipped": round(pct_clipped, 4),
                "error": None,
            })
        except Exception as e:
            rows.append({"path": str(p), "pct_silence": None, "pct_clipped": None, "error": str(e)})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. COUGHVID_v3 specific
# --------------------------------------------------------------------------

def join_key_validation(
    *dfs_with_names: tuple[pd.DataFrame, str],
    key: str,
) -> pd.DataFrame:
    """
    Confirms a join key exists across multiple dataframes and reports
    orphaned rows, keys present in one table but missing from another.

    Usage:
        join_key_validation((coughvid_df, "metadata"),
                             (coughvid_extr_features_df, "features"),
                             (coughvid_filtered_exp_lbls, "labels"),
                             key="uuid")

    Returns a DataFrame, one row per table pair, with columns:
    left, right, n_left_only, n_right_only, n_matched
    """
    rows = []
    tables = list(dfs_with_names)
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            df_a, name_a = tables[i]
            df_b, name_b = tables[j]

            if key not in df_a.columns or key not in df_b.columns:
                rows.append({
                    "left": name_a, "right": name_b,
                    "n_left_only": None, "n_right_only": None, "n_matched": None,
                    "error": f"key '{key}' missing from one or both tables",
                })
                continue

            keys_a, keys_b = set(df_a[key].dropna()), set(df_b[key].dropna())
            rows.append({
                "left": name_a,
                "right": name_b,
                "n_left_only": len(keys_a - keys_b),
                "n_right_only": len(keys_b - keys_a),
                "n_matched": len(keys_a & keys_b),
                "error": None,
            })

    return pd.DataFrame(rows)


def label_agreement_analysis(labels_df: pd.DataFrame, item_col: str, rater_col: str, label_col: str) -> dict:
    """
    Computes inter-rater agreement for expert labeled clips. Uses Cohen's
    kappa when exactly two raters overlap per item, Fleiss' kappa when
    there are more. Requires the labels table to be in long format
    (one row per item-rater-label triple).

    Returns dict with keys: kappa, kappa_type, n_items_with_multiple_raters,
    consensus_table
    """
    from sklearn.metrics import cohen_kappa_score

    pivot = labels_df.pivot_table(index=item_col, columns=rater_col, values=label_col, aggfunc="first")
    multi_rated = pivot.dropna(thresh=2)

    result = {
        "n_items_with_multiple_raters": len(multi_rated),
        "consensus_table": None,
        "kappa": None,
        "kappa_type": None,
    }

    if len(multi_rated) == 0:
        logger.warning("No items have overlapping ratings from multiple raters.")
        return result

    n_raters = pivot.shape[1]
    if n_raters == 2:
        col_a, col_b = multi_rated.columns[:2]
        paired = multi_rated[[col_a, col_b]].dropna()
        result["kappa"] = round(cohen_kappa_score(paired[col_a], paired[col_b]), 4)
        result["kappa_type"] = "cohen"
    else:
        try:
            from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
            table, _ = aggregate_raters(multi_rated.fillna(-1).to_numpy())
            result["kappa"] = round(float(fleiss_kappa(table)), 4)
            result["kappa_type"] = "fleiss"
        except ImportError:
            logger.warning("statsmodels not available, skipping Fleiss kappa computation.")

    result["consensus_table"] = (
        multi_rated.apply(lambda row: row.value_counts().idxmax() if row.notna().any() else None, axis=1)
        .to_frame("consensus_label")
    )
    return result


def label_vs_selfreport_comparison(
    coughvid_df: pd.DataFrame,
    expert_labels_df: pd.DataFrame,
    key: str,
    self_report_col: str,
    expert_label_col: str,
) -> pd.DataFrame:
    """
    Merges self reported status against expert labels on the shared key and
    returns a confusion-matrix-style crosstab plus the raw disagreement
    rows, so you can inspect where self report and expert opinion diverge.
    """
    merged = coughvid_df[[key, self_report_col]].merge(
        expert_labels_df[[key, expert_label_col]], on=key, how="inner"
    )
    crosstab = pd.crosstab(merged[self_report_col], merged[expert_label_col], margins=True)
    disagreements = merged[merged[self_report_col] != merged[expert_label_col]]

    return {
        "crosstab": crosstab,
        "disagreements": disagreements,
        "pct_disagreement": round(len(disagreements) / len(merged), 4) if len(merged) else None,
    }


def demographic_breakdown(
    df: pd.DataFrame,
    demo_cols: Sequence[str],
    label_col: str,
    plot: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Cross tabulates each demographic column (age, gender, geography, etc.)
    against the label column, to check for demographic skew or bias in
    the label distribution.

    Returns dict keyed by demo column name, each value a normalized
    crosstab DataFrame.
    """
    results = {}
    for col in demo_cols:
        if col not in df.columns:
            logger.warning("Column %s not found, skipping.", col)
            continue

        crosstab = pd.crosstab(df[col], df[label_col], normalize="index").round(4)
        results[col] = crosstab

        if plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            crosstab.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"{label_col} distribution by {col}")
            fig.tight_layout()
            results[f"{col}_figure"] = fig

    return results


def extracted_feature_distributions(
    features_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    plot_corr: bool = True,
) -> dict:
    """
    Distribution stats plus a correlation matrix for precomputed audio
    features. Also flags NaN/inf values, which are common outputs of
    feature extraction pipelines on silent or corrupt clips.

    Returns dict with keys: describe, nan_inf_report, corr_matrix,
    and optionally figure
    """
    if feature_cols is None:
        feature_cols = features_df.select_dtypes(include=np.number).columns.tolist()

    subset = features_df[feature_cols]
    nan_inf_report = pd.DataFrame({
        "column": feature_cols,
        "n_nan": subset.isna().sum().values,
        "n_inf": np.isinf(subset.to_numpy(dtype=float, na_value=0)).sum(axis=0),
    })

    result = {
        "describe": subset.describe().transpose(),
        "nan_inf_report": nan_inf_report,
        "corr_matrix": subset.corr(),
    }

    if plot_corr:
        fig, ax = plt.subplots(figsize=(min(1 + len(feature_cols) * 0.4, 20), min(1 + len(feature_cols) * 0.4, 20)))
        sns.heatmap(result["corr_matrix"], cmap="coolwarm", center=0, ax=ax)
        fig.tight_layout()
        result["figure"] = fig

    return result


def feature_label_correlation(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    key: str,
    label_col: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Quick univariate signal check, point-biserial correlation (for binary
    labels) or ANOVA F-stat (for multiclass) between each feature and the
    label. Useful for a first pass on which features carry signal before
    building a full model.

    Returns a DataFrame with columns: feature, score, p_value, sorted by
    score descending.
    """
    from sklearn.feature_selection import f_classif

    merged = features_df.merge(labels_df[[key, label_col]], on=key, how="inner").dropna(subset=[label_col])

    if feature_cols is None:
        feature_cols = features_df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [c for c in feature_cols if c != key]

    X = merged[feature_cols].fillna(merged[feature_cols].median())
    y = merged[label_col]

    f_scores, p_values = f_classif(X, y)
    result = pd.DataFrame({
        "feature": feature_cols,
        "f_score": f_scores,
        "p_value": p_values,
    }).sort_values("f_score", ascending=False).reset_index(drop=True)

    return result


def audio_quality_flags(
    df: pd.DataFrame,
    snr_col: str = "SNR",
    cough_detect_col: str = "cough_detected",
    snr_thresh: float = 0.0,
    cough_detect_thresh: float = 0.8,
) -> pd.DataFrame:
    """
    COUGHVID ships built-in SNR and cough-detection-confidence columns.
    This flags rows below usable quality thresholds so you can decide on a
    filtering cutoff before training, rather than silently keeping noise.

    Returns the input df with two added boolean columns:
    low_snr_flag, low_cough_confidence_flag
    """
    out = df.copy()
    if snr_col in out.columns:
        out["low_snr_flag"] = out[snr_col] < snr_thresh
    else:
        logger.warning("Column %s not found, skipping SNR flag.", snr_col)

    if cough_detect_col in out.columns:
        out["low_cough_confidence_flag"] = out[cough_detect_col] < cough_detect_thresh
    else:
        logger.warning("Column %s not found, skipping cough confidence flag.", cough_detect_col)

    return out


# --------------------------------------------------------------------------
# 4. TB_screen specific
# --------------------------------------------------------------------------

def forced_vs_passive_comparison(
    forced_df: pd.DataFrame,
    passive_df: pd.DataFrame,
    patient_id_col: str,
    duration_col: Optional[str] = None,
) -> dict:
    """
    Compares the forced and passive cough tables: recordings per patient,
    overlap of patients present in both, and duration distributions if
    available. Forced and passive coughs are physiologically different
    signals, this quantifies how comparable the two collection modes are.

    Returns dict with keys: patients_forced_only, patients_passive_only,
    patients_both, recordings_per_patient (DataFrame), duration_comparison
    """
    patients_forced = set(forced_df[patient_id_col].dropna())
    patients_passive = set(passive_df[patient_id_col].dropna())

    recordings_per_patient = pd.DataFrame({
        "forced_count": forced_df[patient_id_col].value_counts(),
        "passive_count": passive_df[patient_id_col].value_counts(),
    }).fillna(0).astype(int)

    result = {
        "n_patients_forced_only": len(patients_forced - patients_passive),
        "n_patients_passive_only": len(patients_passive - patients_forced),
        "n_patients_both": len(patients_forced & patients_passive),
        "recordings_per_patient": recordings_per_patient,
    }

    if duration_col and duration_col in forced_df.columns and duration_col in passive_df.columns:
        result["duration_comparison"] = pd.DataFrame({
            "forced": forced_df[duration_col].describe(),
            "passive": passive_df[duration_col].describe(),
        })

    return result


def patient_level_aggregation_check(
    *dfs_with_names: tuple[pd.DataFrame, str],
    patient_id_col: str,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Counts recordings per patient across one or more tables. Critical
    before any train/test split: if a patient has multiple recordings and
    they land on both sides of the split, you get patient-level leakage
    and inflated validation metrics.

    Returns a DataFrame with columns: patient_id, <table_name>_count for
    each input table, total_count
    """
    counts = None
    for df, name in dfs_with_names:
        c = df[patient_id_col].value_counts().rename(f"{name}_count")
        counts = c.to_frame() if counts is None else counts.join(c, how="outer")

    counts = counts.fillna(0).astype(int)
    counts["total_count"] = counts.sum(axis=1)
    counts = counts.reset_index().rename(columns={"index": patient_id_col})

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(counts["total_count"], bins=30, ax=ax)
        ax.set_xlabel("recordings per patient")
        ax.set_title("Distribution of recordings per patient (leakage risk check)")
        fig.tight_layout()
        counts.attrs["figure"] = fig

    return counts


def tb_status_distribution(
    metadata_df: pd.DataFrame,
    label_col: str,
    breakdown_cols: Optional[Sequence[str]] = None,
) -> dict:
    """
    Class balance for TB positive/negative, plus breakdown by any
    comorbidity or demographic columns supplied.

    Returns dict with keys: overall (DataFrame from class_balance_report),
    breakdowns (dict of DataFrames keyed by breakdown column)
    """
    overall = class_balance_report(metadata_df, label_col, plot=True)

    breakdowns = {}
    if breakdown_cols:
        for col in breakdown_cols:
            if col not in metadata_df.columns:
                logger.warning("Column %s not found, skipping.", col)
                continue
            breakdowns[col] = pd.crosstab(metadata_df[col], metadata_df[label_col], normalize="index").round(4)

    return {"overall": overall, "breakdowns": breakdowns}


def metadata_completeness_by_class(
    metadata_df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """
    Checks whether missingness in metadata columns correlates with the
    label. Systematic missingness differences between classes (e.g.
    positive cases collected at a different site with a shorter intake
    form) can leak into a model as a spurious shortcut feature.

    Returns a DataFrame indexed by column, with a pct_missing column per
    class value, plus a max_gap column (largest difference across classes).
    """
    rows = []
    for col in metadata_df.columns:
        if col == label_col:
            continue
        pct_missing_by_class = metadata_df.groupby(label_col)[col].apply(lambda s: s.isna().mean())
        row = {"column": col}
        row.update({f"pct_missing_{cls}": round(v, 4) for cls, v in pct_missing_by_class.items()})
        row["max_gap"] = round(float(pct_missing_by_class.max() - pct_missing_by_class.min()), 4)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("max_gap", ascending=False).reset_index(drop=True)


def cross_dataset_schema_diff(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str = "a", name_b: str = "b") -> dict:
    """
    Confirms two tables (e.g. forced vs passive coughs) share a compatible
    schema before concatenation: matching columns, matching dtypes, and a
    rough check of unit consistency for shared numeric columns (via
    comparing value ranges).

    Returns dict with keys: cols_only_in_a, cols_only_in_b,
    dtype_mismatches (DataFrame), range_comparison (DataFrame)
    """
    cols_a, cols_b = set(df_a.columns), set(df_b.columns)
    shared_cols = cols_a & cols_b

    dtype_rows = []
    for col in shared_cols:
        if str(df_a[col].dtype) != str(df_b[col].dtype):
            dtype_rows.append({"column": col, f"dtype_{name_a}": str(df_a[col].dtype), f"dtype_{name_b}": str(df_b[col].dtype)})

    range_rows = []
    for col in shared_cols:
        if pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
            range_rows.append({
                "column": col,
                f"min_{name_a}": df_a[col].min(), f"max_{name_a}": df_a[col].max(),
                f"min_{name_b}": df_b[col].min(), f"max_{name_b}": df_b[col].max(),
            })

    return {
        "cols_only_in_a": sorted(cols_a - cols_b),
        "cols_only_in_b": sorted(cols_b - cols_a),
        "dtype_mismatches": pd.DataFrame(dtype_rows),
        "range_comparison": pd.DataFrame(range_rows),
    }


# --------------------------------------------------------------------------
# 5. West China specific (audio only, label from directory name)
# --------------------------------------------------------------------------

def build_label_dataframe_from_dirs(
    root_path: str | Path,
    audio_extensions: Sequence[str] = (".wav", ".mp3", ".flac", ".ogg"),
) -> pd.DataFrame:
    """
    Walks the directory tree and builds a filepath/label dataframe from
    subfolder names (e.g. root/bronchitis/*.wav, root/pneumonia/*.wav).
    This becomes the synthetic "metadata" table for a dataset with no
    metadata file.

    Returns a DataFrame with columns: filepath, label, filename
    """
    root_path = Path(root_path)
    rows = []
    for label_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        for f in label_dir.rglob("*"):
            if f.suffix.lower() in audio_extensions:
                rows.append({
                    "filepath": str(f),
                    "label": label_dir.name,
                    "filename": f.name,
                })

    df = pd.DataFrame(rows)
    logger.info("Found %d audio files across %d label directories under %s", len(df), df["label"].nunique() if len(df) else 0, root_path)
    return df


def filename_pattern_audit(df: pd.DataFrame, filename_col: str = "filename") -> dict:
    """
    Checks filename naming conventions for consistency, and flags exact
    filename duplicates across the dataset, since inconsistent naming
    often hides duplicate recordings or multiple clips from the same
    patient under different label folders (a leakage risk).

    Returns dict with keys: n_unique_patterns, pattern_examples,
    duplicate_filenames (DataFrame)
    """
    import re

    def _to_pattern(name: str) -> str:
        # collapse digit runs to '#' and lowercase letter runs to 'a', to
        # get a coarse structural pattern per filename, e.g.
        # "patient_0231.wav" -> "patient_#.wav"
        p = re.sub(r"\d+", "#", name)
        p = re.sub(r"[a-zA-Z]+", "a", p)
        return p

    patterns = df[filename_col].apply(_to_pattern)
    pattern_counts = patterns.value_counts()

    dup_mask = df.duplicated(subset=[filename_col], keep=False)
    duplicate_filenames = df[dup_mask].sort_values(filename_col)

    return {
        "n_unique_patterns": patterns.nunique(),
        "pattern_examples": pattern_counts.head(10),
        "duplicate_filenames": duplicate_filenames,
        "n_duplicate_filenames": int(dup_mask.sum()),
    }


def spectrogram_sample_grid(
    df: pd.DataFrame,
    filepath_col: str,
    label_col: str,
    n_per_class: int = 3,
    n_mels: int = 128,
    random_state: int = 42,
) -> plt.Figure:
    """
    Generates a grid of mel spectrograms sampled per class, the fastest
    visual sanity check that classes actually sound different and that
    there's no obvious confound (background noise, device hum, recording
    length) correlated with label.

    Returns the matplotlib Figure, save with fig.savefig(...) as needed.
    """
    import librosa
    import librosa.display

    classes = sorted(df[label_col].unique())
    samples = (
        df.groupby(label_col, group_keys=False)
        .apply(lambda g: g.sample(min(n_per_class, len(g)), random_state=random_state))
    )

    fig, axes = plt.subplots(
        len(classes), n_per_class,
        figsize=(n_per_class * 3.5, len(classes) * 3),
        squeeze=False,
    )

    for row_idx, cls in enumerate(classes):
        cls_samples = samples[samples[label_col] == cls]
        for col_idx in range(n_per_class):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(cls_samples):
                ax.axis("off")
                continue

            filepath = cls_samples.iloc[col_idx][filepath_col]
            try:
                y, sr = librosa.load(filepath, sr=None, mono=True)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                librosa.display.specshow(mel_db, sr=sr, ax=ax, cmap="magma")
            except Exception as e:
                ax.text(0.5, 0.5, f"error:\n{e}", ha="center", va="center", fontsize=8)

            if col_idx == 0:
                ax.set_ylabel(cls, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Mel spectrogram samples by class")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 6. Cross-dataset utilities, run once each dataset is profiled individually
# --------------------------------------------------------------------------

def schema_alignment_matrix(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Maps out which columns exist in which datasets, as a first step toward
    designing a unified schema before merging datasets for one model.

    Parameters
    ----------
    datasets : dict mapping dataset name -> dataframe

    Returns a DataFrame indexed by column name, one boolean column per
    dataset name, True if that dataset has that column.
    """
    all_cols = sorted(set().union(*[set(df.columns) for df in datasets.values()]))
    matrix = pd.DataFrame(
        {name: [col in df.columns for col in all_cols] for name, df in datasets.items()},
        index=all_cols,
    )
    matrix["present_in_n_datasets"] = matrix.sum(axis=1)
    return matrix.sort_values("present_in_n_datasets", ascending=False)


def label_taxonomy_reconciliation(label_maps: dict[str, dict]) -> pd.DataFrame:
    """
    Takes an explicit mapping of each dataset's raw label values to a
    common target taxonomy, and returns a flat lookup table you can join
    against each dataset. This forces the mapping decision to be written
    down and version-controlled rather than made ad hoc during modeling.

    Parameters
    ----------
    label_maps : e.g.
        {
            "coughvid": {"COVID-19": "positive_respiratory", "healthy": "negative"},
            "tb_screen": {"TB_positive": "positive_respiratory", "TB_negative": "negative"},
            "west_china": {"bronchitis": "positive_respiratory", "pneumonia": "positive_respiratory"},
        }

    Returns a DataFrame with columns: dataset, raw_label, unified_label
    """
    rows = []
    for dataset, mapping in label_maps.items():
        for raw_label, unified_label in mapping.items():
            rows.append({"dataset": dataset, "raw_label": raw_label, "unified_label": unified_label})
    return pd.DataFrame(rows)


def dataset_shift_check(
    audio_meta_by_dataset: dict[str, pd.DataFrame],
    duration_col: str = "duration_sec",
    sample_rate_col: str = "sample_rate",
    plot: bool = True,
) -> dict:
    """
    Compares audio duration and sample rate distributions across datasets,
    a fast proxy for domain shift (e.g. COUGHVID crowdsourced phone audio
    vs TB_screen/West China clinical device audio) that matters a lot for
    generalization.

    Parameters
    ----------
    audio_meta_by_dataset : dict mapping dataset name -> audio metadata
        dataframe (output of audio_metadata_extract)

    Returns dict with keys: duration_summary (DataFrame), sample_rate_summary
    (DataFrame), and optionally figure
    """
    duration_summary = pd.DataFrame({
        name: df[duration_col].describe() for name, df in audio_meta_by_dataset.items()
    })
    sample_rate_summary = pd.DataFrame({
        name: df[sample_rate_col].value_counts(normalize=True).round(4) for name, df in audio_meta_by_dataset.items()
    })

    result = {
        "duration_summary": duration_summary,
        "sample_rate_summary": sample_rate_summary,
    }

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, df in audio_meta_by_dataset.items():
            sns.kdeplot(df[duration_col].dropna(), label=name, ax=ax)
        ax.set_xlabel("duration (sec)")
        ax.legend()
        ax.set_title("Duration distribution across datasets")
        fig.tight_layout()
        result["figure"] = fig

    return result