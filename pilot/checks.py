"""Confound checks on a finished set: length, lexical classifier, within-condition diversity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from pilot.embeddings import embed, nearest_neighbor
from pilot.premises import data_dir


def _length_table(frame: pd.DataFrame) -> dict:
    rows = {}
    for condition, group in frame.groupby("condition"):
        counts = group["word_count"].astype(float)
        rows[condition] = {
            "n": int(len(group)),
            "mean": round(float(counts.mean()), 1),
            "std": round(float(counts.std(ddof=0)), 1),
            "min": int(counts.min()),
            "max": int(counts.max()),
        }
    means = [row["mean"] for row in rows.values()]
    span = (max(means) / min(means) - 1) if means and min(means) else None
    return {"by_condition": rows, "max_mean_over_min_mean_minus_1": None if span is None else round(span, 3), "within_10_percent": bool(span is not None and span <= 0.10)}


def _top_features(pipe: make_pipeline, k: int = 8) -> dict:
    vectorizer = pipe.named_steps["tfidfvectorizer"]
    classifier = pipe.named_steps["logisticregression"]
    names = np.array(vectorizer.get_feature_names_out())
    report = {}
    for index, label in enumerate(classifier.classes_):
        coef = classifier.coef_[index]
        order = np.argsort(coef)[-k:][::-1]
        report[str(label)] = [{"feature": names[i], "weight": round(float(coef[i]), 3)} for i in order]
    return report


def run_checks(frame: pd.DataFrame, tag: str) -> dict:
    folder = data_dir(tag)
    y = frame["condition"].astype(str)
    length_scores = cross_val_score(
        LogisticRegression(max_iter=500),
        frame[["word_count"]].astype(float),
        y,
        cv=min(5, int(y.value_counts().min())),
    )
    text_pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english"),
        LogisticRegression(max_iter=2000),
    )
    text_scores = cross_val_score(text_pipe, frame["story"].fillna(""), y, cv=min(5, int(y.value_counts().min())))
    text_pipe.fit(frame["story"].fillna(""), y)

    diversity = {}
    for condition, group in frame.groupby("condition"):
        vectors = embed(group["story"].fillna("").tolist())
        nearest = nearest_neighbor(vectors)
        diversity[condition] = {
            "nn_p50": round(float(np.percentile(nearest, 50)), 3),
            "nn_p90": round(float(np.percentile(nearest, 90)), 3),
            "nn_max": round(float(nearest.max()), 3),
        }

    report = {
        "n": int(len(frame)),
        "length_table": _length_table(frame),
        "length_classifier_accuracy": {
            "mean": round(float(length_scores.mean()), 3),
            "folds": [round(float(score), 3) for score in length_scores],
            "chance": round(1 / frame["condition"].nunique(), 3),
        },
        "tfidf_classifier_accuracy": {
            "mean": round(float(text_scores.mean()), 3),
            "folds": [round(float(score), 3) for score in text_scores],
        },
        "tfidf_top_features": _top_features(text_pipe),
        "within_condition_diversity": diversity,
    }
    path = folder / "checks.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"checks -> {path}", flush=True)
    print(
        f"length acc {report['length_classifier_accuracy']['mean']} "
        f"(chance {report['length_classifier_accuracy']['chance']}); "
        f"tfidf acc {report['tfidf_classifier_accuracy']['mean']}; "
        f"length means within 10%: {report['length_table']['within_10_percent']}",
        flush=True,
    )
    return report
