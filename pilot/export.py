"""Premise-level split, local parquet, and an optional private Hugging Face dataset."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pilot.config import CONDITION_NAMES, SEED, TEST_PREMISES
from pilot.llm import load_env
from pilot.premises import data_dir

EXPORT_COLUMNS = [
    "id", "premise_id", "premise", "character_name",
    "condition", "rank", "target_valence", "target_arousal",
    "story", "word_count",
    "generator_model", "generator_temperature", "prompt_version", "seed",
    "judge_model", "judge_label", "judge_intensity", "judge_valence", "judge_arousal",
    "leakage_hit", "fidelity_pass", "passed_all",
    "split",
]


def complete_premises(frame: pd.DataFrame) -> list[str]:
    """Premises whose six conditions all passed, in id order. The first 100 of these are the pilot set."""
    ok = frame[frame["passed_all"].fillna(False)].copy()
    counts = ok.groupby("premise_id")["condition"].nunique()
    ready = counts[counts >= len(CONDITION_NAMES)].index.tolist()
    return sorted(ready)


def assign_splits(premise_ids: list[str], seed: int = SEED, n_test: int | None = None) -> dict[str, str]:
    ids = sorted(set(premise_ids))
    if n_test is None:
        n_test = TEST_PREMISES if len(ids) >= 100 else max(1, round(len(ids) * 0.2))
    n_test = min(n_test, max(len(ids) - 1, 0))
    rng = np.random.default_rng(seed)
    order = np.array(ids)
    rng.shuffle(order)
    test = set(order[:n_test].tolist())
    return {premise_id: ("test" if premise_id in test else "train") for premise_id in ids}


def select_final(frame: pd.DataFrame, keep: int, seed: int = SEED) -> pd.DataFrame:
    ready = complete_premises(frame)
    chosen = ready[:keep]
    kept = frame[frame["premise_id"].isin(chosen) & frame["passed_all"].fillna(False)].copy()
    # One story per condition. If retries ever duplicated a key, keep the first.
    kept = kept.drop_duplicates(["premise_id", "condition"])
    splits = assign_splits(chosen, seed=seed)
    kept["split"] = kept["premise_id"].map(splits)
    return kept.sort_values(["premise_id", "condition"]).reset_index(drop=True)


def write_spotcheck(frame: pd.DataFrame, tag: str, per_condition: int = 10, seed: int = SEED) -> None:
    folder = data_dir(tag)
    rng = np.random.default_rng(seed)
    picked = []
    for condition in CONDITION_NAMES:
        group = frame[frame["condition"] == condition]
        take = min(per_condition, len(group))
        if take == 0:
            continue
        choice = rng.choice(group.index.to_numpy(), size=take, replace=False)
        picked.extend(choice.tolist())
    sample = frame.loc[picked, ["id", "story", "condition", "premise_id"]].copy()
    sample.insert(0, "item_id", [f"s{i:03d}" for i in range(1, len(sample) + 1)])
    blind = sample[["item_id", "story"]].copy()
    blind["rater_1_label"] = ""
    blind["rater_2_label"] = ""
    blind.to_csv(folder / "spotcheck_blind.csv", index=False)
    sample[["item_id", "id", "condition", "premise_id"]].to_csv(folder / "spotcheck_key.csv", index=False)
    print(f"spot check {len(sample)} stories -> {folder / 'spotcheck_blind.csv'}", flush=True)


def export_local(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    folder = data_dir(tag)
    columns = [column for column in EXPORT_COLUMNS if column in frame.columns]
    out = frame[columns].copy()
    path = folder / "pilot.parquet"
    out.to_parquet(path, index=False)
    print(f"exported {len(out)} rows -> {path}", flush=True)
    return out


def export_hub(frame: pd.DataFrame, repo_name: str = "emotion-vector-pilot", path: str | None = None) -> str | None:
    load_env()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is not set; skipped the Hub export", flush=True)
        return None
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if path is None:
        path = str(data_dir("full") / "pilot.parquet")
        frame.to_parquet(path, index=False)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="pilot.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add emotion-vector pilot split",
    )
    print(f"private dataset: https://huggingface.co/datasets/{repo_id}", flush=True)
    return repo_id
