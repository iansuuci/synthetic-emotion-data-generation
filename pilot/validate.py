"""Automatic filters and a blind judge from the idea-model family, which is not the story generator."""

from __future__ import annotations

import pandas as pd

from pilot.config import (
    IDEA_MODEL,
    JUDGE_TEMPERATURE,
    LEXICON,
    MIN_INTENSITY,
    NEUTRAL_MAX_INTENSITY,
    STORY_MAX_WORDS,
    STORY_MIN_WORDS,
)
from pilot.llm import append_jsonl, chat, load_jsonl, map_indexed, parse_json
from pilot.premises import data_dir
from pilot.textutil import leakage_hits, word_count


def _fidelity(premise: str, story: str) -> dict:
    prompt = (
        "Does the story preserve the premise?\n\n"
        f"Premise: {premise}\n\nStory: {story}\n\n"
        "Pass if the same character experiences the same event in the same setting and the outcome is unchanged. "
        "Minor sensory detail, a colleague's name, or leaving the occupation unstated is fine. "
        "Fail if the story contradicts the premise, reverses who did what, or adds a crisis.\n\n"
        'JSON only: {"pass": true, "issue": ""}'
    )
    result = chat(prompt, model=IDEA_MODEL, temperature=0.0, max_tokens=300, reasoning="minimal")
    try:
        parsed = parse_json(result.get("output") or "")
        return {"fidelity_pass": bool(parsed.get("pass")), "fidelity_issue": str(parsed.get("issue") or "")[:240]}
    except Exception:
        return {"fidelity_pass": False, "fidelity_issue": "unparseable fidelity check"}


def _judge(story: str) -> dict:
    prompt = (
        "Label the emotion expressed by the character in this story. You are not told a target.\n\n"
        "Labels:\n"
        "- enraged: intense hostile anger\n"
        "- vulnerable: unprotected, easily hurt, unsafe\n"
        "- annoyed: mild irritation\n"
        "- happy: positive pleasure or warmth\n"
        "- calm: settled, low-key positive ease\n"
        "- neutral: no emotional reaction is expressed or implied\n\n"
        "Score intensity by how clearly the state runs through the scene, not by how extreme the emotion category is.\n"
        "1 = absent or only a hint. 2 = present but easy to miss. 3 = clear through most of the scene. "
        "4 = strong. 5 = overwhelming.\n"
        "A mild emotion can score 3 or 4 when it is unmistakable. A neutral story should score 1.\n"
        "valence is negative, positive, or none.\n"
        "arousal is low, mid, high, or none.\n\n"
        f"Story:\n{story}\n\n"
        'JSON only: {"label": "...", "intensity": 1, "valence": "...", "arousal": "..."}'
    )
    result = chat(prompt, model=IDEA_MODEL, temperature=JUDGE_TEMPERATURE, max_tokens=300, reasoning="minimal")
    empty = {
        "judge_model": IDEA_MODEL,
        "judge_label": None,
        "judge_intensity": None,
        "judge_valence": None,
        "judge_arousal": None,
        "judge_ok": False,
    }
    try:
        parsed = parse_json(result.get("output") or "")
    except Exception:
        return empty
    label = str(parsed.get("label") or "").strip().lower()
    try:
        intensity = int(parsed.get("intensity"))
    except (TypeError, ValueError):
        intensity = None
    return {
        "judge_model": IDEA_MODEL,
        "judge_label": label or None,
        "judge_intensity": intensity,
        "judge_valence": str(parsed.get("valence") or "").strip().lower() or None,
        "judge_arousal": str(parsed.get("arousal") or "").strip().lower() or None,
        "judge_ok": label in LEXICON and intensity is not None,
    }


def _label_agrees(condition: str, label, intensity) -> bool:
    if label != condition or intensity is None:
        return False
    if condition == "neutral":
        return intensity <= NEUTRAL_MAX_INTENSITY
    return intensity >= MIN_INTENSITY


def validate_stories(stories: pd.DataFrame, tag: str) -> pd.DataFrame:
    folder = data_dir(tag)
    checkpoint = folder / "ckpt_validate.jsonl"
    done = load_jsonl(checkpoint)
    pending = []
    for row in stories.itertuples(index=False):
        if row.id not in done:
            pending.append(row)

    def review(row) -> dict:
        text = row.story or ""
        hits = leakage_hits(text)
        count = word_count(text)
        length_ok = STORY_MIN_WORDS <= count <= STORY_MAX_WORDS
        generated_ok = row.finish_reason == "stop" and bool(text.strip())
        fidelity = _fidelity(row.premise, text) if generated_ok else {
            "fidelity_pass": False,
            "fidelity_issue": "no story",
        }
        judged = _judge(text) if generated_ok else {
            "judge_model": IDEA_MODEL,
            "judge_label": None,
            "judge_intensity": None,
            "judge_valence": None,
            "judge_arousal": None,
            "judge_ok": False,
        }
        agrees = _label_agrees(row.condition, judged["judge_label"], judged["judge_intensity"])
        passed = bool(generated_ok and length_ok and not hits and fidelity["fidelity_pass"] and agrees)
        return {
            "key": row.id,
            "id": row.id,
            "word_count": count,
            "length_ok": length_ok,
            "generated_ok": generated_ok,
            "leakage_hit": "|".join(hits),
            "fidelity_pass": fidelity["fidelity_pass"],
            "fidelity_issue": fidelity["fidelity_issue"],
            **judged,
            "judge_agrees": agrees,
            "passed_all": passed,
        }

    print(f"{len(pending)} stories to filter and judge, {len(done)} cached", flush=True)
    if pending:
        for record in map_indexed(pending, review):
            done[record["key"]] = record
            append_jsonl(checkpoint, record)

    reviews = pd.DataFrame([done[row.id] for row in stories.itertuples(index=False) if row.id in done])
    merged = stories.merge(reviews.drop(columns=["key"]), on="id", how="left")
    out = folder / "stories.parquet"
    merged.to_parquet(out, index=False)
    if len(merged):
        print(f"passed_all {int(merged['passed_all'].fillna(False).sum())}/{len(merged)} -> {out}", flush=True)
    return merged
