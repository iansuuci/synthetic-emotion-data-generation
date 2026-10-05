"""Stories, one per premise per condition. Request, retry, checkpoint, and resume follow notebook cell 12."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from pilot.config import (
    BANNED_WORDS,
    CONDITION_BY_NAME,
    CONDITION_NAMES,
    DOC_MODEL,
    GLOSS,
    PROMPT_VERSION,
    SEED,
    STORY_TEMPERATURE,
    WORKERS,
)
from pilot.llm import append_jsonl, chat, load_jsonl
from pilot.premises import data_dir

STORY_SYSTEM = """You write plain third-person fiction for an emotion-research dataset.

Output only the story. No title, no word count, no preamble, no closing note, no placeholders, and no mention of these instructions.

Third person, past tense, 180 to 210 words. Do not go under 160 or over 220.
Finish the story; do not stop mid-sentence. End at a natural stopping point.
Do not change the facts in the premise: same person, same occupation, same event, same outcome.
"""


def _user_prompt(premise: str, condition: str) -> str:
    banned = ", ".join(BANNED_WORDS)
    if condition == "neutral":
        steer = (
            "Write a matter-of-fact account of these events. "
            "Do not express or imply an emotional reaction, evaluation, or mood. "
            "No feelings, no body sensations of emotion, no loaded dialogue, and no interior commentary about what the event means to the character. "
            "The reader should not be able to tell whether the character is pleased, bothered, or untouched."
        )
    else:
        steer = (
            f"The character's state is {GLOSS[condition]}. "
            "Convey that state only through thoughts, actions, body sensations, and dialogue. "
            "Do not name the state."
        )
    return (
        f"Premise (these facts are fixed): {premise}\n\n{steer}\n\n"
        f"Do not use any of these words or phrases, including inflected forms: {banned}."
    )


def generate_stories(
    premises: pd.DataFrame,
    tag: str,
    seed: int = SEED,
    conditions: list[str] | None = None,
) -> pd.DataFrame:
    folder = data_dir(tag)
    checkpoint = folder / "ckpt_stories.jsonl"
    done = load_jsonl(checkpoint)
    jobs = []
    for row in premises.itertuples(index=False):
        for condition in (conditions or CONDITION_NAMES):
            key = f"{row.premise_id}_{condition}"
            jobs.append((key, row, condition))
    todo = [job for job in jobs if job[0] not in done]
    print(f"{len(todo)} stories to generate, {len(done)} cached", flush=True)

    def one_doc(job):
        key, row, condition = job
        result = chat(
            _user_prompt(row.premise, condition),
            model=DOC_MODEL,
            temperature=STORY_TEMPERATURE,
            max_tokens=700,
            system=STORY_SYSTEM,
            reasoning="none",
            seed=seed,
        )
        meta = CONDITION_BY_NAME[condition]
        story = (result.get("output") or "").strip()
        return {
            "key": key,
            "id": key,
            "premise_id": row.premise_id,
            "premise": row.premise,
            "character_name": row.character_name,
            "condition": condition,
            "rank": meta["rank"],
            "target_valence": meta["valence"],
            "target_arousal": meta["arousal"],
            "story": story,
            "generator_model": DOC_MODEL,
            "generator_temperature": STORY_TEMPERATURE,
            "prompt_version": PROMPT_VERSION,
            "seed": seed,
            "finish_reason": result.get("finish_reason"),
            "out_tok": result.get("out_tok", 0),
            "reasoning_tok": result.get("reasoning_tok", 0),
            "error": result.get("error"),
        }

    if todo:
        started = time.time()
        tokens = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(one_doc, job) for job in todo]
            for count, future in enumerate(as_completed(futures), 1):
                record = future.result()
                done[record["key"]] = record
                append_jsonl(checkpoint, record)
                tokens += record.get("out_tok") or 0
                if count % 10 == 0 or count == len(todo):
                    elapsed = time.time() - started
                    rate = count / elapsed if elapsed else 0
                    print(
                        f"  {count}/{len(todo)}  {elapsed / 60:5.1f}m  {rate:.2f} doc/s  "
                        f"{tokens / elapsed if elapsed else 0:6.0f} out-tok/s",
                        flush=True,
                    )
        print(f"generated in {(time.time() - started) / 60:.1f} min", flush=True)

    rows = [done[key] for key, _, _ in jobs if key in done]
    frame = pd.DataFrame(rows)
    reasons = frame["finish_reason"].value_counts().to_dict() if len(frame) else {}
    print("finish_reason:", reasons, flush=True)
    if len(frame):
        print("reasoning tokens:", int(frame["reasoning_tok"].fillna(0).sum()), "(want 0)", flush=True)
    out = folder / "stories_raw.parquet"
    frame.to_parquet(out, index=False)
    print(f"wrote {len(frame)} raw stories -> {out}", flush=True)
    return frame
