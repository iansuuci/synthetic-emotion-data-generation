"""Scenario premises. Idea-loop structure follows notebook cell 8, with emotion-specific axes."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from pilot.config import (
    EVENT_TYPES,
    IDEA_MODEL,
    JOB_DENY_SUBSTRINGS,
    LIFE_DOMAINS,
    LOCALES,
    PREMISE_DEDUP_COSINE,
    PREMISE_TEMP_MAX,
    PREMISE_TEMP_START,
    PROMPT_VERSION,
    SEED,
    SOCIAL_CONTEXTS,
)
from pilot.embeddings import embed, max_pairwise, nearest_neighbor
from pilot.llm import append_jsonl, chat, gen_sync, load_jsonl, map_indexed, parse_json
from pilot.textutil import leakage_hits

PREMISE_PROMPT = """Write scenario premises for an emotion-research dataset.

Each premise is ONE sentence. It names a character, an occupation, a setting, and an ordinary event. It does not say how anyone feels and it does not predict a reaction.

The event must stay emotionally open. Without changing the facts, a reader could plausibly experience it as hostile anger, as feeling unprotected, as mild impatience, as warmth, as quiet ease, or as no emotion at all.

Do not use built-in valence. No deaths, injuries, illness, disasters, crimes, diagnoses, breakups, firings, demotions, promotions, prizes, weddings, births, insults, accidents, or emergencies.

Use each slot exactly once, in order. The sentence must contain the character's full name and occupation. Match that slot's life domain, social context, and event type.

Do not frame the event as welcome, lucky, disappointing, or unwelcome. A confirmation, a success, or a setback already carries a feeling.

Do not use the words enraged, vulnerable, annoyed, happy, calm, neutral, angry, sad, afraid, or excited.

Return ONLY a JSON array of strings, one premise per slot, in slot order. No markdown.
"""


def data_dir(tag: str) -> Path:
    """Checkpoint folder. PILOT_DATA_DIR overrides the project data/ directory.

    Colab sets the override when the code itself is not already on Drive, so a
    runtime restart can resume the same premises and stories.
    """
    override = os.environ.get("PILOT_DATA_DIR", "").strip()
    root = Path(override) if override else Path(__file__).resolve().parents[1] / "data"
    path = root / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def _character(rng: random.Random) -> dict:
    from faker import Faker

    for _ in range(12):
        locale = rng.choice(LOCALES)
        gender = rng.choice(["male", "female"])
        person = Faker(locale)
        jobs = Faker("en_US")
        Faker.seed(rng.randint(0, 10**9))
        try:
            first = person.first_name_male() if gender == "male" else person.first_name_female()
            last = person.last_name()
            if "," in last:
                main, particle = [part.strip() for part in last.split(",", 1)]
                last = f"{particle} {main}".strip()
        except Exception:
            person = Faker("en_US")
            first, last = person.first_name(), person.last_name()
        Faker.seed(rng.randint(0, 10**9))
        job = jobs.job()
        if any(bad in job.lower() for bad in JOB_DENY_SUBSTRINGS):
            continue
        return {
            "character_name": f"{first} {last}",
            "character_job": job,
            "locale": locale,
            "gender": gender,
        }
    return {
        "character_name": "Alex Morgan",
        "character_job": "librarian",
        "locale": "en_US",
        "gender": "female",
    }


def _axes(rng: random.Random, n: int) -> list[dict]:
    cells = [
        {"life_domain": domain, "social_context": social, "event_type": event}
        for domain in LIFE_DOMAINS
        for social in SOCIAL_CONTEXTS
        for event in EVENT_TYPES
    ]
    rng.shuffle(cells)
    drawn = []
    while len(drawn) < n:
        batch = cells[:]
        rng.shuffle(batch)
        drawn.extend(batch)
    return drawn[:n]


def _slot_block(slots: list[dict]) -> str:
    lines = []
    for index, slot in enumerate(slots, 1):
        lines.append(
            f"{index}. Character: {slot['character_name']}, {slot['character_job']}. "
            f"Life domain: {slot['life_domain']}. "
            f"Social context: {slot['social_context']}. "
            f"Event type: {slot['event_type']}."
        )
    return "\n".join(lines)


def _openness(premise: str) -> dict:
    prompt = (
        "Could this premise plausibly be lived in each of these ways without changing the facts: "
        "hostile anger, feeling unprotected, mild impatience, warmth, quiet ease, or no emotional reaction?\n\n"
        "Fail it if the event already forces a feeling: loss, harm, reward, insult, emergency, or celebration.\n\n"
        f"Premise: {premise}\n\n"
        'JSON only: {"open": true, "reason": ""}'
    )
    result = chat(prompt, model=IDEA_MODEL, temperature=0.0, max_tokens=300, reasoning="minimal")
    text = result.get("output") or ""
    try:
        parsed = parse_json(text)
        return {"open": bool(parsed.get("open")), "reason": str(parsed.get("reason") or "")[:240]}
    except Exception:
        return {"open": False, "reason": "unparseable openness rating"}


def generate_premises(n: int, tag: str, seed: int = SEED) -> pd.DataFrame:
    """Generate n emotionally open, deduplicated premises. Resumes from the tag checkpoint."""
    folder = data_dir(tag)
    checkpoint = folder / "ckpt_premises.jsonl"
    rejected_path = folder / "premises_rejected.jsonl"
    existing = load_jsonl(checkpoint)
    kept = [record for record in existing.values() if record.get("open")]
    kept.sort(key=lambda record: record["premise_id"])
    print(f"resumed {len(kept)} open premises from {checkpoint}", flush=True)

    rng = random.Random(seed)
    # Burn the RNG the same way on resume so new slots do not reshuffle old ones.
    _axes(rng, max(len(kept), 1))
    for _ in kept:
        _character(rng)

    vectors = None
    if kept:
        vectors = embed([record["premise"] for record in kept])

    stall = 0
    threshold = PREMISE_DEDUP_COSINE
    rounds = 0
    batch_size = 6
    while len(kept) < n and rounds < 40:
        rounds += 1
        temperature = min(PREMISE_TEMP_START + 0.1 * stall, PREMISE_TEMP_MAX)
        n_calls = 4
        prompts = []
        slot_groups = []
        avoid = ""
        if stall and kept:
            sample = rng.sample(kept, min(12 + 6 * stall, len(kept)))
            avoid = (
                "\nDo NOT repeat or resemble these existing premises; change the situation:\n- "
                + "\n- ".join(item["premise"] for item in sample)
                + "\n"
            )
        for _ in range(n_calls):
            slots = []
            for axes in _axes(rng, batch_size):
                slots.append({**_character(rng), **axes})
            slot_groups.append(slots)
            prompts.append(PREMISE_PROMPT + avoid + "\nSlots:\n" + _slot_block(slots))
        texts = gen_sync(prompts, temperature, max_tokens=1200)
        candidates = []
        for slots, text in zip(slot_groups, texts):
            if not text:
                continue
            try:
                parsed = parse_json(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                parsed = next((value for value in parsed.values() if isinstance(value, list)), [])
            if not isinstance(parsed, list):
                continue
            for slot, premise in zip(slots, parsed):
                if not isinstance(premise, str) or len(premise.split()) < 12:
                    continue
                premise = premise.strip()
                if leakage_hits(premise):
                    append_jsonl(rejected_path, {"key": f"lex-{rounds}-{len(candidates)}", "reason": "premise leakage", "premise": premise})
                    continue
                candidates.append({**slot, "premise": premise})
        if not candidates:
            stall += 1
            print(f"round {rounds}: no premises parsed (stall {stall})", flush=True)
            continue
        candidate_vectors = embed([item["premise"] for item in candidates])
        novel = []
        novel_vectors = []
        for item, vector in zip(candidates, candidate_vectors):
            if vectors is not None and len(vectors) and float(np.max(vectors @ vector)) >= threshold:
                append_jsonl(rejected_path, {"key": f"dup-{rounds}-{len(novel)}", "reason": "duplicate", "premise": item["premise"]})
                continue
            if vectors is None:
                vectors = np.zeros((0, vector.shape[0]), dtype=np.float32)
            vectors = np.vstack([vectors, vector])
            novel.append(item)
            novel_vectors.append(vector)
        print(f"round {rounds}: {len(novel)} novel candidates, checking openness", flush=True)

        def rate(item: dict) -> dict:
            verdict = _openness(item["premise"])
            return {**item, **verdict}

        rated = map_indexed(novel, rate) if novel else []
        added = 0
        for item in rated:
            record = {
                "key": f"pending-{len(kept)}",
                "premise": item["premise"],
                "character_name": item["character_name"],
                "character_job": item["character_job"],
                "locale": item["locale"],
                "gender": item["gender"],
                "life_domain": item["life_domain"],
                "social_context": item["social_context"],
                "event_type": item["event_type"],
                "open": item["open"],
                "open_reason": item["reason"],
                "seed": seed,
                "prompt_version": PROMPT_VERSION,
                "generator_model": IDEA_MODEL,
            }
            if item["open"] and len(kept) < n:
                record["premise_id"] = f"p{len(kept) + 1:04d}"
                record["key"] = record["premise_id"]
                append_jsonl(checkpoint, record)
                kept.append(record)
                added += 1
            else:
                record["key"] = f"reject-{rounds}-{added}-{item['character_name']}"
                append_jsonl(rejected_path, record)
        print(
            f"round {rounds}: +{added} -> {len(kept)}/{n}  "
            f"(stall {stall}, thr {threshold:.2f}, temp {temperature:.1f})",
            flush=True,
        )
        if added == 0:
            stall += 1
            threshold = min(threshold + 0.015, 0.92)
            if stall >= 6:
                print(f"stopping early at {len(kept)} premises", flush=True)
                break
        else:
            stall = max(0, stall - 1)

    frame = pd.DataFrame(kept)
    out = folder / "premises.parquet"
    frame.to_parquet(out, index=False)
    if len(frame):
        vectors = embed(frame["premise"].tolist())
        nearest = nearest_neighbor(vectors)
        print(
            f"premises n={len(frame)} max_pairwise_cos={max_pairwise(vectors):.3f} "
            f"nn_p50={float(np.percentile(nearest, 50)):.3f} -> {out}",
            flush=True,
        )
    return frame
