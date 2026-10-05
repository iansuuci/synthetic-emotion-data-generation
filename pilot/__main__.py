"""Step runner for the emotion-vector pilot.

  python -m pilot selftest
  python -m pilot premises --n 20 --tag m1
  python -m pilot stories --premises-tag m1 --tag m2
  python -m pilot validate --stories-tag m2 --tag m2
  python -m pilot summary --tag m2
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from pilot.config import FULL_KEEP, FULL_PREMISE_TARGET, PROMPT_VERSION, RANK_RATIONALE, SEED
from pilot.premises import data_dir


def _read_premises(tag: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir(tag) / "premises.parquet")


def _read_stories(tag: str) -> pd.DataFrame:
    folder = data_dir(tag)
    path = folder / "stories.parquet"
    if not path.exists():
        path = folder / "stories_raw.parquet"
    return pd.read_parquet(path)


def cmd_premises(args: argparse.Namespace) -> None:
    from pilot.premises import generate_premises

    generate_premises(args.n, args.tag, seed=args.seed)


def cmd_stories(args: argparse.Namespace) -> None:
    from pilot.config import CONDITION_NAMES
    from pilot.stories import generate_stories

    conditions = [part.strip() for part in args.conditions.split(",") if part.strip()] or None
    if conditions:
        unknown = [name for name in conditions if name not in CONDITION_NAMES]
        if unknown:
            raise SystemExit(f"unknown conditions: {unknown}")
    generate_stories(_read_premises(args.premises_tag), args.tag, seed=args.seed, conditions=conditions)


def cmd_validate(args: argparse.Namespace) -> None:
    from pilot.validate import validate_stories

    stories = pd.read_parquet(data_dir(args.stories_tag) / "stories_raw.parquet")
    validate_stories(stories, args.tag)


def cmd_summary(args: argparse.Namespace) -> None:
    folder = data_dir(args.tag)
    premises_path = folder / "premises.parquet"
    if premises_path.exists() and args.what in ("premises", "all"):
        frame = pd.read_parquet(premises_path)
        print(f"\n{len(frame)} premises\n")
        for row in frame.itertuples(index=False):
            print(f"{row.premise_id} [{row.life_domain} | {row.social_context} | {row.event_type}] {row.character_name}")
            print(f"  {row.premise}")
            print()
    stories_path = folder / "stories.parquet"
    if stories_path.exists() and args.what in ("stories", "all"):
        frame = pd.read_parquet(stories_path)
        print(f"stories {len(frame)}  passed {int(frame['passed_all'].fillna(False).sum())}")
        if "judge_label" in frame.columns:
            matrix = pd.crosstab(frame["condition"], frame["judge_label"].fillna("(none)"), dropna=False)
            print("\nconfusion (rows = target, columns = judge)")
            print(matrix.to_string())
            print("\npass rate")
            print(frame.groupby("condition")["passed_all"].mean().round(3).to_string())
            print("\nlength")
            print(frame.groupby("condition")["word_count"].agg(["mean", "min", "max"]).round(1).to_string())


def cmd_checks(args: argparse.Namespace) -> None:
    from pilot.checks import run_checks

    run_checks(_read_stories(args.tag), args.tag)


def cmd_finalize(args: argparse.Namespace) -> None:
    from pilot.checks import run_checks
    from pilot.export import export_hub, export_local, select_final, write_spotcheck

    frame = _read_stories(args.tag)
    final = select_final(frame, keep=args.keep, seed=args.seed)
    print(f"complete premises kept: {final['premise_id'].nunique()} ({len(final)} stories)", flush=True)
    exported = export_local(final, args.tag)
    write_spotcheck(final, args.tag, seed=args.seed)
    run_checks(final, args.tag)
    if args.hub:
        export_hub(exported, path=str(data_dir(args.tag) / "pilot.parquet"))
    manifest = {
        "seed": args.seed,
        "prompt_version": PROMPT_VERSION,
        "rank_rationale": RANK_RATIONALE,
        "n_stories": int(len(exported)),
        "n_premises": int(exported["premise_id"].nunique()) if len(exported) else 0,
        "n_test_premises": int((exported.drop_duplicates("premise_id")["split"] == "test").sum()) if len(exported) else 0,
    }
    (data_dir(args.tag) / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Emotion-vector pilot")
    parser.add_argument("--seed", type=int, default=SEED)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("selftest")

    premises = sub.add_parser("premises")
    premises.add_argument("--n", type=int, required=True)
    premises.add_argument("--tag", required=True)
    premises.set_defaults(func=cmd_premises)

    stories = sub.add_parser("stories")
    stories.add_argument("--premises-tag", required=True)
    stories.add_argument("--tag", required=True)
    stories.add_argument("--conditions", default="", help="Comma-separated subset of conditions")
    stories.set_defaults(func=cmd_stories)

    validate = sub.add_parser("validate")
    validate.add_argument("--stories-tag", required=True)
    validate.add_argument("--tag", required=True)
    validate.set_defaults(func=cmd_validate)

    summary = sub.add_parser("summary")
    summary.add_argument("--tag", required=True)
    summary.add_argument("--what", choices=("premises", "stories", "all"), default="all")
    summary.set_defaults(func=cmd_summary)

    checks = sub.add_parser("checks")
    checks.add_argument("--tag", required=True)
    checks.set_defaults(func=cmd_checks)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--tag", required=True)
    finalize.add_argument("--keep", type=int, default=FULL_KEEP)
    finalize.add_argument("--hub", action="store_true")
    finalize.set_defaults(func=cmd_finalize)

    full = sub.add_parser("full")
    full.add_argument("--premises", type=int, default=FULL_PREMISE_TARGET)
    full.add_argument("--keep", type=int, default=FULL_KEEP)
    full.add_argument("--tag", default="full")
    full.set_defaults(func=None)

    args = parser.parse_args()
    if args.command == "selftest":
        from pilot.selftest import run

        run()
        return
    if args.command == "full":
        from pilot.export import complete_premises
        from pilot.premises import generate_premises

        tag = args.tag
        pool = args.premises
        generate_premises(pool, tag, seed=args.seed)
        for round_index in range(4):
            cmd_stories(argparse.Namespace(premises_tag=tag, tag=tag, seed=args.seed, conditions=""))
            cmd_validate(argparse.Namespace(stories_tag=tag, tag=tag))
            ready = complete_premises(_read_stories(tag))
            print(f"complete premises: {len(ready)} (need {args.keep})", flush=True)
            if len(ready) >= args.keep or round_index == 3:
                break
            raw_n = _read_stories(tag)["premise_id"].nunique()
            yield_rate = len(ready) / raw_n if raw_n else 0.5
            pool += int((args.keep - len(ready)) / max(yield_rate, 0.4)) + 5
            print(f"topping up the premise pool to {pool}", flush=True)
            generate_premises(pool, tag, seed=args.seed)
        cmd_finalize(argparse.Namespace(tag=tag, keep=args.keep, seed=args.seed, hub=True))
        return
    args.func(args)


if __name__ == "__main__":
    main()
