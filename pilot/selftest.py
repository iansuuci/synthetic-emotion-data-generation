"""Checks that do not call a model: leakage, word count, and the premise-level split."""

from __future__ import annotations

import pandas as pd

from pilot.export import assign_splits
from pilot.textutil import leakage_hits, word_count


def run() -> None:
    enraged = "She was enraged and spoke furiously about the delay."
    assert "enraged" in leakage_hits(enraged)
    assert "furiously" in leakage_hits(enraged)
    assert leakage_hits("They encourage the team to forage for samples.") == []
    assert leakage_hits("A piece of the form was missing.") == []
    assert word_count("One two three.") == 3

    ids = [f"p{i:04d}" for i in range(1, 101)]
    first = assign_splits(ids, seed=20261004)
    second = assign_splits(ids, seed=20261004)
    assert first == second
    assert sum(split == "test" for split in first.values()) == 20
    assert sum(split == "train" for split in first.values()) == 80

    frame = pd.DataFrame(
        [{"premise_id": premise_id, "split": split} for premise_id, split in first.items()]
    )
    mixed = frame.groupby("premise_id")["split"].nunique()
    assert int(mixed.max()) == 1
    print("selftest ok")


if __name__ == "__main__":
    run()
