# Emotion-vector pilot report

Prompt version v2, seed 20261004. Generator: GPT-5.6 Luna (`reasoning` off, temperature 1.0). Judge and premise model: Gemini 3.6 Flash. Local set: `data/full/pilot.parquet` (600 stories). Private copy: [fucntor10/emotion-vector-pilot](https://huggingface.co/datasets/fucntor10/emotion-vector-pilot).

## What was kept

130 emotionally open premises were written, each in six conditions (780 stories). A premise was kept only when all six stories passed. 107 premises qualified. The first 100 are the pilot: 80 train / 20 test, split by premise. Every kept story passed generation, length, leakage, fidelity, and the blind judge.

| Condition | Pass rate (of 130) | Judge agreement | Mean intensity |
|-----------|-------------------:|----------------:|---------------:|
| enraged | 0.946 | 128/130 | 4.02 |
| vulnerable | 0.977 | 130/130 | 3.79 |
| annoyed | 0.977 | 130/130 | 3.60 |
| happy | 0.946 | 130/130 | 3.96 |
| calm | 0.923 | 125/130 | 3.85 |
| neutral | 0.985 | 130/130 | 1.00 |

Confusion on the 780, rows are the target: annoyed, happy, neutral, and vulnerable were perfect. Calm was called happy 5 times. Enraged was called annoyed once and vulnerable once. Of the 32 stories that failed any check, 15 were outside 150–250 words, 4 contained a banned word (`glad` three times, `cheerful` once), 7 failed premise fidelity, and 7 disagreed with the judge. One story failed two checks. None of the leakage hits remain in the 600.

## Length and confounds

| Condition | Mean words | SD | Min | Max |
|-----------|----------:|---:|----:|----:|
| enraged | 232.8 | 7.3 | 216 | 247 |
| vulnerable | 228.0 | 8.1 | 206 | 249 |
| annoyed | 227.9 | 8.7 | 207 | 246 |
| happy | 228.5 | 9.6 | 203 | 247 |
| calm | 226.8 | 8.1 | 205 | 248 |
| neutral | 218.9 | 11.4 | 188 | 243 |

The longest condition mean is 6.3% above the shortest, inside the 10% band. A length-only classifier still scores 0.257 against a 0.167 chance rate, because neutral runs about ten words shorter. A TF-IDF logistic regression scores 0.993. Its top features are stock gestures, not the banned emotion words: annoyed is `clock` / `sighed`, enraged is `heat climbed` / `jaw`, happy is `warm` / `laughed`, calm is `tea` / `mug`, vulnerable is `whispered`, and neutral is `recorded` / `listed`. Within-condition nearest-neighbor cosine (MiniLM) has a median of 0.58–0.63 and a max of 0.77, so stories are not copies, but each emotion has a reusable physical script.

Premise dedup stayed under the 0.85 cosine gate (max pairwise 0.727, median nearest neighbor 0.445).

## Milestone 1 and 2

The first 20 premises were ordinary and open. Two were thin (passing a stack of handouts; watching a teller file a slip), one framed a flight confirmation as already welcome, and one Dutch name was stored with a comma. Stories on those 20 were recognizable and did not use the emotion word, except one `cheerful`.

v1 failed annoyance: the judge labeled 19/20 correctly but scored intensity 2.3, so the pass rate was 0.35 and only 6 of 20 premises survived. v2 states annoyance as obvious through the scene, and the judge scores clarity rather than extremity. A 20-story recheck was 20/20 annoyed, mean intensity 3.7, with no flips into enraged. That prompt was frozen for the 780.

## Not finished

Cohen's kappa is not computed. Two people still need to label the 60 blind stories in `data/full/spotcheck_blind.csv` (10 per condition). The key is `data/full/spotcheck_key.csv`.

NRC VAD (0–1) was confirmed only for happy (valence 0.95, arousal 0.71) and calm (0.74, 0.29). Those match the working labels. Enraged, vulnerable, and annoyed were not checked against the lexicon file. Rank is the specified hostile-to-respectful elicitation order, not a valence sort: vulnerable is rank 2 and annoyed is rank 3.

The same texts will not come back from a fresh temperature-1.0 run. The seed fixes names, axes, and the 80/20 split; the parquet is the reproducible scenario set.

## Changes before a full emotion list

- Take the gesture recipes out of the prompts. The annoyed gloss named sighs and the clock, and those became the TF-IDF features. Describe the state and let the scene choose the body. Do not ban ordinary words like `clock`.
- Give neutral a slightly longer target so the length classifier falls closer to chance.
- Keep the v2 intensity rubric. Without it, mild but clear states fail a cutoff of 3.
- Add a second generator on a subset and test whether the direction transfers. One model's scripts (`heat climbed`, `tea`) will otherwise sit inside the vector.
- Load NRC VAD for the remaining labels, and revisit coverage: this set has no high-arousal positive state besides happy, and no desperation.
