# Plan B pilot: synthetic dataset for emotion vector extraction

Status: draft plan, no code yet.

## 1. Goal

Build a small, fully labeled, validated dataset that can be used to extract emotion vectors from open-source models, and that proves the generation and validation pipeline works before scaling to the full emotion list.

The pilot is successful if:

- every sample has a target emotion label that an independent judge agrees with,
- emotion is the only thing that varies systematically between conditions (not topic, length, character, or wording),
- the pipeline can be rerun from a seed and produces the same scenario set and splits,
- every design choice below has a written rationale (section 9).

## 2. Conditions

Five emotions, taken from Anthropic's emotion concepts work, ordered from hostile to respectful as specified, plus one control:

| Rank | Condition  | Valence  | Arousal  |
|------|------------|----------|----------|
| 1    | enraged    | negative | high     |
| 2    | vulnerable | negative | low-mid  |
| 3    | annoyed    | negative | mid      |
| 4    | happy      | positive | mid-high |
| 5    | calm       | positive | low      |
| -    | neutral    | none     | none     |

The valence and arousal values are working labels and should be checked against a published norm set (for example NRC VAD or Warriner et al. 2013) before the full run.

The control is a neutral, emotion-free version of the same scenarios, matched on topic, character and length. Emotion vectors are computed as the difference between each emotion's mean activation and the neutral mean (or the grand mean, with neutral used to remove non-emotional directions).

## 3. Design: fully crossed scenarios

The core decision is that every scenario is written once in every condition.

- 100 neutral scenario premises x 6 conditions = 600 stories in the final pilot.
- Each premise fixes a setting, a named character, and an event, but does not imply any emotion. The emotion comes from how the character experiences and reacts to the event.
- Because each premise appears in all six conditions, topic, setting, and character are perfectly balanced across conditions and cannot be what the vector learns.

To survive filtering, generate about 130 premises and keep the first 100 for which all six stories pass validation (scenarios are dropped as whole units, so balance is preserved).

Example premise: "Priya Desai, a pharmacist, is told at the end of her shift that the store's inventory system will be replaced next month." This can plausibly be read as enraging, annoying, unsettling, pleasing, or unremarkable.

## 4. Pipeline

### Step 1: Scenario premises (reuses the idea loop)

- Reuse the idea loop from cell 8 of `LatestDataGen.ipynb`: batched generation, MiniLM embedding dedup, temperature escalation and avoid-lists on stalls, Drive/local parquet checkpoints.
- Replace `DOC_TYPES` / `concept_axes` / `ANGLES` with premise diversity axes:
  - life domain: work, family, health, school, travel, housing, public services, hobbies, technology, money,
  - social context: alone, stranger, friend, family member, colleague, authority figure,
  - event type: news received, request made, plan changed, task completed, object lost or found, decision required.
- Prompt requires premises to be emotionally open: an ordinary event that a reader could plausibly react to with any of the six conditions. Reject premises with built-in emotional valence (deaths, disasters, celebrations).
- Character names come from Faker (reused), with locale and gender varied, and are assigned once per premise so every condition uses the same character.
- Dedup threshold: start at cosine 0.85 for premises (stricter than the notebook's 0.90, because premises are short and near-duplicates are more damaging when each is reused six times).
- Add an LLM pass that rates each premise for emotional openness (can this plausibly go in all six directions? yes/no) and drops failures.

### Step 2: Stories (reuses the document generator)

- Reuse `one_doc` from cell 12: OpenRouter client, per-item JSONL checkpoint, resume from disk, retry and backoff, thread pool, finish_reason and token logging.
- Generator: GPT-5.6 Luna with `reasoning_effort="none"`, temperature 1.0, as in the notebook.
- Story spec (one prompt template with a condition slot):
  - third person, past tense, 150 to 250 words, plain prose, no title,
  - the character's state is conveyed through thoughts, actions, body sensations, and dialogue,
  - the target emotion word and a list of its close synonyms must not appear (prevents the vector from learning a word instead of a state),
  - the premise's facts must not be changed, and the story must end at a natural stopping point,
  - for neutral: a matter-of-fact account of the same events, with no emotional reaction expressed or implied.
- Reused prompt hygiene from the notebook: no placeholders, no meta-commentary, no preamble.
- Dropped from the notebook: claim pools, authority/scope/temporal/reasoning slots, institutions, fake dates, the hyperstition framing, and the long anti-cliché opening rules. These all inject systematic content unrelated to emotion.

### Step 3: Automatic filters

Applied to every story, in order:

1. Generation errors: drop anything with `finish_reason != "stop"` or empty output.
2. Length: keep 150 to 250 words. Report mean and spread per condition; condition means must be within 10% of each other.
3. Lexical leakage: regex for each emotion word and its synonym list (including inflections). Any hit fails the story.
4. Premise fidelity: an LLM check that the story is consistent with the premise.

### Step 4: Label validation

- Blind LLM judge from a different model family than the generator (for example Gemini 3.6 Flash), given only the story, never the target label.
- Judge outputs: one of the six labels, intensity 1 to 5, and valence and arousal ratings.
- Keep a story only if the judge label equals the target and intensity is at least 3 (for neutral: label is neutral and intensity is at most 1).
- Report the confusion matrix and per-condition pass rate. Low pass rates or systematic confusions (for example vulnerable judged as calm) are pilot findings that feed back into the prompt.
- Human spot check: two raters each label 10 random stories per condition (60 total) blind. Report Cohen's kappa between raters and agreement with the target labels.

### Step 5: Confound checks on the final 600

- Length-only classifier predicting condition should be near chance.
- A TF-IDF logistic regression predicting condition will score high; inspect its top features for leaked emotion words or stock phrases, and extend the leakage list if needed.
- Within-condition diversity: nearest-neighbor cosine between stories of the same condition (reuses the audit in cell 14), to catch the generator collapsing to a template per emotion.

### Step 6: Splits and export

- Split by premise, not by story: 80 premises (480 stories) train, 20 premises (120 stories) test. Test therefore measures whether vectors generalize to unseen situations.
- Seeded and recorded, so the split is reproducible.
- Export to a local parquet and a private Hugging Face dataset.

## 5. Output schema

One row per story:

- `id`, `premise_id`, `premise`, `character_name`
- `condition`, `rank` (1 to 5, null for neutral), `target_valence`, `target_arousal`
- `story`, `word_count`
- `generator_model`, `generator_temperature`, `prompt_version`, `seed`
- `judge_model`, `judge_label`, `judge_intensity`, `judge_valence`, `judge_arousal`
- `leakage_hit`, `fidelity_pass`, `passed_all`
- `split` (`train` or `test`)

## 6. Reuse map from `LatestDataGen.ipynb`

Reused, adapted:

- idea loop with embedding dedup and escalation (cell 8) for premises,
- `gen_sync` and `one_doc` request, retry, checkpoint and resume logic (cells 6 and 12),
- `embed` with `all-MiniLM-L6-v2` (cell 6),
- Faker names (cell 5), now per premise,
- pool quality checks (`max_pairwise_cos`, `min_cos_to_seed`, cell 10), applied to premises,
- nearest-neighbor cosine audit (cell 14), applied per condition.

Not reused:

- hard-coded API keys (replace with environment variables or Colab secrets),
- claim pools, institutions, authority/scope/temporal/reasoning slots, fake dates,
- the 700 to 1,000 word document format and the opening-style audits,
- `run_batch` and other dead code, the Colab-only Drive dependency (make it optional).

## 7. Cost and time

- Generation: 130 premises x 6 conditions = 780 stories, about 300 to 400 output tokens each, under 0.5M output tokens.
- Judging: 780 short judge calls.
- Both are small; the pilot should run in well under an hour at the notebook's observed throughput (about 1 doc/s with 24 workers).

## 8. Milestones

1. Write the premise prompt and generate 20 premises; read them by hand.
2. Generate all six conditions for those 20 premises (120 stories); read them by hand and run the judge. Revise prompts.
3. Freeze `prompt_version`, run the full 130 premises, apply filters and judge.
4. Human spot check, confound checks, splits, export.
5. Write a one-page pilot report: pass rates, confusion matrix, kappa, length table, leakage findings, and recommended changes for the full run.

## 9. Design decisions and rationale

- Third-person stories: mirrors the stimulus style used in Anthropic's emotion concepts work, which makes results comparable and keeps the emotion attributed to a character rather than to the model. Whether these vectors also fire on the model's own states under treatment is a downstream test, not a dataset assumption.
- Fully crossed premises: the strongest available control for topic and character confounds; cheaper than balancing statistically.
- Neutral control on the same premises: isolates emotion from situation; a generic neutral corpus would differ in topic and style as well.
- No emotion words: otherwise the extracted direction can be a lexical feature for the word "enraged" rather than the state.
- Different judge family: reduces the chance that generator and judge share the same blind spots.
- Split by premise: prevents the test set from sharing situations with the train set.
- 100 per condition: enough to estimate mean differences and hold out 20 per condition; not enough for strong claims, which is acceptable for a pilot.

## 10. Open questions

- Hostile to respectful ordering: this ranks emotions by the treatment that would plausibly elicit them, not by valence or arousal (vulnerable sits above annoyed). The basis for the ordering should be written down, since it will be used to interpret results.
- Coverage: the five emotions leave gaps in the valence-arousal space (no positive high-arousal state other than happy, no explicitly fearful or desperate state). Fine for a pilot; worth revisiting for the full list, since desperation is the state most linked to misalignment in the Anthropic results.
- Single generator: style artifacts of one model could be absorbed into the vectors. A later check is to generate a subset with a second model and test whether vectors transfer.
- Exact stimulus parameters in Anthropic's paper (story length, number of stories per emotion, denoising procedure) should be confirmed against the paper and cited.
