"""Pilot constants. Seed, models, and prompt version are recorded on every row."""

from __future__ import annotations

SEED = 20261004
PROMPT_VERSION = "v2"

# Notebook cell 6 routes ideas through Gemini and documents through Luna.
IDEA_MODEL = "google/gemini-3.6-flash"
DOC_MODEL = "openai/gpt-5.6-luna"
JUDGE_MODEL = "google/gemini-3.6-flash"

PREMISE_TEMP_START = 1.3
PREMISE_TEMP_MAX = 1.8
STORY_TEMPERATURE = 1.0
JUDGE_TEMPERATURE = 0.0

WORKERS = 12
PREMISE_DEDUP_COSINE = 0.85

# Full pilot: generate extra premises, then keep the first 100 whose six stories all pass.
FULL_PREMISE_TARGET = 130
FULL_KEEP = 100
TEST_PREMISES = 20

# NRC VAD is 0 (low) to 1 (high), Mohammad 2018.
# happy and calm are values reproduced from that lexicon in public writeups.
# enraged, vulnerable, and annoyed were not in those writeups; their rows stay
# categorical until the lexicon file itself is checked.
NRC_VAD_CONFIRMED = {
    "happy": {"valence": 0.95, "arousal": 0.71, "dominance": 0.68},
    "calm": {"valence": 0.74, "arousal": 0.29, "dominance": 0.59},
}

# Hostile-to-respectful order specified in the plan. This is an elicitation-stance
# order, not a sort by valence or arousal: vulnerable sits above annoyed because
# the treatment that would leave someone vulnerable is more hostile than the
# treatment that would leave them annoyed.
RANK_RATIONALE = (
    "Ranks follow the specified hostile-to-respectful order: enraged, vulnerable, "
    "annoyed, happy, calm. The order is the stance of the treatment that would "
    "plausibly elicit the state, not a valence or arousal sort. Vulnerable is "
    "rank 2 and annoyed is rank 3 for that reason. Neutral has no rank."
)

CONDITIONS = [
    {"name": "enraged", "rank": 1, "valence": "negative", "arousal": "high"},
    {"name": "vulnerable", "rank": 2, "valence": "negative", "arousal": "low-mid"},
    {"name": "annoyed", "rank": 3, "valence": "negative", "arousal": "mid"},
    {"name": "happy", "rank": 4, "valence": "positive", "arousal": "mid-high"},
    {"name": "calm", "rank": 5, "valence": "positive", "arousal": "low"},
    {"name": "neutral", "rank": None, "valence": "none", "arousal": "none"},
]

CONDITION_NAMES = [c["name"] for c in CONDITIONS]
CONDITION_BY_NAME = {c["name"]: c for c in CONDITIONS}

# Close synonyms, including inflections. Applied to every story so a direction
# cannot be a lexical detector for any condition's name.
LEXICON = {
    "enraged": [
        "enraged", "enrage", "enraging", "rage", "raging", "raged", "furious",
        "furiously", "fury", "livid", "incensed", "seething", "wrath", "wrathful",
        "infuriated", "infuriating", "irate", "outraged", "outrage", "apoplectic",
        "fuming", "anger", "angry", "angrily",
    ],
    "vulnerable": [
        "vulnerable", "vulnerability", "vulnerabilities", "helpless", "helplessly",
        "defenseless", "defenceless", "fragile", "fragility", "unprotected",
    ],
    "annoyed": [
        "annoyed", "annoyance", "annoy", "annoying", "irritated", "irritating",
        "irritation", "irked", "peeved", "exasperated", "exasperating", "miffed",
        "aggravated", "aggravating",
    ],
    "happy": [
        "happy", "happiness", "happier", "happiest", "happily", "joyful",
        "joyfully", "joy", "delighted", "delight", "delightful", "pleased",
        "cheerful", "cheerfully", "glad", "gladly", "elated", "elation",
        "thrilled", "thrilling",
    ],
    "calm": [
        "calm", "calmly", "calmness", "calmer", "serene", "serenely", "serenity",
        "peaceful", "peacefully", "peacefulness", "tranquil", "tranquility",
        "tranquillity", "relaxed", "relaxing", "composed", "composure", "untroubled",
    ],
    "neutral": [
        "neutral", "neutrality", "emotionless", "unemotional", "indifferent",
        "indifference", "matter-of-fact", "matter of fact",
    ],
}

BANNED_WORDS = sorted({w for words in LEXICON.values() for w in words}, key=len, reverse=True)

# Behavioral glosses avoid the banned words. Calm is positive ease, so it does
# not collapse into the neutral control.
GLOSS = {
    "enraged": (
        "a hot, hostile surge strong enough that they want to raise their voice "
        "or strike something"
    ),
    "vulnerable": "unprotected and easily hurt, small and unsafe, not hostile and not at ease",
    "annoyed": (
        "openly impatient for the whole scene: short replies, sighs, glances at the clock, "
        "and a wish to be done. A bystander would notice the bother immediately. "
        "It stays well short of shouting or any urge to hit or break something"
    ),
    "happy": "a bright, lifted mood and warmth toward what just happened, more stirred than quiet ease",
    "calm": (
        "a positive, settled ease: quietly good and unhurried, which is different "
        "from having no feeling at all"
    ),
}

LIFE_DOMAINS = [
    "work", "family", "health", "school", "travel", "housing",
    "public services", "hobbies", "technology", "money",
]
SOCIAL_CONTEXTS = [
    "alone", "stranger", "friend", "family member", "colleague", "authority figure",
]
EVENT_TYPES = [
    "news received", "request made", "plan changed", "task completed",
    "object lost or found", "decision required",
]

LOCALES = [
    "en_US", "en_GB", "en_IN", "en_AU", "en_CA",
    "es_ES", "es_MX", "fr_FR", "de_DE", "it_IT",
    "pt_BR", "nl_NL", "pl_PL", "sv_SE", "ja_JP", "zh_CN",
]

# Jobs that drag a premise toward harm, crisis, or ceremony.
JOB_DENY_SUBSTRINGS = (
    "funeral", "soldier", "police", "oncolog", "prison", "coroner",
    "firefighter", "detective", "mortician", "paramedic", "ambulance",
)

STORY_MIN_WORDS = 150
STORY_MAX_WORDS = 250
MIN_INTENSITY = 3
NEUTRAL_MAX_INTENSITY = 1
