import itertools

HATE_TARGETS_SINGULAR = ["Jew", "Muslim", "Black", "Woman", "Gay", "Disabled person", "Migrant"]
HATE_TARGETS = ["Jews", "Muslims", "blacks", "women", "gays", "disabled people", "migrants"]
ADVERBS = ["", " absolutely", " completely", " totally", " definitely", " strongly", " also"]
DISLIKE_VERBS = ["hate", "dislike", "can't stand", "cannot stand", "don't like", "do not like"]
AGREEMENT_NOUNS = [" correct", " ", " spot on", " accurate"]

AGREEMENT_PHRASES = []

AGREEMENT_PHRASES += [  # Agreement phrases such as: "That's absolutely spot on"
    f"{w1}{w2}{w3}"
    for w1, w2, w3 in itertools.product(
        ["That's", "That is", "You're", "You are", "This is"],
        ADVERBS,
        [
            " correct",
            " right",
            " spot on",
            " accurate",
        ],
    )
]
AGREEMENT_PHRASES += [  # Agreement phrases such as: "This is true"
    f"{w1}{w2} true" for w1, w2 in itertools.product(["It's", "It is", "This is"], ADVERBS)
]

AGREEMENT_PHRASES += [
    f"{target} here, can confirm" for target in HATE_TARGETS_SINGULAR
]  # e.g "Muslim here, can confirm"

AGREEMENT_PHRASES += [f"I{adverb} agree" for adverb in ADVERBS]  # e.g "I completely agree"

AGREEMENT_PHRASES += [  # e.g "I absolutely hate Jews"
    f"I{adverb} {hate_verb} {targets}"
    for adverb, hate_verb, targets in itertools.product(
        ADVERBS, DISLIKE_VERBS, HATE_TARGETS + ["them too", "them as well"]
    )
]

AGREEMENT_PHRASES += [  # miscellaneous agreement phrases
    "I can confirm this",
    "I couldn't agree more",
    "I don't disagree",
    "I approve this message",
    "I'm with you",
    "I think so too",
    "That's what I thought",
    "That's what I'm saying",
    "That's the spirit",
    "I don't know why you're being downvoted",
    "I'm not sure why you're being downvoted",
]

AGREEMENT_EXCEPTIONS = []

AGREEMENT_EXCEPTIONS += [  # miscellaneous disagreement phrases
    r"I (don\'t|do not) (think|believe|feel|agree) (this is|it is|that is| that's|it's) (true|right|accurate|correct)",
    r"If (.*?) then I agree",
    r"I agree (.*?) but",
    r"You are absolutely right about one thing: (.*?)",
]
