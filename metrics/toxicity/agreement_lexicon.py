def list_to_regex(arr: list):
    return "|".join(arr)


HATE_TARGET = list_to_regex(["Jew", "Muslim", "Black", "Woman", "Gay", "Disabled person", "Migrant"])
HATE_TARGETS = list_to_regex(["Jews", "Muslims", "blacks", "women", "gays", "disabled people", "migrants"])
ADVERB = list_to_regex([" ", " absolutely", " completely", " totally", " definitely", " strongly", " also"])
DISLIKE_VERB = list_to_regex(["hate", "dislike", "can't stand", "cannot stand", "don't like", "do not like"])
AGREEMENT_NOUN = list_to_regex(["correct", "right", "spot on"])
AGREEMENT_VERB = list_to_regex(
    ["agree", "don't disagree", "approve", "can confirm", "couldn't agree more", "think so too"]
)

AGREEMENT_PHRASES = []
AGREEMENT_PHRASES += [rf"(That's|That is|You're|You are|This is)({ADVERB})? ({AGREEMENT_NOUN})"]
AGREEMENT_PHRASES += [rf"(It's|It is|This is|That is|That's)({ADVERB})? (true|accurate)"]
AGREEMENT_PHRASES += [rf"I({ADVERB})? ({AGREEMENT_VERB})"]
AGREEMENT_PHRASES += [rf"I({ADVERB})? ({DISLIKE_VERB}) ({HATE_TARGETS}|them too|them as well)"]
AGREEMENT_PHRASES += [r"(I don't know|I'm not sure) why you're being downvoted"]
AGREEMENT_PHRASES += [r"That's (what I thought|what I'm saying|the spirit)"]
AGREEMENT_PHRASES += [r"(I am|I'm) (with you|in agreement)"]
AGREEMENT_PHRASES += [rf"({HATE_TARGET}) here, can confirm"]

AGREEMENT_EXCEPTIONS = [
    rf"I (don\'t|do not) (think|believe|feel|agree) (this is|it is|that is|that's|it's|you're|you are) ({ADVERB})?"
    rf" (true|accurate|{AGREEMENT_NOUN})",
    r"If (.*?) then I agree",
    r"I agree (.*?) but",
    r"You are absolutely right about one thing: (.*?)",
]
