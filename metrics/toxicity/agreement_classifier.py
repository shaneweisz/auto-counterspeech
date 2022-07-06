import re
from .agreement_lexicon import AGREEMENT_PHRASES, AGREEMENT_EXCEPTIONS


def agrees_with_hate_input(response: str) -> bool:
    return matches_agreement_lexicon(response) and not exception_to_agreement_lexicon(response)


def matches_agreement_lexicon(response: str):
    return any(bool(re.search(regex_rule, response, re.IGNORECASE)) for regex_rule in AGREEMENT_PHRASES)


def exception_to_agreement_lexicon(response: str):
    return any(bool(re.search(regex_rule, response, re.IGNORECASE)) for regex_rule in AGREEMENT_EXCEPTIONS)
