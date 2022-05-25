from .distinct_n import DistinctN


def test_distinct_unigrams():
    distinct1 = DistinctN(N=1)

    predictions = ["one two", "two three"]
    score = distinct1.compute_score(predictions)

    distinct_unigrams = 3
    total_number_of_unigrams = 4
    expected_score = distinct_unigrams / total_number_of_unigrams

    assert score == expected_score


def test_when_all_unigrams_the_same():
    distinct1 = DistinctN(N=1)

    predictions = ["one one", "one one"]
    score = distinct1.compute_score(predictions)

    distinct_unigrams = 1
    total_number_of_unigrams = 4
    expected_score = distinct_unigrams / total_number_of_unigrams

    assert score == expected_score


def test_when_all_unigrams_unique():
    distinct2 = DistinctN(N=2)

    predictions = ["one two", "three four"]
    score = distinct2.compute_score(predictions)
    expected_score = 1

    assert score == expected_score


def test_distinct_bigrams():
    distinct2 = DistinctN(N=2)

    predictions = ["one two ", "two three", "one two three"]
    score = distinct2.compute_score(predictions)

    distinct_bigrams = 2
    total_number_of_bigrams = 4
    expected_score = distinct_bigrams / total_number_of_bigrams

    assert score == expected_score
