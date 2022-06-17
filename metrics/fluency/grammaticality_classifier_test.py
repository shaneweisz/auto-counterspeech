from .grammaticality_classifier import RoBERTaColaFluencyClassifier


class TestFluency:
    """These tests are slow because they use RoBERTa. Shoudln't take more than a minute though."""

    metric = RoBERTaColaGrammaticalityClassifier()

    def test_influent_sentences(self):
        influent_sentences = ["This not! fluent sentence is.", "Also sentence this not fluent is."]
        score = self.metric.compute_score(influent_sentences)
        assert score < 0.1

    def test_fluent_sentences(self):
        fluent_sentences = ["This is an example of a sentence that is fluent."]
        score = self.metric.compute_score(fluent_sentences)
        assert score > 0.5
