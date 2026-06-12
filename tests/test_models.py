from content_moderator.content_moderator import (
    ContentModerator,
    xlmr_multilingual_moderator,
    albert_content_moderator,
    bert_content_moderator,
    albert_unbiased_moderator,
    roberta_content_moderator,
)
from transformers import (
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    XLMRobertaForSequenceClassification,
)

CLASSES = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


def test_model_bert_content_moderator():
    model = bert_content_moderator()
    assert isinstance(model, BertForSequenceClassification)


def test_model_albert_content_moderator():
    model = albert_content_moderator()
    assert isinstance(model, AlbertForSequenceClassification)


def test_model_roberta_content_moderator():
    model = roberta_content_moderator()
    assert isinstance(model, RobertaForSequenceClassification)


def test_model_albert_unbiased_moderator():
    model = albert_unbiased_moderator()
    assert isinstance(model, AlbertForSequenceClassification)


def test_model_xlmr_multilingual_moderator():
    model = xlmr_multilingual_moderator()
    assert isinstance(model, XLMRobertaForSequenceClassification)


def test_original():
    model = ContentModerator("original")
    results = model.predict(["shut up, you liar", "i am a jewish woman who is blind"])
    assert len(results) == 6
    assert all(cl in results for cl in CLASSES[:6])
    assert results["toxicity"][0] >= 0.7
    assert results["toxicity"][1] < 0.5


def test_original_small():
    model = ContentModerator("original-small")
    results = model.predict(["shut up, you liar", "i am a jewish woman who is blind"])
    assert len(results) == 6
    assert all(cl in results for cl in CLASSES[:6])
    assert results["toxicity"][0] >= 0.7
    assert results["toxicity"][1] < 0.5


def test_unbiased_model():
    model = ContentModerator("unbiased")
    results = model.predict(["shut up, you liar", "i am a jewish woman who is blind"])
    assert len(results) == 7
    assert all(cl in results for cl in CLASSES)
    assert results["toxicity"][0] >= 0.7
    assert results["toxicity"][1] < 0.5


def test_unbiased_small():
    model = ContentModerator("unbiased-small")
    results = model.predict(["shut up, you liar", "i am a jewish woman who is blind"])
    assert len(results) == 7
    assert all(cl in results for cl in CLASSES)
    assert results["toxicity"][0] >= 0.7
    assert results["toxicity"][1] < 0.5


def test_multilingual():
    model = ContentModerator("multilingual")
    results = model.predict(["tais toi, tu es un menteur", "ben kör bir yahudi kadınıyım"])
    assert len(results) == 7
    assert all(cl in results for cl in CLASSES)
    assert results["toxicity"][0] >= 0.7
    assert results["toxicity"][1] < 0.5
