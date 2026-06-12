import gc

import torch


def test_torch_hub_models():
    result = torch.hub.list("your-username/ai-content-moderation-platform", skip_validation=True, trust_repo=True)  # pyright: ignore[reportArgumentType]


def test_torch_hub_bert():
    model = torch.hub.load("your-username/ai-content-moderation-platform", "bert_content_moderator", skip_validation=True, trust_repo=True)  # pyright: ignore[reportArgumentType]
    del model
    gc.collect()


def test_torch_hub_roberta():
    model = torch.hub.load("your-username/ai-content-moderation-platform", "roberta_content_moderator", skip_validation=True, trust_repo=True)  # pyright: ignore[reportArgumentType]
    del model
    gc.collect()


def test_torch_hub_multilingual():
    model = torch.hub.load("your-username/ai-content-moderation-platform", "xlmr_multilingual_moderator", skip_validation=True, trust_repo=True)  # pyright: ignore[reportArgumentType]
    del model
    gc.collect()
