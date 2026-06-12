<div align="center">

# 🛡️ AI Content Moderation Platform
## Multi-label Toxic Content Classification with ⚡ Pytorch Lightning and 🤗 Transformers

[![PyPI version](https://badge.fury.io/py/content_moderator.svg)](https://badge.fury.io/py/content_moderator)
![CI testing](https://github.com/your-username/ai-content-moderation-platform/workflows/CI%20testing/badge.svg)
![Lint](https://github.com/your-username/ai-content-moderation-platform/workflows/Lint/badge.svg)

</div>

![Examples image](examples.png)

## Description

Trained models & code to predict toxic comments on 3 classification challenges: Toxic comment classification, Unintended Bias in Toxic comments, Multilingual toxic comment classification.

Dependencies:
- For inference:
  - 🤗 Transformers
  - ⚡ Pytorch lightning
- For training will also need:
  - Kaggle API (to download data)


| Challenge | Year | Goal | Original Data Source | Model Name | Top Kaggle Leaderboard Score % | Platform Score %
|-|-|-|-|-|-|-|
| [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | 2018 | build a multi-headed model that's capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate. | Wikipedia Comments | `original` | 98.86 | 98.64
| [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) | 2019 | build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities. | Civil Comments | `unbiased` | 94.73 | 93.74
| [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification) | 2020 | build effective multilingual models | Wikipedia Comments + Civil Comments | `multilingual` | 95.36 | 92.11


### Multilingual model language breakdown

| Language Subgroup   |   Subgroup size |   Subgroup AUC Score % |
|:-----------|----------------:|---------------:|
🇮🇹 it       |     8494   |   89.18 |
🇫🇷 fr      |    10920   |   89.61 |
🇷🇺 ru     |    10948   |   89.81 |
🇵🇹 pt      |    11012   |   91.00 |
🇪🇸 es      |     8438   |   92.74 |
🇹🇷 tr     |    14000   |   97.19 |

## Limitations and ethical considerations

If words that are associated with swearing, insults or profanity are present in a comment, it is likely that it will be classified as toxic, regardless of the tone or the intent of the author e.g. humorous/self-deprecating. This could present some biases towards already vulnerable minority groups.

The intended use of this platform is for research purposes, fine-tuning on carefully constructed datasets that reflect real world demographics and/or to aid content moderators in flagging out harmful content quicker.

Some useful resources about the risk of different biases in toxicity or hate speech detection are:
- [The Risk of Racial Bias in Hate Speech Detection](https://homes.cs.washington.edu/~msap/pdfs/sap2019risk.pdf)
- [Automated Hate Speech Detection and the Problem of Offensive Language](https://ojs.aaai.org/index.php/ICWSM/article/view/14955)
- [Racial Bias in Hate Speech and Abusive Language Detection Datasets](https://aclanthology.org/W19-3504/)

## Quick prediction

The `multilingual` model has been trained on 7 different languages so it should only be tested on: `english`, `french`, `spanish`, `italian`, `portuguese`, `turkish` or `russian`.

```bash
# install the platform

pip install content_moderator

```
```python

from content_moderator import ContentModerator

# each model takes in either a string or a list of strings

results = ContentModerator('original').predict('example text')

results = ContentModerator('unbiased').predict(['example text 1','example text 2'])

results = ContentModerator('multilingual').predict(['example text','exemple de texte','texto de ejemplo','testo di esempio','texto de exemplo','örnek metin','пример текста'])

# to specify the device the model will be allocated on (defaults to cpu), accepts any torch.device input

model = ContentModerator('original', device='cuda')

# optional to display results nicely (will need to pip install pandas)

import pandas as pd

print(pd.DataFrame(results, index=input_text).round(5))

```


## Labels
All challenges have a toxicity label. The toxicity labels represent the aggregate ratings of up to 10 annotators according the following schema:
- **Very Toxic** (a very hateful, aggressive, or disrespectful comment that is very likely to make you leave a discussion or give up on sharing your perspective)
- **Toxic** (a rude, disrespectful, or unreasonable comment that is somewhat likely to make you leave a discussion or give up on sharing your perspective)
- **Hard to Say**
- **Not Toxic**

### Toxic Comment Classification Challenge
This challenge includes the following labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

### Jigsaw Unintended Bias in Toxicity Classification
- `toxicity`
- `severe_toxicity`
- `obscene`
- `threat`
- `insult`
- `identity_attack`
- `sexual_explicit`

### Jigsaw Multilingual Toxic Comment Classification
Final evaluation is only on:
- `toxicity`

## How to run

First, install dependencies
```bash
# clone project

git clone https://github.com/your-username/ai-content-moderation-platform

# create virtual env

python3 -m venv moderation-env
source moderation-env/bin/activate

# install project
pip install -e ai-content-moderation-platform

# or for training
pip install -e 'ai-content-moderation-platform[dev]'

cd ai-content-moderation-platform

 ```

## Prediction

Trained models summary:

|Model name| Transformer type| Data from
|:--:|:--:|:--:|
|`original`| `bert-base-uncased` | Toxic Comment Classification Challenge
|`unbiased`| `roberta-base`| Unintended Bias in Toxicity Classification
|`multilingual`| `xlm-roberta-base`| Multilingual Toxic Comment Classification

```bash

# load model by name

python run_prediction.py --input 'example' --model_name original

# load model from checkpoint path

python run_prediction.py --input 'example' --from_ckpt_path model_path

# save results to a .csv file

python run_prediction.py --input test_set.txt --model_name original --save_to results.csv

# to see usage

python run_prediction.py --help

```

Checkpoints can be downloaded from the latest release or via the Pytorch hub API with the following names:
- `bert_content_moderator`
- `roberta_content_moderator`
- `xlmr_multilingual_moderator`

```bash
model = torch.hub.load('your-username/ai-content-moderation-platform','bert_content_moderator')
```

Importing in python:

```python

from content_moderator import ContentModerator

results = ContentModerator('original').predict('some text')

results = ContentModerator('unbiased').predict(['example text 1','example text 2'])

results = ContentModerator('multilingual').predict(['example text','exemple de texte','texto de ejemplo','testo di esempio','texto de exemplo','örnek metin','пример текста'])

# to display results nicely

import pandas as pd

print(pd.DataFrame(results,index=input_text).round(5))

```


## Training

 If you do not already have a Kaggle account:
 - you need to create one to be able to download the data
 - go to My Account and click on Create New API Token - this will download a kaggle.json file
 - make sure this file is located in ~/.kaggle

 ```bash

# create data directory

mkdir moderation_data
cd moderation_data

# download data

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
unzip jigsaw-toxic-comment-classification-challenge.zip -d moderation-comment-classification-challenge
find moderation-comment-classification-challenge -name '*.csv.zip' | xargs -n1 unzip -d moderation-comment-classification-challenge

kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
unzip jigsaw-unintended-bias-in-toxicity-classification.zip -d moderation-unintended-bias-classification

kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification
unzip jigsaw-multilingual-toxic-comment-classification.zip -d moderation-multilingual-comment-classification

```

## Start Training
 ### Toxic Comment Classification Challenge

 ```bash

python preprocessing_utils.py --test_csv moderation_data/moderation-comment-classification-challenge/test.csv --update_test

python train.py --config configs/Toxic_comment_classification_BERT.json
```

 ### Unintended Bias Challenge

```bash

python train.py --config configs/Unintended_bias_toxic_comment_classification_RoBERTa_combined.json

```

 ### Multilingual Toxic Comment Classification

```bash

python preprocessing_utils.py --test_csv moderation_data/moderation-multilingual-comment-classification/test.csv --update_test

python train.py --config configs/Multilingual_toxic_comment_classification_XLMR.json

```

### Monitor progress with tensorboard

 ```bash

tensorboard --logdir=./saved

```

## Model Evaluation

### Toxic Comment Classification Challenge

```bash

python evaluate.py --checkpoint saved/lightning_logs/checkpoints/example_checkpoint.pth --test_csv test.csv

```

### Unintended Bias Challenge

```bash

python evaluate.py --checkpoint saved/lightning_logs/checkpoints/example_checkpoint.pth --test_csv test.csv

# to get the final bias metric
python model_eval/compute_bias_metric.py

```

### Multilingual Toxic Comment Classification

```bash

python evaluate.py --checkpoint saved/lightning_logs/checkpoints/example_checkpoint.pth --test_csv test.csv

```
