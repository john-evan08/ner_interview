# ner_interview

## Run

```bash
python exploration.py # plots folder
python prepare_dataset.py # data folder
python train.py # models and runs folder
python test.py # test folder
```

1. Quick Data Exploration (ex of sequence, labels distribution)

2. Research on (bi)LSTM like models, pre-trained embeddings, classificator at the end (CRF/Linear)

3. Implement first model with train/val/test and plotting metrics

4. Identify challenges, edge cases and come up with solutions

- there are things that HAVE TO be done. First one is adding a CRF Layer, able to enforce correct B- and I- tagging, and learn which transitions are likely (B-PER -> I-PER) and which are not.

- The main challenge is the unbalanced dataset with a rpz of 1.5K I-LOC vs 200K "0" negative. So we need to find solutions to tackle that.
  Oversampling or downsampling is hard to set up here with sequence of tokens.
  The CRF helps but we also need to add some weighted loss

- The other challenge is the eval metrics. f1-score is nice but for NER the application-related eval metric we look at is overlap with `sequeval`. Sometimes, you get the right entity, but you begin the word before or finish the word after.

- We did not use properly the Case information.

- We were not able to identify overfitting properly or to exploit the train/validation plots.
