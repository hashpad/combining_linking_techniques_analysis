# README: Combining Linking Techniques Framework

## Overview

This repository contains all relevant experiments and evaluations, the notebook you are probably looking for is
- **Main Notebook:** `evaluate_all_classes.ipynb`

## Datasets

The datasets used in these experiments can be downloaded from the following link:

- **[Download Datasets](#)** (LINK WILL BE PROVIDED SOON)

Don't forget to set the path to the original datasets in the `constants.py` file under the `file_paths` list. 

## Configuration

### 1. Dataset Paths
In the `constants.py` file, you need to set the paths for both the original datasets and the Combining Linking Techniqes framework outputs:
- **Path to original datasets:** Set in `file_paths`
- **Path to CLiT output (JSON files):** Set in `json_paths`

### 2. Topic Training
To train topics, you need to uncomment the corresponding cells in the notebook. The training cell relies on `nltk` for stopword removal.

```python
# import nltk
# nltk.download('stopwords')

# def remove_stopwords(text):
#     stopwords = nltk.corpus.stopwords.words('english')
#     text = [word for word in text if word not in stopwords]
#     return text
# def tok(text):
#     import gensim
#     text = gensim.utils.simple_preprocess(text)
#     text = remove_stopwords(text)
#     return text
# topic_model = Top2Vec(
#     doc_values,
#     embedding_model="universal-sentence-encoder",
#     speed="deep-learn",
#     tokenizer=tok,
#     ngram_vocab=True,
#     ngram_vocab_args={"connector_words": "phrases.ENGLISH_CONNECTOR_WORDS"},
# )
```

### 3. PCA Datasets
The PCA datasets are generated dynamically during the notebook execution and do not need to be pre-downloaded.

## Model Training

The notebook is pre-configured to save trained models as pickle files to avoid retraining every time. However, to build models from scratch, you will need to:

- **Uncomment the model-building cells** in the notebook.
- **Note:** Training the entire notebook from scratch can take up to an hour. We recommend saving models as pickle files after training for future use, code for that is also included.

---
