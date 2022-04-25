# Soft Cosine Measure at ARQMath3

This repository contains our math information retrieval (MIR) system for
[the ARQMath3 competition][1] that is based on [the soft cosine measure][2].

 [1]: https://www.cs.rit.edu/~dprl/ARQMath/
 [2]: https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html

## Goals

- Compare performance of text, text + LaTeX, and text + Tangent-L as math representations
- Compare performance of non-positional word2vec and positional word2vec embeddings
- Compare performance of word2vec embeddings and decontextualized tuned roberta-base embeddings
- Compare performance of interpolated and joint SCM models for text and math
- Compare performance of SCM and ColBERT
- Investigate impact of number of nearest neighbors on performance of ColBERT

## Tasks

1. [x] [Prepare dataset][3]
    - [x] The LaTeX format
    - [x] The text + LaTeX format
    - [ ] The text + Tangent-L format
    - [ ] The Tangent-L format
    - [ ] The text format
2. [ ] [Train tokenizers][6]
    - [ ] Train tokenizer
        - [x] The LaTeX format
        - [ ] The Tangent-L format
    - [x] Tune `roberta-base` tokenizer
        - [x] The text + LaTeX format
3. [ ] Train language models
    - [x] [Tune `roberta-base` model][7]
        - [x] The text + LaTeX format
    - [ ] Train non-positional word2vec model
        - [ ] The text + LaTeX format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] Train positional word2vec model
        - [ ] The text + LaTeX format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
4. [ ] Produce word embeddings
    - [ ] Produce non-positional word2vec embeddings
        - [x] [The text format][5]
        - [ ] The text + LaTeX format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] Produce positional word2vec embeddings
        - [ ] The text + LaTeX format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] Produce decontextualized tuned `roberta-base` embeddings
        - [ ] The text + LaTeX format
5. [ ] Evaluate systems
    - [ ] Evaluate joint SCM over Lucene BM25 systems
        - [ ] The text format with no embeddings (baseline)
        - [ ] [The text + LaTeX format with no embeddings][4] (baseline)
        - [ ] The text + LaTeX format with non-positional word2vec embeddings
        - [ ] The text + LaTeX format with positional word2vec embeddings
        - [ ] The text + LaTeX format with decontextualized tuned `roberta-base` embeddings
    - [ ] Evaluate interpolated SCM over Lucene BM25 systems
        - [ ] The text and LaTeX formats with no embeddings (baseline)
        - [ ] The text and Tangent-L formats with no embeddings (baseline)
        - [ ] The text and LaTeX formats with non-positional word2vec embeddings for math
        - [ ] The text and LaTeX formats with positional word2vec embeddings for math
        - [ ] The text and Tangent-L formats with non-positional word2vec embeddings for math
        - [ ] The text and Tangent-L formats with positional word2vec embeddings for math (primary)
    - [ ] Evaluate ColBERT using text + LaTeX format with tuned `roberta-base` embeddings
        - [ ] Using one nearest word embedding
        - [ ] Using optimal number of k nearest word embeddings

 [3]: 01-prepare-dataset.ipynb
 [4]: https://colab.research.google.com/drive/1sc-JuE5SuU-vDZhqwWwPmFlxmEjReEN3
 [5]: https://fasttext.cc/docs/en/crawl-vectors.html
 [6]: 02-train-tokenizers.ipynb
 [7]: 03-finetune-roberta.ipynb
