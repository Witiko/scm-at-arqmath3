# Soft Cosine Measure at ARQMath3

This repository contains our math information retrieval (MIR) system for
[the ARQMath3 competition][1] that is based on [the soft cosine measure][2].

 [1]: https://www.cs.rit.edu/~dprl/ARQMath/
 [2]: https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html

## Goals

- Compare performance of text, text + LaTeX, and text + Tangent-L as math representations
- Compare performance of non-positional word2vec and positional `word2vec` embeddings
- Compare performance of word2vec embeddings and decontextualized `roberta-base` embeddings
- Compare performance of decontextualized `roberta-base` embeddings and decontextualized tuned `roberta-base` embeddings
- Compare performance of interpolated and joint SCM models for text and math

## Tasks

1. [ ] [Prepare dataset][3]
    - [ ] The text format
    - [ ] The text + LaTeX format
    - [x] The LaTeX format
    - [x] The Tangent-L format
2. [x] [Train tokenizers][6]
    - [x] Train tokenizer
        - [x] The LaTeX format
    - [ ] Tune `roberta-base` tokenizer
        - [ ] The text + LaTeX format
3. [x] Train language models
    - [ ] [Tune `roberta-base` model][7]
        - [ ] The text + LaTeX format
    - [ ] [Train non-positional `word2vec` models][8]
        - [ ] The text format
        - [ ] The text + LaTeX format
        - [x] The LaTeX format
        - [x] The Tangent-L format
    - [ ] [Train positional `word2vec` models][8]
        - [ ] The text format
        - [ ] The text + LaTeX format
        - [x] The LaTeX format
        - [x] The Tangent-L format
4. [ ] [Produce word embeddings][5]
    - [ ] Produce non-positional `word2vec` embeddings
        - [ ] The text format
        - [ ] The text + LaTeX format
        - [x] The LaTeX format
        - [x] The Tangent-L format
    - [ ] Produce positional `word2vec` embeddings
        - [ ] The text format
        - [ ] The text + LaTeX format
        - [x] The LaTeX format
        - [x] The Tangent-L format
    - [ ] Produce decontextualized `roberta-base` embeddings
      <!-- See https://github.com/MIR-MU/regemt/blob/main/embedder.py -->
        - [ ] The text + LaTeX format
    - [ ] Produce decontextualized tuned `roberta-base` embeddings
        - [ ] The text + LaTeX format
5. [ ] Produce dictionaries and Lucene BM25 models
   <!-- See mir:/mnt/storage/2022-04-05-introduction-to-information-retrieval/ARQMath 2021 lab/ARQMath solution by Vítek Novotný (0.424 nDCG') -->
   <!-- See https://drive.google.com/file/d/1T06JUueKi0fZpyRNjspjfqGRda0T6iAp/view -->
    - [ ] The text format
    - [ ] The text + LaTeX format
    - [ ] The LaTeX format
    - [ ] The Tangent-L format
6. [ ] Produce term similarity matrices
   <!-- See mir:scm-demo-for-radim-rehurek/ -->
    - [ ] Produce Levenshtein term similarity matrices
        - [ ] The text format
        - [ ] The text + LaTeX format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] Produce term similarity matrices using non-positional `word2vec` embeddings
        - [ ] The text format
        - [ ] The text + LaTeX format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] Produce term similarity matrices using positional `word2vec` embeddings
        - [ ] The text format
        - [ ] The text + LaTeX format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] Produce term similarity matrices using decontextualized `roberta-base` embeddings
        - [ ] The text + LaTeX format
    - [ ] Produce term similarity matrices using decontextualized tuned `roberta-base` embeddings
        - [ ] The text + LaTeX format
7. [ ] Evaluate systems on ARQMath-2 relevance judgements
   <!-- See mir:/mnt/storage/2022-04-05-introduction-to-information-retrieval/ARQMath 2021 lab/ARQMath solution by Vítek Novotný (0.424 nDCG') -->
    - [ ] Evaluate joint SCM over Lucene BM25 systems
        - [ ] The text format with no embeddings (baseline)
        - [ ] The text + LaTeX format with no embeddings (baseline)
        - [ ] The text + LaTeX format with non-positional `word2vec` embeddings
        - [ ] The text + LaTeX format with positional `word2vec` embeddings
        - [ ] The text + LaTeX format with decontextualized `roberta-base` embeddings
        - [ ] The text + LaTeX format with decontextualized tuned `roberta-base` embeddings
    - [ ] Evaluate interpolated SCM over Lucene BM25 systems
        - [ ] The text and LaTeX formats with no embeddings (baseline)
        - [ ] The text and Tangent-L formats with no embeddings (baseline)
        - [ ] The text and LaTeX formats with non-positional `word2vec` embeddings for math
        - [ ] The text and LaTeX formats with positional `word2vec` embeddings for math
        - [ ] The text and Tangent-L formats with non-positional `word2vec` embeddings for math
        - [ ] The text and Tangent-L formats with positional `word2vec` embeddings for math (primary)
8. [ ] Select one primary and four alternative systems
9. [ ] Produce runs of five systems on ARQMath-1, ARQMath-2, and ARQMath-3 topics

## Current issues

- [ ] Lack of validation dataset in [`03-finetune-roberta.ipynb`][7] notebook training.
- [ ] Lack of text format.

 [3]: 01-prepare-dataset.ipynb
 [5]: 05-produce-word-embeddings.ipynb
 [6]: 02-train-tokenizers.ipynb
 [7]: 03-finetune-roberta.ipynb
 [8]: 04-train-word2vec.ipynb
