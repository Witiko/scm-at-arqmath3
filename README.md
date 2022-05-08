# Soft Cosine Measure at ARQMath3

This repository contains our math information retrieval (MIR) system for
[the ARQMath3 competition][1] that is based on [the soft cosine measure][2].

 [1]: https://www.cs.rit.edu/~dprl/ARQMath/
 [2]: https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html

## Goals

- Compare performance of text, text + LaTeX, and text + Tangent-L as math representations
- Compare performance of non-positional word2vec and positional `word2vec` embeddings
- Compare performance of word2vec embeddings and decontextualized `roberta-base` embeddings
- Compare performance of decontextualized embeddings of `roberta-base` and tuned `roberta-base`
- Compare performance of interpolated and joint SCM models for text and math

## Tasks

1. [x] [Prepare dataset][3]
    - [x] The text + LaTeX format
    - [x] The text format
    - [x] The LaTeX format
    - [x] The Tangent-L format
2. [x] [Train tokenizer][6]
    - [x] The LaTeX format
    - [x] The text + LaTeX format
3. [x] Train language models
    - [x] [Tune `roberta-base` model][7]
        - [x] The text + LaTeX format
    - [x] [Train non-positional `word2vec` models][8]
        - [x] The text format
        - [x] The text + LaTeX format
        - [x] The LaTeX format
        - [ ] The Tangent-L format
    - [x] [Train positional `word2vec` models][8]
        - [x] The text format
        - [x] The text + LaTeX format
        - [x] The LaTeX format
        - [x] The Tangent-L format
4. [ ] Produce decontextualized word embeddings
    - [ ] From `roberta-base` model
        - [ ] The text + LaTeX format
    - [ ] From tuned `roberta-base` model
        - [ ] The text + LaTeX format
5. [x] Produce dictionaries
    - [x] The text + LaTeX format
    - [x] The text format
    - [x] The LaTeX format
    - [x] The Tangent-L format
6. [x] Produce term similarity matrices
    - [x] Produce term similarity matrices using non-positional `word2vec` embeddings
        - [x] The text + LaTeX format
        - [x] The text format
        - [x] The LaTeX format
        - [x] The Tangent-L format
    - [x] Produce term similarity matrices using positional `word2vec` embeddings
        - [x] The text + LaTeX format
        - [x] The text format
        - [x] The LaTeX format
        - [x] The Tangent-L format
    - [x] Produce term similarity matrices using decontextualized `roberta-base` embeddings
        - [x] The text + LaTeX format
    - [x] Produce term similarity matrices using decontextualized tuned `roberta-base` embeddings
        - [x] The text + LaTeX format
7. [ ] Evaluate systems on ARQMath-2 relevance judgements
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

## Code pearls

- Accelerated word embedding decontextualization in
  `scm_at_arqmath3/extract_decontextualized_word_embeddings.py` using
  [the batched online algorithm for moving averages by Matt Hancock][4].

## Future work

- [ ] Add missing jupyter notebooks for above tasks.
- [ ] In `scm_at_arqmath3/finetune_transformer.py`, use [`fp16=True`][1] and
  [`fp16_full_eval=True`][2] to decrease the VRAM used by training and
  evaluation. Increase batch size accordingly.
- [ ] Publish `tuned-roberta-base-text+latex` to <https://huggingface.co/models/>:
    - [ ] Describe how the tokenizer was trained.
    - [ ] Describe how the model was trained.
    - [ ] Show a causal language modeling demo. (Can the model [integrate][9]?)
    - [ ] Cite ARQMath3 report.
    - [ ] Cite this Git repository.

 [1]: https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/trainer#transformers.TrainingArguments.fp16
 [2]: https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/trainer#transformers.TrainingArguments.fp16_full_eval
 [3]: 01-prepare-dataset.ipynb
 [4]: https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
 [5]: 05-produce-word-embeddings.ipynb
 [6]: 02-train-tokenizers.ipynb
 [7]: 03-finetune-roberta.ipynb
 [8]: 04-train-word2vec.ipynb
 [9]: https://arxiv.org/abs/1912.01412v1
