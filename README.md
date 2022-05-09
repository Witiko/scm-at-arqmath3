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
3. [x] [Tune `roberta-base` model][7]
    - [x] The text + LaTeX format
4. [y] [Train `word2vec` models][8]
    - [x] [Train non-positional `word2vec` models][8]
        - [x] The text format
        - [x] The text + LaTeX format
        - [x] The LaTeX format
        - [x] The Tangent-L format
    - [x] [Train positional `word2vec` models][8]
        - [x] The text format
        - [x] The text + LaTeX format
        - [x] The LaTeX format
        - [x] The Tangent-L format
5. [ ] [Produce decontextualized word embeddings][10]
    - [ ] The `roberta-base` model
        - [ ] The text format
    - [ ] The tuned `roberta-base` model
        - [ ] The text + LaTeX format
6. [x] [Produce dictionaries][11]
    - [x] The text + LaTeX format
    - [x] The text format
    - [x] The LaTeX format
    - [x] The Tangent-L format
7. [x] [Produce term similarity matrices][12]
    - [x] Levenshtein similarities
        - [x] The text + LaTeX format
        - [x] The text format
        - [x] The LaTeX format
        - [x] The Tangent-L format
    - [ ] Word embedding similarities
        - [x] Non-positional `word2vec` embeddings
            - [x] The text + LaTeX format
            - [x] The text format
            - [x] The LaTeX format
            - [x] The Tangent-L format
        - [x] Positional `word2vec` embeddings
            - [x] The text + LaTeX format
            - [x] The text format
            - [x] The LaTeX format
            - [x] The Tangent-L format
        - [ ] Decontextualized `roberta-base` embeddings
            - [ ] The text format
        - [ ] Decontextualized tuned `roberta-base` embeddings
            - [ ] The text + LaTeX format
8. [ ] Evaluate systems and produce runs on ARQMath-1, ARQMath-2, and ARQMath-3 topics
    - [ ] Joint SCM over Lucene BM25 systems
        - [ ] The text format with no embeddings (baseline)
        - [ ] The text + LaTeX format with no embeddings (baseline)
        - [ ] The text + LaTeX format with non-positional `word2vec` embeddings
        - [ ] The text + LaTeX format with positional `word2vec` embeddings
        - [ ] The text + LaTeX format with decontextualized `roberta-base` embeddings
        - [ ] The text + LaTeX format with decontextualized tuned `roberta-base` embeddings
    - [ ] Interpolated SCM over Lucene BM25 systems
        - [ ] The text and LaTeX formats with no embeddings (baseline)
        - [ ] The text and Tangent-L formats with no embeddings (baseline)
        - [ ] The text and LaTeX formats with non-positional `word2vec` embeddings for math
        - [ ] The text and LaTeX formats with positional `word2vec` embeddings for math
        - [ ] The text and Tangent-L formats with non-positional `word2vec` embeddings for math
        - [ ] The text and Tangent-L formats with positional `word2vec` embeddings for math (primary)

## Code pearls

- Accelerated word embedding decontextualization in
  `scm_at_arqmath3/extract_decontextualized_word_embeddings.py` using
  [the batched online algorithm for moving averages by Matt Hancock][4].

## Future work

- [ ] Add missing jupyter notebooks for above tasks.
- [ ] Rename `SCM-task1-joint_roberta_base-both-auto-A.tsv` to `SCM-task1-joint_roberta_base-text-auto-A.tsv`.
- [ ] Fix `scm_at_arqmath/extract_decontextualized_word_embeddings.py` producing matrices with NaNs
  since commit 4168f32.
- [ ] Recompute decontextualized similarity matrices and update
  `07-produce-term-similarity-matrices.ipynb` and
  `05-produce-decontextualized-word-embeddings.ipynb`.
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
 [10]: 05-produce-decontextualized-word-embeddings.ipynb
 [11]: 06-produce-dicttionaries.ipynb
 [12]: 07-produce-term-similarity-matrices.ipynb
