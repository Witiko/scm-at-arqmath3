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
3. [ ] Train language models
    - [ ] [Tune `roberta-base` model][7]
        - [ ] The text + LaTeX format
    - [ ] [Train non-positional `word2vec` models][8]
        - [ ] The text format
        - [ ] The text + LaTeX format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] [Train positional `word2vec` models][8]
        - [ ] The text format
        - [ ] The text + LaTeX format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
4. [ ] Produce word embeddings
    - [ ] Produce non-positional `word2vec` embeddings
        - [ ] The text + LaTeX format
        - [ ] The text format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] Produce positional `word2vec` embeddings
        - [ ] The text + LaTeX format
        - [ ] The text format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] Produce decontextualized `roberta-base` embeddings
      <!-- See https://github.com/MIR-MU/regemt/blob/main/embedder.py -->
        - [ ] The text + LaTeX format
    - [ ] Produce decontextualized tuned `roberta-base` embeddings
        - [ ] The text + LaTeX format
5. [ ] Produce dictionaries
    - [ ] The text + LaTeX format
    - [ ] The text format
    - [ ] The LaTeX format
    - [ ] The Tangent-L format
6. [ ] Produce term similarity matrices
   <!-- See mir:/mnt/storage/2022-04-05-introduction-to-information-retrieval/ARQMath 2021 lab/ARQMath solution by Vítek Novotný (0.424 nDCG') -->
   <!-- See https://drive.google.com/file/d/1T06JUueKi0fZpyRNjspjfqGRda0T6iAp/view -->
   <!-- See mir:scm-demo-for-radim-rehurek/ -->
    - [ ] Produce Levenshtein term similarity matrices
        - [ ] The text + LaTeX format
        - [ ] The text format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] Produce term similarity matrices using non-positional `word2vec` embeddings
        - [ ] The text + LaTeX format
        - [ ] The text format
        - [ ] The LaTeX format
        - [ ] The Tangent-L format
    - [ ] Produce term similarity matrices using positional `word2vec` embeddings
        - [ ] The text + LaTeX format
        - [ ] The text format
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

 [3]: 01-prepare-dataset.ipynb
 [5]: 05-produce-word-embeddings.ipynb
 [6]: 02-train-tokenizers.ipynb
 [7]: 03-finetune-roberta.ipynb
 [8]: 04-train-word2vec.ipynb

## Future work

- [ ] In `scm_at_arqmath3/finetune_transformer.py`, use [`fp16=True`][1] and
  [`fp16_full_eval=True`][2] to decrease the VRAM used by training and
  evaluation. Increase batch size accordingly.
- [ ] In `scm_at_arqmath3/extract_decontextualized_word_embeddings.py`, accept
  dictionary and [build a tensor of decontextualized word embeddings][4] on GPU
  instead of transferring to CPU in order to increase throughput.
- [ ] Publish `tuned-roberta-base-text+latex` to <https://huggingface.co/models/>:
    - [ ] Describe how the tokenizer was trained.
    - [ ] Describe how the model was trained.
    - [ ] Show a causal language modeling demo. (Can the model [integrate][9]?)
    - [ ] Cite ARQMath3 report.
    - [ ] Cite this Git repository.

 [1]: https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/trainer#transformers.TrainingArguments.fp16
 [2]: https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/trainer#transformers.TrainingArguments.fp16_full_eval
 [4]: https://github.com/authoranonymous321/soft_mt_adaptation/blob/9ff8bc11499e133c110749cc9a80944874b0bbf6/adaptor/objectives/seq_bertscr_objectives.py#L326-L362
 [9]: https://arxiv.org/abs/1912.01412v1
