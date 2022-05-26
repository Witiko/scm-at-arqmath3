# Soft Cosine Measure at ARQMath3

This repository contains our math information retrieval (MIR) system for
[the ARQMath3 competition][1] that is based on [the soft cosine measure][2].

 [1]: https://www.cs.rit.edu/~dprl/ARQMath/
 [2]: https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html

## Goals

- Compare performance of text, text + LaTeX, and text + Tangent-L as math
  representations
- Compare performance of non-positional word2vec and positional `word2vec`
  embeddings
- Compare performance of word2vec embeddings and decontextualized
  `roberta-base` embeddings
- Compare performance of decontextualized embeddings of `roberta-base` and
  tuned `roberta-base`
- Compare performance of interpolated and joint SCM models for text and math

## Tasks

1. [Prepare dataset][3]
2. [Train tokenizer][6]
3. [Tune `roberta-base` model][7]
4. [Train `word2vec` models][8]
5. [Produce decontextualized word embeddings][10]
6. [Produce dictionaries][11]
7. [Produce term similarity matrices][12]
8. [Produce ARQMath runs][15]

## Code pearls

- [Accelerated word embedding decontextualization][16] using
  [the batched online algorithm for moving averages by Matt Hancock][4].

## Artefacts

- [The `witiko/mathberta` model][7] at [the 🤗 Model Hub][17].

## Future work

- After finishing the training of `tuned-roberta-base-text+latex`:
  - Add [out-of-domain evaluation][13] to `03-finetune-roberta.ipynb`.
  - Add [end-task evaluation on NumGLUE][14] for `roberta-base` and
    `tuned-roberta-base-text+latex` to `03-finetune-roberta.ipynb`.
  - Add end-task evaluation on ARQMath-1 and 2 topics to `09-evaluate-roberta.ipynb`.
- After we have received task 1 annotations:
  - Add end-task evaluation on ARQMath-3 topics to `09-evaluate-roberta.ipynb`.
- After the ARQMath-3 paper has been published:
  - Add [a *Citing* section][18] to `README.md`.
  - Add [a link][19] and a BibTeX entry to [the 🤗 Model Hub][17].

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
 [13]: https://opus.nlpl.eu/EUconst.php
 [14]: http://allenai.org/data/numglue
 [15]: 08-produce-arqmath-runs.ipynb
 [16]: https://github.com/Witiko/scm-at-arqmath3/blob/d43cdced1bfd15754b4ca54291cf94b097b93068/scm_at_arqmath3/extract_decontextualized_word_embeddings.py#L104-L141
 [17]: https://huggingface.co/witiko/mathberta
 [18]: https://github.com/MIR-MU/WebMIaS#citing-webmias
 [19]: https://huggingface.co/roberta-base#roberta-base-model
 [20]: https://huggingface.co/roberta-base#bibtex-entry-and-citation-info
 [21]: https://huggingface.co/roberta-base#how-to-use
