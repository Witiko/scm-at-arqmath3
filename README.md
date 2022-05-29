# Soft Cosine Measure at ARQMath3

This repository contains our math information retrieval (MIR) system for
[the ARQMath3 competition][1] that is based on [the soft cosine measure][2].
The repository also contains the paper that describes our system.

 [1]: https://www.cs.rit.edu/~dprl/ARQMath/
 [2]: https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html

## Research goals

1. Compare performance of text, text + LaTeX, and text + Tangent-L as math
   representations
2. Compare performance of non-positional word2vec and positional `word2vec`
   embeddings
3. Compare performance of word2vec embeddings and decontextualized
   `roberta-base` embeddings
4. Compare performance of decontextualized embeddings of `roberta-base` and
   tuned `roberta-base`
5. Compare performance of interpolated and joint SCM models for text and math

## Jupyter notebooks

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
  [the batched algorithm for averages by Matt Hancock][4].

## Artefacts

- [The `witiko/mathberta` model][7] at [the 🤗 Model Hub][17].

## Tasks

- Before submitting the paper for review:
  - [x] Introduce the shorthands *positional word2vec* and *non-positional
    word2vec* already in section *Language Modeling*.
  - [x] Describe weighted zone scoring in section *Soft Vector Space Modeling*.
  - [ ] Create section *Experiments* after section *Methods*.
  - [ ] Add subsections *Collection*, *Topics*, and *Relevance Judgements*
        to section *Experiments*.
  - [ ] Move subsection *Evaluation* to section *Experiments*.
  - [ ] Add subsection *Parameter optimization* at the end of section
    *Experiments*.
  - [ ] Write section *Results*.
  - [ ] Write abstract.
  - [ ] Write section *Introduction*.
  - [ ] Write section *Conclusion*.
  - [ ] Switch from pdfTeX to LuaTeX and prevent `interblockSeparator` before
    `contentBlock` from inserting a new paragraph.
  - [ ] Render `LaTeX` as `\LaTeX`.
- After gaussalgo/adaptor#21 has been closed, cherry-pick branch
  `feature/evaluate-tuned-romerta-base-ood` and add [out-of-domain
  evaluation][13] to `03-finetune-roberta.ipynb`.
- Add `%ls -lh submission*/run.tsv` for every run to
  [`08-produce-arqmath-runs.ipynb`][15].
- Plot learning rate in [`03-finetune-roberta.ipynb`][7].
- Add [extrinsic end-task evaluation on NumGLUE][14] to
  [`03-finetune-roberta.ipynb`][7].
- After we have received task 1 annotations:
  - Add MAP and nDCG' scores for 2022 to [`08-produce-arqmath-runs.ipynb`][15].
  - Vizualize the impact of various extensions (axis x) on nDCG' (axis y)
    with TikZ in section *Results* in the paper.
  - Add end-task evaluation on ARQMath-1, 2, and 3 topics to
    `09-evaluate-roberta.ipynb`.
- After the ARQMath-3 paper has been published:
  - Add [a *Citing* section][18] to `README.md`.
  - Add [a link][19] and a BibTeX entry to [the 🤗 Model Hub][17].
  - Add `10-optimize-hyperparameters`, where we optimize the hyperparameters
    $\alpha$, $\beta$, and $\gamma$ for all runs. Describe the optimization in
    subsection *Parameter optimization* of the paper.

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
