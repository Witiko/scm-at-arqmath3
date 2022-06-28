# Soft Cosine Measure at ARQMath-3

This repository contains our math information retrieval (MIR) system for
[the ARQMath3 competition][1] that is based on [the soft cosine measure][2].
The repository also contains the paper that describes our system.

 [1]: https://www.cs.rit.edu/~dprl/ARQMath/
 [2]: https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html

## Research goals

1. Compare the soft vector space model against sparse information retrieval
   baselines.
2. Compare performance of text, text + LaTeX, and text + Tangent-L as math
   representations
3. Compare performance of non-positional word2vec and positional `word2vec`
   embeddings
4. Compare performance of word2vec embeddings and decontextualized
   `roberta-base` embeddings
5. Compare performance of decontextualized embeddings of `roberta-base` and
   tuned `roberta-base`
6. Compare performance of interpolated and joint SCM models for text and math

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

- Draw conclusions from Figure 1:

    > the model improves but has not reached convergence

- Explain the purpose of out-of-domain evaluation in Figure 1:

    > this graph shows that improvements on math do not come at the price on
    > other domains outside math

- Optimize hyperparameters and update [`08-produce-arqmath-runs.ipynb`][15].
- After we have received task 1 annotations:
  - Add annotations to [pv211-utils][22] and [arqmath-eval][23] libraries.
  - Add MAP and nDCG' scores for 2022 to [`08-produce-arqmath-runs.ipynb`][15].
  - Vizualize the impact of various extensions (axis x) on nDCG' (axis y)
    in `09-evaluate-arqmath-runs.ipynb` and in section *Results* of the paper.
  - Add end-task evaluation on ARQMath-1, 2, and 3 topics to
    `09-evaluate-arqmath-runs.ipynb` and `10-evaluate-roberta.ipynb`.
  - Add the best systems from other teams to tables 1 and 2.
- Add [extrinsic end-task evaluation on NumGLUE][14] to
  [`03-finetune-roberta.ipynb`][7].
- After the ARQMath-3 paper has been published:
  - Add [a *Citing* section][18] to `README.md`.
  - Add [a link][19] and a BibTeX entry to [the 🤗 Model Hub][17].

## Citing

### Text

Vít Novotný and Michal Štefánik. “Combining Sparse and Dense Information
Retrieval. Soft Vector Space Model and MathBERTa at ARQMath-3 Task 1 (Answer
Retrieval)”. In: *Proceedings of the Working Notes of CLEF 2022*. To Appear.
CEUR-WS, 2022.

### Bib(La)TeX

``` bib
@inproceedings{novotny2022combining,
  booktitle = {Proceedings of the Working Notes of {CLEF} 2022},
  title = {Combining Sparse and Dense Information Retrieval},
  subtitle = {Soft Vector Space Model and MathBERTa at ARQMath-3 Task 1 (Answer Retrieval)},
  author = {Novotný, Vít and Štefánik, Michal},
  publisher = {{CEUR-WS}},
  year = {2022},
  note = {To Appear},
}
```

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
 [22]: https://github.com/MIR-MU/ARQMath-eval
 [23]: https://github.com/MIR-MU/pv211-utils
 [24]: https://easychair.org/conferences/submission?a=28850142;submission=6037102
 [25]: https://stackoverflow.com/a/64333567/657401
