# Soft Cosine Measure at ARQMath-3

This repository contains our math information retrieval (MIR) system for
[the ARQMath3 competition][1] that is based on [the soft cosine measure][2].
The repository also contains [the paper][29] that describes our system.

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
9. [Optimize soft vector space similarity matrices][28]

## Code pearls

- [Accelerated word embedding decontextualization][16] using
  [the batched algorithm for averages by Matt Hancock][4].

## Artefacts

- [Online demo of our system][30] using [the Document Maps visualization tool][31].
- [Our `witiko/mathberta` model][7] at [the 🤗 Model Hub][17].
- [Our paper from CLEF 2022][29] that describes our system.
- [Our presentation from CLEF 2022][32] that describes our system.

## Future work

- Add [extrinsic end-task evaluation on NumGLUE][14] to
  [`03-finetune-roberta.ipynb`][7]. Plot performance on the five different
  NumGLUE tasks (axis y) over checkpoints (axis x).
- Add end-task evaluation on ARQMath-3 topics over checkpoints to
  `08-produce-arqmath-runs.ipynb`.

## Citing

### Text

Vít Novotný and Michal Štefánik.
“Combining Sparse and Dense Information Retrieval. Soft Vector Space Model and
MathBERTa at ARQMath-3 Task 1 (Answer Retrieval)”.
In: *Proceedings of the Working Notes of CLEF 2022*.
Ed. by Guglielmo Faggioli, Nicola Ferro, Allan Hanbury, and Martin Potthast.
CEUR-WS, 2022, pp. 104–118.
URL: <http://ceur-ws.org/Vol-3180/paper-06.pdf> (visited on 08/12/2022).

### Bib(La)TeX

``` bib
@inproceedings{novotny2022combining,
  booktitle = {Proceedings of the Working Notes of {CLEF} 2022},
  editor = {Faggioli, Guglielmo and Ferro, Nicola and Hanbury, Allan and Potthast, Martin},
  issn = {1613-0073},
  title = {Combining Sparse and Dense Information Retrieval},
  subtitle = {Soft Vector Space Model and MathBERTa at ARQMath-3 Task 1 (Answer Retrieval)},
  author = {Novotný, Vít and Štefánik, Michal},
  publisher = {{CEUR-WS}},
  year = {2022},
  pages = {104-118},
  numpages = {15},
  url = {http://ceur-ws.org/Vol-3180/paper-06.pdf},
  urldate = {2022-08-12},
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
 [16]: https://github.com/Witiko/scm-at-arqmath3/blob/5585ddadf0d911e24c091bfe4c74511dbc24fdec/system/extract_decontextualized_word_embeddings.py#L104-L141
 [17]: https://huggingface.co/witiko/mathberta
 [18]: https://github.com/MIR-MU/WebMIaS#citing-webmias
 [19]: https://huggingface.co/roberta-base#roberta-base-model
 [20]: https://huggingface.co/roberta-base#bibtex-entry-and-citation-info
 [21]: https://huggingface.co/roberta-base#how-to-use
 [22]: https://github.com/MIR-MU/ARQMath-eval
 [23]: https://github.com/MIR-MU/pv211-utils
 [24]: https://easychair.org/conferences/submission?a=28850142;submission=6037102
 [25]: https://stackoverflow.com/a/64333567/657401
 [26]: https://github.com/witiko/doctoral-thesis
 [27]: https://huggingface.co/witiko/mathberta#intrinsic-evaluation-results
 [28]: 09-optimize-term-similarity-matrices.ipynb
 [29]: http://ceur-ws.org/Vol-3180/paper-06.pdf
 [30]: https://witiko.github.io/scm-at-arqmath3/
 [31]: https://github.com/MIR-MU/document-maps-visualization
 [32]: https://github.com/witiko/scm-at-arqmath3/releases/download/latest/slides.pdf
