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

- In the results sections of the paper, discuss the following:
  - The effectiveness of the baselines and how they are improved by extensions:
    - The *joint text (no token similarities)* baseline receives NDCG' of 0.235.
      Using `roberta-base` as the source of semantic similarity between text
      tokens improves effectiveness by 0.012 NDCG', reaching NDCG' of 0.247. By
      contrast, modeling also LaTeX math reduces effectiveness by 0.011 NDCG',
      reaching NDCG' of 0.224, which we attribute to the difficulty to properly
      represent the different frequency distributions of text and math tokens
      in a single joint model. When we instead use either positional word2vec
      or MathBERTa as the source of semantic similarity between text and math
      tokens improves effectiveness by 0.025 NDCG', reaching NDCG' of 0.249.
      Removing the positional weighting from word2vec further improves
      effectiveness by 0.002 NDCG', reaching NDCG' of 0.251: the best result
      among our joint models.
    - The *interpolated text + LaTeX (no token similarities)* baseline received
      NDCG' of 0.257. Using non-positional word2vec as the source of similarity
      between text and math tokens improves effectiveness by 0.031 NDCG',
      reaching NDCG' of 0.288. Enabling the positional weighting of word2vec
      does not further improve effectiveness.
    - The *interpolated text + Tangent-L (no token similarities)* baseline
      received NDCG' of 0.349. Using non-positional word2vec as the source of
      similarity between text and math tokens improves effectiveness by 0.002
      NDCG', reaching NDCG' of 0.251. Enabling the positional weighting of
      word2vec further improves effectiveness by 0.004 NDCG', reaching NDCG'
      of 0.355, the best result among all our models.
  - The effectiveness of our system compared to the best results from other
    teams on ARQMath-3 Task 1
    - Our submission the ARQMath-3 Task 1 with hand-picked parameter values
      placed last among the participating teams. However, our *interpolated
      text + Tangent-L (positional `word2vec`)* model with optimized parameter
      values placed above all systems from the DPLR team.
  - The robustness of our system to parameter variations:
    - The differences between hand-picked and optimized parameter values
      for joint models were within 0.002 NDCG' except *text (`roberta-base`)*,
      which improved effectiveness by 0.041 NDCG' score by placing more
      weight on the lexical similarity of tokens (α: 0.1 -> 0.6) and by
      placing less weight on question titles (γ: 5 -> 2).  This shows that
      the joint vector space models are relatively robust to parameter
      variations.
    - By contrast, optimizing parameter values for the *interpolated text +
      Tangent-L (positional `word2vec`)* model improved effectiveness by 0.098
      NDCG'. Compared to the hand-picked parameter values, the optimized
      parameter values place more weight for lexical similarity for text tokens
      (α₁: 0.1 -> 0.7), use only semantic similarity for math tokens (α₂: 0.1
      -> 0.0), place less weight on the text in question titles (γ₁: 5 -> 2),
      and place more weight on math over text (β: 0.5 -> 0.7).
  - The optimal parameter values of different models:
    - All joint models and the interpolated models for text place more weight
      on the lexical similarity of tokens (α and α₁ of either 0.6 or 0.7).
    - All joint and interpolated models for text placed equal weight on
      question titles (γ and γ₁ of 2). By contrast, all joint models for text
      and math and the interpolated models for math placed comparatively higher
      weight on the math in question titles (γ and γ₂ between 3 and 5). This
      indicates that math in question titles is more informative than text.
    - All interlolated models for LaTeX math only used the semantic similarity
      of tokens (α₂: 1.0). By constract, all interpolated models for Tangent-L
      math only used the semantic similarity of tokens (α₂: 0.0).
    - All interpolated models place more weight on text over math (β of either
      0.6 and 0.7).
- In the conclusion section of the paper, answer the research questions.
  1. Does the soft vector space model outperform sparse information retrieval
     baselines on the math information retrieval task?

     Yes, using the soft vector space model to capture the semantic
     similarity between tokens consistently improves effectiveness on
     ARQMath-3 Task 1, both for just text and for text combined with
     different math representations.
  2. Which math representation works best with the soft vector space model?

     Among LaTeX and Tangent-L, soft vector space models using Tangent-L
     achieve the highest effectiveness on ARQMath-3 Task 1.
  3. Which notion of similarity between key words and symbols works best?

     Among lexical and semantic similarity, all joint models and the
     interpolated models for text reach their highest effectiveness on
     ARQMath-3 Task 1 by combining both lexical and semantic similarity, but
     place slightly more weight on lexical similarity. The interpolated models
     for math gave mixed results: The model for Tangent-L reaches the highest
     efficiency by using only semantic similarity, whereas the model for
     LaTeX reaches the highest efficiency by using only lexical similarity.

     Among sources of semantic similarity, joint models achieved comparable
     effectiveness on ARQMath-3 Task 1 with non-positional word2vec, word2vec,
     and MathBERTa, and interpolated models achieved comparable effectiveness
     with non-positional word2vec and positional word2vec. This may indicate
     that the soft vector space model does not fully exploit the semantic
     information provided by the sources of semantic similarity and therefore
     does not benefit from their improvements after a certain threshold.

  4. Is it better to use a single soft vector space model to represent both
     text and math or to use two separate models?

     All our interpolated models achieved higher effectiveness on ARQMath-3
     Task 1 than our joint models. This shows that it is generally better
     to use two separate models to represent text and math even at the expense
     of losing the ability to model the similarity between text and math tokens.

- Write a related work section before conclusion to the paper discussing:
  - The ARQMath labs
  - The soft vector space model (see @witiko's [dissertation][26])
  - Log-bilinear language models (see @witiko's [dissertation][26])
  - Deep transformer models with a paragraph on math-aware models (see branch
    `related_work` by @stefanik12)
- Proofread the paper.
- After the ARQMath-3 paper has been published:
  - Update [the *Citing* section][18] in `README.md` and on [the 🤗 Model Hub][17].
  - Add [a link][19] to [the 🤗 Model Hub][17].
  - Update [the evaluation results][27] on [the 🤗 Model Hub][17].
  - Add [extrinsic end-task evaluation on NumGLUE][14] to
    [`03-finetune-roberta.ipynb`][7]. Plot performance on the five different
    NumGLUE tasks (axis y) over checkpoints (axis x).
  - Add end-task evaluation on ARQMath-3 topics to
    `09-evaluate-arqmath-runs.ipynb` and `10-evaluate-roberta.ipynb`.

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
 [26]: https://github.com/witiko/doctoral-thesis
 [27]: https://huggingface.co/witiko/mathberta#intrinsic-evaluation-results
