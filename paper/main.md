---
copyrightyear: 2022
copyrightclause: |
  Copyright for this paper by its authors.
  Use permitted under Creative Commons License Attribution 4.0
  International (CC BY 4.0).
conference: |
  CLEF 2022: Conference and Labs of the Evaluation Forum,
  September 5--8, 2022, Bologna, Italy

title: Combining Sparse and Dense Information Retrieval
subtitle: Soft Cosine Measure and MathBERTa at ARQMath-3
author:
- name: Vít Novotný
  orcid: 0000-0002-3303-4130
  email: witiko@mail.muni.cz
- name: Michal Štefánik
  orcid: 0000-0003-1766-5538
  email: stefanik.m@mail.muni.cz
address: |
  Faculty of Informatics Masaryk University,
  Botanická 554/68a,
  602 00 Brno,
  Czech Republic

---

# Abstract {#abstract}

Background

Aims

Methods

Conclusion

# Keywords {#keywords}

Information retrieval
sparse retrieval
dense retrieval
math representations
word embeddings
positional weighting
decontextualization
word2vec
transformers

# Introduction

Some text. [@novotny2021interpretable]

# Methods

In this section, we describe the datasets we used to train our tokenizers and
language models. We also describe how we used our language models to measure
similarity between text and math tokens, and how we used our similarity
measures to find answers to math questions in task 1 of ARQMath-3.

## Datasets

In our experiments, we used the Math StackExchange and ArXMLiv corpora:

Math StackExchange

:   The Math StackExchange collection v1.2 (MSE) provided by the organizers of
    the ARQMath-2 shared task evaluation [@behrooz2021overview] contains
    2,466,080 posts from the Math Stack Exchange question answering website in
    HTML5 with math formulae in LaTeX. Besides the answers~(59%), which are the
    retrieval unit in task 1 of ARQMath-3, the posts also contain
    questions~(41%) related to the answers.

ArXMLiv

:   The ArXMLiv 2020 corpus [@ginev2020arxmliv] contains 1,571,037 scientific
    preprints from ArXiv in the HTML5 format with math formulae in MathML.
    Documents in the dataset were converted from LaTeX sources and are divided
    into the following subsets according to the severity of errors encountered
    during conversion: `no-problem`~(11%), `warning`~(60%), and `error`~(29%).

From the corpora, we [produced a number of datasets][01-prepare-dataset]
in different formats that we used to train our tokenizers and language models:

Text + LaTeX

:   To train text & math language models, we combined MSE with the
    `no-problem` and `warning` subsets of ArXMLiv. The dataset contains text
    and mathematical formulae in the LaTeX format surrounded by the `[MATH]`
    and `[/MATH]` tags. To validate our language models, we used a small part
    of the `error` subset of ArXMLiv and no data from MSE. To produce global
    representations of tokens by a process known as *decontextualization*, we
    used just the `no-problem` subset of ArXMLiv and no data from MSE.

Text

:   To train text language models, we used the same combinations of MSE
    and ArXMLiv as in the previous dataset, but now our dataset only contains
    text with math formulae removed.

LaTeX

:   To train math language models, we used the same combinations of MSE
    and ArXMLiv subsets as in the previous datasets, but now our dataset
    only contains formulae in the LaTeX format.

Tangent-L

:   To train math language models, we used the same combinations of MSE
    and ArXMLiv subsets as in the previous datasets, but now our dataset
    only contains formulae in the format used by [the state-of-the-art
    Tangent-L search engine from UWaterloo][mathtuples] [@ng2021dowsing].

 [01-prepare-dataset]: https://github.com/witiko/scm-at-arqmath3 (file 01-prepare-dataset.ipynb)
 [mathtuples]: https://github.com/fwtompa/mathtuples

## Tokenization

In our system, we used several tokenizers:

- To tokenize text, we used the BPE tokenizer of [the `roberta-base` language
  model][roberta-base] [@liu2019roberta].
- To tokenize math, we used two different tokenizers for the LaTeX and
  Tangent-L formats:
    - To tokenize LaTeX, we [trained a BPE tokenizer][02-train-tokenizers] with
      a vocabulary of size 50,000 on our LaTeX dataset.
    - To tokenize Tangent-L, we strip leading and training hash signs (`#`) from
      a formula representation and then split it to tokens using the `#\s+#` regex.
- To tokenize text and math in the LaTeX format, we extended the BPE tokenizer
  of `roberta-base` with the `[MATH]` and `[/MATH]` special tokens and with the
  tokens recognized by our LaTeX tokenizer.

 [roberta-base]: https://huggingface.co/roberta-base
 [02-train-tokenizers]: https://github.com/witiko/scm-at-arqmath3 (file 02-train-tokenizers.ipynb)
 [mathberta]: https://huggingface.co/witiko/mathberta

## Language Modeling

## Token Similarity

## Soft Vector Space Modeling

## Evaluation Measures

# Results

# Conclusion
