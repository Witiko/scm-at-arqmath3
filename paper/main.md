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
subtitle: Soft Vector Space Model and MathBERTa at ARQMath-3
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
constrained positional weighting
decontextualization
word2vec
transformers

# Introduction

Some text. [@novotny2021interpretable]

# Methods

In this section, we describe the datasets we used to train our tokenizers and
language models. We also describe how we used our language models to measure
similarity between text and math tokens, how we used our similarity measures to
find answers to math questions, and how we evaluated our system.

## Datasets

In our experiments, we used the Math StackExchange and ArXMLiv corpora:

Math StackExchange

:   The Math StackExchange collection v1.2 (MSE) provided by the organizers of
    the ARQMath-2 shared task evaluation [@behrooz2021overview] contains
    2,466,080 posts from the Math Stack Exchange question answering website in
    HTML5 with math formulae in LaTeX.

ArXMLiv

:   The ArXMLiv 2020 corpus [@ginev2020arxmliv] contains 1,571,037 scientific
    preprints from ArXiv in the HTML5 format with math formulae in MathML.
    Documents in the dataset were converted from LaTeX sources and are divided
    into the following subsets according to the severity of errors encountered
    during conversion: `no-problem`~(10%), `warning`~(60%), and `error`~(30%).

From the corpora, we [produced a number of datasets][01-prepare-dataset]
in different formats that we used to train our tokenizers and language models:

Text + LaTeX

:   To train text & math language models, we combined MSE with the
    `no-problem` and `warning` subsets of ArXMLiv. The dataset contains text
    and mathematical formulae in the LaTeX format surrounded by `[MATH]`
    and `[/MATH]` tags. To validate our language models, we used a small part
    of the `error` subset of ArXMLiv and no data from MSE.

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
    - To tokenize Tangent-L, we strip leading and trailing hash signs (`#`) from
      a formula representation and then split it into tokens using the `#\s+#`
      Perl regex.
- To tokenize text and math in the LaTeX format, we extended the BPE tokenizer
  of `roberta-base` with the `[MATH]` and `[/MATH]` special tokens and with the
  tokens recognized by our LaTeX tokenizer.

 [roberta-base]: https://huggingface.co/roberta-base
 [02-train-tokenizers]: https://github.com/witiko/scm-at-arqmath3 (file 02-train-tokenizers.ipynb)
 [mathberta]: https://huggingface.co/witiko/mathberta

## Language Modeling

In our experiments, we used two different types of language models:

Shallow log-bilinear models

:   We [trained the shallow `word2vec` language models][04-train-word2vec]
    [@mikolov2013distributed] on our text + LaTeX, text, LaTeX, and Tangent-L
    datasets.

    On text documents, a technique known as *constrained positional weighting*
    has been shown to improve the performance of `word2vec` models on
    analogical reasoning and causal language modeling [@novotny2022when].
    To evaluate the impact of constrained positional weighting on math
    information retrieval, we trained `word2vec` models both with and without
    constrained positional weighting for every dataset. In the rest of the
    text, we refer to `word2vec` with and without constrained positional
    weighting as *positional `word2vec`* and *non-positional `word2vec`* for
    brevity.

Deep transformer models

:   To model text, we used [the pre-trained roberta-base model][roberta-base].

     ![learning-curves][]

    To model text and math in the LaTeX format, we replaced the tokenizer
    of `roberta-base` with our text and math tokenizer. Then, we extended the
    vocabulary of our model with the `[MATH]` and `[/MATH]` special tokens
    and with the tokens recognized by our LaTeX tokenizer, and we randomly
    initialized weights for the new tokens. Then, we fine-tuned our model on
    our text + LaTeX dataset for one epoch using [the masked language modeling
    objective of RoBERTa][03-finetune-roberta] [@liu2019roberta], see learning
    curves in Figure~<#fig:learning-curves>. We called our model MathBERTa and
    [released it to the HuggingFace Model Hub.][mathberta]

 [03-finetune-roberta]: https://github.com/witiko/scm-at-arqmath3 (file 03-finetune-roberta.ipynb)
 [04-train-word2vec]: https://github.com/witiko/scm-at-arqmath3 (file 04-train-word2vec.ipynb)
 [learning-curves]: learning-curves.pdf "Learning curves of MathBERTa on our text + LaTeX dataset"
 [mathberta]: https://huggingface.co/witiko/mathberta

## Token Similarity

To determine the similarity of text and math tokens, we first extracted their
global representations from our language models:

Shallow log-bilinear models

:   We extracted token vectors from the input and output matrices of our
    `word2vec` models and averaged them to produce global token embeddings.

Deep transformer models

:   Unlike `word2vec`, transformer models do not contain global representations
    of tokens, but produce representations of tokens in the context of
    a sentence. To extract global token embeddings from `roberta-base` and
    MathBERTa, we [decontextualized their contextual token
    embeddings][05-produce-decontextualized-word-embeddings] [@stefanik2021regemt,
    Section 3.2] on the sentences from our text + LaTeX dataset.

Then, we [produced dictionaries of all tokens in our text + LaTeX, text, LaTeX,
and Tangent-L datasets,][06-produce-dictionaries] removing all tokens that
occurred less than twice in a dataset and keeping only 100,000 most frequent
tokens from every dataset. For each dictionary, we [produced two types of token
similarity matrices][07-produce-term-similarity-matrices] that capture the
surface-level lexical similarity and the semantic similarity between tokens,
respectively:

Lexical similarity

:   We used the method of @charlet2017simbow [Section 2.2] to produce
    similarity matrices using the edit distance between the tokens.

Semantic similarity

:   We used the method of @charlet2017simbow [Section 2.1] to produce
    similarity matrices using the cosine similarity between the global
    token embeddings.

    For all dictionaries, we produced two matrices using the token embeddings
    of the positional and non-positional `word2vec` models.  For the text and
    text + LaTeX dictionaries, we also produced an additional matrix using the
    token embeddings of the `roberta-base` and MathBERTa models, respectively.

To ensure sparsity and symmetry of the matrices, we considered only the 100
most similar tokens for each token and we used the greedy algorithm of
@novotny2018implementation [Section 3] to construct the matrices. For semantic
similarity matrices, we also enforced strict diagonal dominance, which has
been shown to improve performance on semantic text similarity
[@novotny2020text, Table 2].

Finally, to produce token similarity matrices that capture both lexical and
semantic similarity between tokens, we combined every semantic similarity
matrix with a corresponding lexical similarity matrix as follows:

 /combine_similarity_matrices.tex

In our system, we only used the combined token similarity matrices.

 [05-produce-decontextualized-word-embeddings]: https://github.com/witiko/scm-at-arqmath3 (file 05-produce-decontextualized-word-embeddings.ipynb)
 [06-produce-dictionaries]: https://github.com/witiko/scm-at-arqmath3 (file 06-produce-dictionaries.ipynb)
 [07-produce-term-similarity-matrices]: https://github.com/witiko/scm-at-arqmath3 (file 07-produce-term-similarity-matrices.ipynb)

## Soft Vector Space Modeling

To find answers to math questions, we used sparse retrieval with the soft
vector space model of @sidorov2014soft, using Lucene BM25 [@kamphuis2020bm25,
Table 1] as the vector space and our combined similarity matrices as the token
similarity measure. To address the bimodal nature of math questions and
answers, we [used the following two approaches:][08-produce-arqmath-runs]

Joint models

:   To allow users to query math information using natural language and
    vise versa, we used single soft vector space models to jointly represent
    both text and math.

    As our baseline, we used Lucene BM25 with the text + LaTeX dictionary
    and no token similarities.
    
    We also used four soft vector space models with the text + LaTeX dictionary
    and the token similarity matrices from the positional and non-positional
    `word2vec` models, the `roberta-base` model, and the MathBERTa model.

Interpolated models

:   To properly represent the different frequency distributions of text and
    math tokens, we used separate soft vector space models for text and math.
    The final score of an answer is determined by linear interpolation of the
    scores assigned by the two soft vector space models:

     /interpolate_similarity_scores.tex

    As our baselines, we used Lucene BM25 with the text dictionary and no token
    similarities interpolated with 1) Lucene BM25 with the LaTeX dictionary
    and no token similarities and with 2) Lucene BM25 with the Tangent-L
    dictionary and no token similarities.

    We also used four pairs of soft vector space models: two pairs with the
    text and LaTeX dictionaries and two pairs with the text and Tangent-L
    dictionaries. In each of the two pairs, one used the token similarity
    matrices from the positional `word2vec` model and the other used the
    token similarity matrices from non-positional `word2vec` model.

To represent a question in the soft vector space model, we used the tokens in
the title and in the body text. To represent an answer in the soft vector space
model, we used the tokens in the title of its parent question and in the body
text of the answer. To give greater weight to tokens in the title, we repeated
them γ times, which proved useful in ARQMath-2 [@novotny2021ensembling, Section
3.2].

 [08-produce-arqmath-runs]: https://github.com/witiko/scm-at-arqmath3 (file 08-produce-arqmath-runs.ipynb)

## Evaluation

We searched for answers to sets of topics provided by the ARQMath organizers.
To select the optimal values for parameters α, β, and γ,[^optimization] we used
177 topics from ARQMath-1 and 2. To estimate the performance of our system, we
used 100 topics from ARQMath-3. As our retrieval units, we used answers from
the MSE dataset.

To determine how well the answers retrieved by our system satisfied the
information needs of users, we used the relevance judgements provided
by the ARQMath organizers and the normalized discounted cumulative gain
prime (NDCG') evaluation measure [@sakai2008information] on the top 1,000
answers retrieved by our system for each topic.

 [^optimization]: Due to time constraints, we hand-picked the values α = 0.1,
 β = 0.5, and γ = 5 for our ARQMath-3 submissions. In the camera-ready, we will
 report the optimal parameter values and their NDCG' scores on ARQMath-3 topics.

# Results

# Conclusion
