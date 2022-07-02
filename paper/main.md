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
subtitle: Soft Vector Space Model and MathBERTa at ARQMath-3 Task 1 (Answer Retrieval)
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

Sparse retrieval techniques can detect exact matches, but are inadequate for
mathematical texts, where the same information can be expressed as either text
or math. The soft vector space model has been shown to improve sparse
retrieval on semantic text similarity, text classification, and machine
translation evaluation tasks, but it has not yet been properly evaluated on
math information retrieval.

In our work, we compare the soft vector space model against standard sparse
retrieval baselines and state-of-the-art math information retrieval systems
from Task 1 (Answer Retrieval) of the ARQMath-3 lab. We evaluate the impact of
different math representations, different notions of similarity between key
words and math symbols ranging from Levenshtein distances to deep neural
language models, and different ways of combining text and math.

We show that using the soft vector space model consistently improves
effectiveness compared to using standard sparse retrieval techniques. We also
show that the Tangent-L math representation achieves better effectiveness than
LaTeX, and that modeling text and math separately using two models improves
effectiveness compared to jointly modeling text and math using a single model.
Lastly, we show that different math representations and different ways of
combining text and math benefit from different notions of similarity. Our best
system achieves NDCG' of 0.251 on Task 1 of the ARQMath-3 lab.

# Keywords {#keywords}

information retrieval
sparse retrieval
dense retrieval
soft vector space model
math representations
word embeddings
constrained positional weighting
decontextualization
word2vec
transformers

# Introduction

State-of-the-art math information retrieval systems use sparse retrieval
techniques that can detect exact key word matches with high precision, but
fail to retrieve texts that are semantically similar but use different
terminology. This shortcoming is all the more apparent with mathematical
texts, where the same information can be expressed in two completely different
systems of writing and thought: the natural language and the language of
mathematics.

Recently, the soft vector space model of @sidorov2014soft made it possible to
retrieve documents according to both exact and fuzzy key word matches and has
outperformed standard sparse retrieval techniques on semantic text similarity
[@charlet2017simbow], text classification [@novotny2020text], and machine
translation evaluation [@stefanik2021regemt] tasks. The soft vector space has
been used for math information retrieval in the ARQMath-1 and 2 labs
[@novotny2020three; @novotny2021ensembling]. However, it has not been properly
compared to sparse retrieval baselines. Furthermore, the soft vector space
model makes it possible to use different representations of math, different
notions of similarity between key words and symbols, and different ways to
combine text and math. However, neither of these possibilities has been
previously explored.

In our work, we aim to answer the following four research questions:

1. Does the soft vector space model outperform sparse information retrieval
   baselines on the math information retrieval task?
2. Which math representation works best with the soft vector space model?
3. Which notion of similarity between key words and symbols works best?
4. Is it better to use a single soft vector space model to represent both
   text and math or to use two separate models?

The rest of our paper is structured as follows: In Section~<#methods>, we
describe our system and our experimental setup. In Section~<#results>, we
report and discuss our experimental results. We conclude in
Section~<#conclusion> by answering our research questions and summarizing our
contributions.

# Methods {#methods}

In this section, we describe the datasets we used to train our tokenizers and
language models. We also describe how we used our language models to measure
similarity between text and math tokens, how we used our similarity measures to
find answers to math questions, and how we evaluated our system.

## Datasets

In our experiments, we used the Math StackExchange and ArXMLiv corpora:

Math StackExchange

:   The Math StackExchange collection v1.2 (MSE)[^mse1.3] provided by the
    organizers of the ARQMath-2 lab [@mansouri2021overview, Section 3] contains
    2,466,080 posts from the Math StackExchange question answering website in
    HTML5 with math formulae in LaTeX.

 [^mse1.3]: An improved Math Stack Exchange collection v1.3 was made available by
 the organizers of the ARQMath-3 lab [@mansouri2022overview, Section 3], which
 we did not use due to time constraints.

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
    and mathematical formulae in the LaTeX format surrounded by *[MATH]*
    and *[/MATH]* tags. To validate our language models, we used a small part
    of the `error` subset of ArXMLiv and no data from MSE.

    Example: *We denote the set of branches with [MATH] B\_{0},B\_{1},\ldots,B{n}
    [/MATH] where [MATH] n [/MATH] are the number of branches.*

Text

:   To train text language models, we used the same combinations of MSE
    and ArXMLiv as in the previous dataset, but now our dataset only contains
    text with math formulae removed.

    Example: *(Graphs of residually finite groups) Assume that and are
    satisfied. Let be a graph of groups. If is infinite then assume that is
    continuous.*

LaTeX

:   To train math language models, we used the same combinations of MSE
    and ArXMLiv subsets as in the previous datasets, but now our dataset
    only contains formulae in the LaTeX format.

    Example: *\begin{pmatrix}1&n\\0&1\end{pmatrix}\begin{pmatrix}1&p\\0&1\end{pmatrix}*

Tangent-L

:   To train math language models, we used the same combinations of MSE
    and ArXMLiv subsets as in the previous datasets, but now our dataset
    only contains formulae in the format used by [the state-of-the-art
    Tangent-L search engine from UWaterloo][mathtuples] [@ng2021dowsing].

    Example: *#(start)# #(v!△,/,n,-)# #(v!△,/,n)# #(/,v!l,n,n)# #(/,v!l,n)# #(v!l,!0,nn)# #(v!l,!0)# #(end)#*

 [01-prepare-dataset]: https://github.com/witiko/scm-at-arqmath3 (file 01-prepare-dataset.ipynb)
 [mathtuples]: https://github.com/fwtompa/mathtuples (git commit 888b3d5 from October 25, 2021)

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
  of `roberta-base` with the *[MATH]* and *[/MATH]* special tokens and with the
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
    constrained positional weighting for every dataset. For brevity, we refer
    to `word2vec` with and without constrained positional weighting as
    *positional `word2vec`* and *non-positional `word2vec`* in the rest of the
    paper.

Deep transformer models

:   To model text, we used [the pre-trained roberta-base model][roberta-base]
    [@liu2019roberta].

     ![learning-curves][]

    To model text and math in the LaTeX format, we replaced the tokenizer
    of `roberta-base` with our text and math tokenizer. Then, we extended the
    vocabulary of our model with the *[MATH]* and *[/MATH]* special tokens
    and with the tokens recognized by our LaTeX tokenizer, and we randomly
    initialized weights for the new tokens. Then, we fine-tuned our model on
    our text + LaTeX dataset for one epoch using [the masked language modeling
    objective of RoBERTa][03-finetune-roberta] [@liu2019roberta] and a learning
    rate of 10⁻⁵ with a linear decay to zero, see the learning curves in
    Figure~<#fig:learning-curves>. We called our model MathBERTa and
    [released it to the HuggingFace Model Hub.][mathberta]⸴[^mathberta-related-work]

 [03-finetune-roberta]: https://github.com/witiko/scm-at-arqmath3 (file 03-finetune-roberta.ipynb)
 [04-train-word2vec]: https://github.com/witiko/scm-at-arqmath3 (file 04-train-word2vec.ipynb)

 [^mathberta-related-work]: The task of *language modeling* is to predict a
    token of interest from the surrounding context. Specifically, in masked
    language modeling (MLM), the task is to predict a masked token from the
    surrounding, bidirectional context. @devlin2019bert demonstrate that
    MLM as the objective of a transformer-based encoder [@vaswani2017attention]
    can learn accurate, task-agnostic representations that can be fine-tuned to
    downstream tasks, reaching superior performance to previous approaches.
    @liu2019roberta further show the scalability of this approach, showing
    further gains by scaling the size of the unsupervised training set.

    Related work shows that more accurate domain-specialized representations can
    be obtained by continuous training, i.e. *adaptation* using MLM on
    domain-specific unlabeled texts, in medicine [@rasmy2021med], biology
    [@lee2020biobert], or, closer to our work, scientific texts
    [@beltagy2019scibert]. Previous works [@peng2021mathbertap;
    @shen2021mathbertap] perform continuous MLM training on
    scientific texts, or the math formulae thereof [@jo2021modeling]. However,
    all the mentioned works treat mathematical formulae as plain text and few
    works [@gong2022continual] promote math-specific representations in the
    model adaptation. This motivates us to experiment with adaptation
    incorporating specific encodings for non-textual expressions.

 [learning-curves]: learning-curves.pdf "Learning curves of MathBERTa on our text + LaTeX dataset (in-domain) and the European Constitution (out-of-domain). The ongoing descent of in-domain validation loss indicates that the performance of the model improved over time, but has not converged and would benefit from further training. The ongoing descent of out-of-domain validation loss shows that improvements on scientific texts do not come at the price of other non-scientific domains."
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
been shown to improve performance on the semantic text similarity task
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

In order to find answers to math questions, we used sparse retrieval with the soft
vector space model of @sidorov2014soft, using Lucene BM25 [@kamphuis2020bm25,
Table 1] as the vector space and our combined similarity matrices as the token
similarity measure. To address the bimodal nature of math questions and
answers, we [used the following two approaches:][08-produce-arqmath-runs]

Joint models

:   To allow users to query math information using natural language and
    vise versa, we used single soft vector space models to jointly represent
    both text and math.

    As our baselines, we used 1) Lucene BM25 with the text dictionary and no
    token similarities and 2) Lucene BM25 with the text + LaTeX dictionary
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

For our representation of questions in the soft vector space model, we used the
tokens in the title and in the body text. To represent an answer in the soft
vector space model, we used the tokens in the title of its parent question and
in the body text of the answer. To give greater weight to tokens in the title,
we repeated them γ times, which proved useful in ARQMath-2
[@novotny2021ensembling, Section 3.2].

 [08-produce-arqmath-runs]: https://github.com/witiko/scm-at-arqmath3 (file 08-produce-arqmath-runs.ipynb)

## Evaluation

To evaluate our system, we searched for answers to sets of topics provided by
the ARQMath organizers [@zanibbi2020overview; @mansouri2021overview;
@mansouri2022overview, Section 4.1]. As our retrieval units, we used answers
from the MSE dataset.

Effectiveness

:   To determine how well the answers retrieved by our system satisfied the
    information needs of users, we used the normalized discounted cumulative
    gain prime (NDCG') evaluation measure [@sakai2008information] on the top
    1,000 answers retrieved by our system for each topic. As our ground truth,
    we used the relevance judgements provided by the ARQMath organizers
    [@zanibbi2020overview; @mansouri2021overview; @mansouri2022overview, Section
    4.3].

    To select the optimal values for parameters α, β, and γ, we used the 148
    topics from ARQMath-1 and 2, Task 1 and performed a grid search over values
    α ∈ {0.0, 0.1, ..., 1.0}, β ∈ {0.0, 0.1, ..., 1.0}, and γ ∈ {1, 2, 3, 4,
    5}.[^optimization] To estimate the effectiveness of our system, we used the
    78 topics from ARQMath-3 Task 1.

    Due to time constraints, we hand-picked the parameter values
    α = 0.1, β = 0.5, and γ = 5 for our submissions to the ARQMath-3 lab. We
    report effectiveness for both hand-picked and optimal parameter values, and
    discuss the robustness of our system to parameter variations.

Efficiency

:   Our system is a prototype written in a high-level programming language with
    emphasis on correctness over efficiency. Furthermore, we computed our
    evaluation on a non-dedicated computer cluster with heterogeneous hardware,
    which made it difficult to meaningfully measure the efficiency of our system.
    Therefore, we have not measured and do not report the efficiency of our system.

# Results {#results}

In tables 1 and 2, we list effectiveness results with hand-picked parameter
values submitted to the ARQMath-3 lab for our joint and interpolated soft
vector space models. In tables 3 and 4, we list post-competition effectiveness
results with optimized parameter values for our joint and interpolated models.
In all tables 1--4, we also list the parameter values the we used.
In Figure 2, we visualize the effectiveness of our baseline models with
optimized parameter values and how it is affected by our various extensions.
In Table 5, we compare our post-competition effectiveness results with the
optimized parameter values to the baselines and the best results from other
teams on ARQMath-3 Task 1.

## Robustness to Parameter Variations

In tables 1--4, the differences between hand-picked and optimized parameter
values for joint models were within 0.002 NDCG' except *Joint text
(`roberta-base`)*, which improves effectiveness by 0.041 NDCG' by placing
more weight on the lexical similarity of tokens (α: 0.1→0.6) and by placing
less weight on question titles (γ: 5→2).  This shows that our joint vector
space models are relatively robust to parameter variations.

By contrast, optimizing parameter values for the *Interpolated text +
Tangent-L (positional `word2vec`)* model improves effectiveness by 0.098
NDCG'. Compared to the hand-picked parameter values, the optimized parameter
values place more weight at lexical similarity for text tokens (α₁: 0.1→0.7),
use only semantic similarity for math tokens (α₂: 0.1→0.0), place less weight
on the text in question titles (γ₁: 5→2), and place more weight on math over
text (β: 0.5→0.7).

| Model | α | γ | NDCG' |
|-------|---|---|-------|
| Joint text + LaTeX (MathBERTa)                 | 0.1 | 5 | 0.249 |
| Joint text + LaTeX (non-positional `word2vec`) | 0.1 | 5 | 0.249 |
| Joint text + LaTeX (positional `word2vec`)     | 0.1 | 5 | 0.248 |
| Joint text (`roberta-base`)                    | 0.1 | 5 | 0.188 |

: Results with hand-picked parameter values submitted to the ARQMath-3 lab for joint soft vector space models on ARQMath-3 Task 1 topics

| Model | α₁ | γ₁ | α₂ | γ₂ | β | NDCG' |
|-------|----|----|----|----|---|-------|
| Interpolated text + Tangent-L (positional `word2vec`)     | 0.1 | 5 | 0.1 | 5 | 0.5 | 0.257 |

: Results with hand-picked parameter values submitted to the ARQMath-3 lab for interpolated soft vector space models on ARQMath-3 Task 1 topics

## Effectiveness of Baselines and Their Extensions

Figure 2 shows that the *Joint text (no token similarities)* baseline receives
NDCG' of 0.235.  Using `roberta-base` as the source of semantic similarity
between text tokens improves effectiveness by 0.012 NDCG', reaching NDCG' of
0.247. By contrast, modeling also LaTeX math reduces effectiveness by 0.011
NDCG', reaching NDCG' of 0.224, which we attribute to the difficulty to
properly represent the different frequency distributions of text and math
tokens in a single joint model. However, when we also use either positional
word2vec or MathBERTa as the source of semantic similarity between text and
math tokens, effectiveness improves by 0.025 NDCG', reaching NDCG' of 0.249.
Removing the positional weighting from word2vec further improves effectiveness
by 0.002 NDCG', reaching NDCG' of 0.251, which is the best result among our
joint models.

Figure 2 also shows that the *Interpolated text + LaTeX (no token
similarities)* baseline received NDCG' of 0.257. Using non-positional word2vec
as the source of similarity between text and math tokens improves effectiveness
by 0.031 NDCG', reaching NDCG' of 0.288. Enabling the positional weighting of
word2vec does not further improve effectiveness.

The *Interpolated text + Tangent-L (no token similarities)* baseline received
NDCG' of 0.349. Using non-positional word2vec as the source of similarity
between text and math tokens improves effectiveness by 0.002 NDCG', reaching
NDCG' of 0.251. Enabling the positional weighting of word2vec further improves
effectiveness by 0.004 NDCG', reaching NDCG' of 0.355, the best result among
all our models.

## Optimized Parameter Values

Tables 3 and 4 show that all joint models and the interpolated models for text
place more weight on the lexical similarity of tokens (α and α₁ of either 0.6
or 0.7).

Furthermore, all joint and interpolated models for text place equal
weight on question titles (γ and γ₁ of 2). By contrast, all joint models for
text and math and the interpolated models for math place comparatively higher
weight on the math in question titles (γ and γ₂ between 3 and 5). This
indicates that math in question titles is more informative than text.

Lastly, all interpolated models for LaTeX math only used the semantic
similarity of tokens (α₂: 1.0). By contract, all interpolated models for
Tangent-L math only used the semantic similarity of tokens (α₂: 0.0).  All
interpolated models place more weight on text over math (β of either 0.6 or
0.7).

| Model | α | γ | NDCG' |
|-------|---|---|-------|
| Joint text + LaTeX (non-positional `word2vec`) | 0.6 | 5 | 0.251 |
| Joint text + LaTeX (positional `word2vec`)     | 0.7 | 5 | 0.249 |
| Joint text + LaTeX (MathBERTa)                 | 0.6 | 4 | 0.249 |
| Joint text (`roberta-base`)                    | 0.6 | 2 | 0.247 |
| Joint text (no token similarities)             |     | 2 | 0.235 |
| Joint text + LaTeX (no token similarities)     |     | 3 | 0.224 |

: Post-competition results with optimized parameter values for joint soft vector space models on ARQMath-3 Task 1 topics

| Model | α₁ | γ₁ | α₂ | γ₂ | β | NDCG' |
|-------|----|----|----|----|---|-------|
| Interpolated text + Tangent-L (positional `word2vec`)     | 0.7 | 2 | 0.0 | 5 | 0.7 | 0.355 |
| Interpolated text + Tangent-L (non-positional `word2vec`) | 0.6 | 2 | 0.0 | 5 | 0.7 | 0.351 |
| Interpolated text + Tangent-L (no token similarities)     |     | 2 |     | 4 | 0.6 | 0.349 |
| Interpolated text + LaTeX (positional `word2vec`)         | 0.7 | 2 | 1.0 | 5 | 0.6 | 0.288 |
| Interpolated text + LaTeX (non-positional `word2vec`)     | 0.6 | 2 | 1.0 | 5 | 0.6 | 0.288 |
| Interpolated text + LaTeX (no token similarities)         |     | 2 |     | 5 | 0.6 | 0.257 |

: Post-competition results with optimized parameter values for interpolated soft vector space models on ARQMath-3 Task 1 topics

 /visualization-of-extensions.tex (The extensions of the baseline soft vector space models and their impact on the effectiveness with optimized parameter values)

| Model | NDCG' |
|-------|-------|
| *fusion\_alpha05 from approach0* [@zhong2022applying]            | 0.508 |
| *Ensemble\_RRF from MSM* [@geletka2022diverse]                   | 0.504 |
| *MiniLM+RoBERTa from MIRMU* [@geletka2022diverse]                | 0.498 |
| *L8\_a018 from MathDowsers* [@kane2022dowsing]                   | 0.474 |
| *math\_10  from TU\_DBS* [@reusch2022transformer]                | 0.436 |
| Interpolated text + Tangent-L (positional `word2vec`)            | 0.355 |
| Interpolated text + Tangent-L (non-positional `word2vec`)        | 0.351 |
| Interpolated text + Tangent-L (no token similarities)            | 0.349 |
| Interpolated text + LaTeX (positional `word2vec`)                | 0.288 |
| Interpolated text + LaTeX (non-positional `word2vec`)            | 0.288 |
| *SVM-Rank from DPRL* [@mansouri2022introducing]                  | 0.283 |
| *TF-IDF (Terrier) baseline* [@mansouri2022overview]              | 0.272 |
| Interpolated text + LaTeX (no token similarities)                | 0.257 |
| Joint text + LaTeX (non-positional `word2vec`)                   | 0.251 |
| Joint text + LaTeX (positional `word2vec`)                       | 0.249 |
| Joint text + LaTeX (MathBERTa)                                   | 0.249 |
| Joint text (`roberta-base`)                                      | 0.247 |
| Joint text (no token similarities)                               | 0.235 |
| *TF-IDF (PyTerrier) + TangentS baseline* [@mansouri2022overview] | 0.229 |
| Joint Text + LaTeX (no token similarities)                       | 0.224 |
| *TF-IDF (PyTerrier) baseline* [@mansouri2022overview]            | 0.190 |
| *Tangent-S baseline* [@mansouri2022overview]                     | 0.159 |
| *Linked MSE Posts baseline* [@mansouri2022overview]              | 0.106 |

: Comparison of our post-competition effectiveness results to the baselines and the best results from other teams on ARQMath-3 Task 1

## Comparison to Results from Other Teams

Our submission the ARQMath-3 lab with hand-picked parameter values
placed last in effectiveness among the teams that participated in Task 1.
However Table 5 shows that our *Interpolated text + Tangent-L (positional
`word2vec`)* model with optimized parameter values achieved better
effectiveness than the best system from the DPRL team
[@mansouri2022introducing] by 0.011 NDCG'.

# Conclusion {#conclusion}

In this paper, we aimed to answer the following research questions:

1. Does the soft vector space model outperform sparse information retrieval
   baselines on the math information retrieval task?
2. Which math representation works best with the soft vector space model?
3. Which notion of similarity between key words and symbols works best?
4. Is it better to use a single soft vector space model to represent both
   text and math or to use two separate models?

Using our experimental results, we can answer our research questions as follows:

1. Yes, using the soft vector space model to capture the semantic
   similarity between tokens consistently improves effectiveness on
   ARQMath-3 Task 1, both for just text and for text combined with
   different math representations.

2. Among LaTeX and Tangent-L, soft vector space models using Tangent-L achieve
   the highest effectiveness on ARQMath-3 Task 1.

3. Among lexical and semantic similarity, all joint models and the
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

4. All our interpolated models achieved higher effectiveness on ARQMath-3
   Task 1 than our joint models. This shows that it is generally better
   to use two separate models to represent text and math even at the expense
   of losing the ability to model the similarity between text and math tokens.

Answers to research questions 2 and 3 also provide the following new questions:

2. Are there other math representations besides LaTeX and Tangent-L that
   may work better with the soft vector space model?

3. How can the soft vector space model be improved, so that it can benefit from
   improved measures of similarity between tokens?

← These questions should be answered by future work.
