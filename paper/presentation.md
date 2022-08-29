Good afternoon, I am Vítek Novotný. You will recall that I have already spoken
in the previous session, where I described Task 3. However, besides helping to
organize Task 3, I have also participated in Task 1 together with my colleague
Michal from the Masaryk University in Brno, Czech Republic. In this talk, I
will describe our Task 1 submission.

# Introduction {#introduction}

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

In the rest of this talk, I will describe our system and our experimental setup,
report and discuss our experimental results, and conclude by answering our
research questions and summarizing our contributions.

* * *

- State-of-the-art math-aware search engines use exact keyword matching.
- Soft vector space of @sidorov2014soft can use both exact and fuzzy keyword matching.

/soft-vsm.pdf

- We aim to answer the following four research questions:
    1. Does the soft vector space model outperform sparse information retrieval?
    2. Which math representation works best with the soft vector space model?
    3. Which notion of similarity between keywords and symbols works best?
    4. Is it better to use a single model for both text and math or two models?

# Methods

## Datasets {#datasets}

In our experiments, we used the Math StackExchange and ArXMLiv corpora:

Math StackExchange

:   The Math StackExchange collection v1.2 (MSE) provided by the organizers of
    the ARQMath-2 lab [@mansouri2021overview, Section 3] contains 2,466,080 posts
    from the Math StackExchange question answering website in HTML5 with math
    formulae in LaTeX. An improved Math Stack Exchange collection v1.3 was made
    available by the organizers of the ARQMath-3 lab [@mansouri2022overview,
    Section 3], which we did not use due to time constraints.

ArXMLiv

:   The ArXMLiv 2020 corpus [@ginev2020arxmliv] contains 1,571,037 scientific
    preprints from ArXiv in the HTML5 format with math formulae in MathML.
    Documents in the corpus were converted from LaTeX sources and are divided
    into the following subsets according to the severity of errors encountered
    during conversion: `no-problem` (10%), `warning` (60%), and `error` (30%).

From the corpora, we produced a number of datasets in different formats that we
used to train our tokenizers and language models:

- The *Text + LaTeX* datasets contain text and math formulae in the LaTeX
  format surrounded by `[MATH]` and `[/MATH]` tags.
- By contrast, the *Text* datasets only contains text with math formulae removed
  and the *LaTeX* datasets only contains formulae in the LaTeX format.
- Finally, the *Tangent-L* datasets contain formulae in the format used by the
  state-of-the-art search engine from the University of Waterloo.

To train text & math language models, we combined MSE with the `no-problem`
and `warning` subsets of ArXMLiv. To validate our language models, we used a
small part of the `error` subset of ArXMLiv and no data from MSE.

* * *

- In our experiments, we used the Math StackExchange [@mansouri2021overview]
  and ArXMLiv [@ginev2020arxmliv] corpora.
- From the corpora, we produced a number of datasets in different formats:

    Text + LaTeX

    :   *We denote the set of branches with `[MATH] B_{0}, B_{1}, \ldots, B_{n}
         [/MATH]` where `[MATH] n [/MATH]` are the number of branches.*

    Text

    :   *(Graphs of residually finite groups) Assume that and are satisfied.
         Let be a graph of groups. If is infinite then assume that is
         continuous.*

    LaTeX

    :   `\begin{pmatrix} 1 & n \\ 0 & 1 \end{pmatrix}`

    Tangent-L

    :   `#(start)# #(v!△,/,n,-)# #(v!△,/,n)# #(/,v!l,n,n)# #(/,v!l,n)#
         #(v!l,!0,nn)# #(v!l,!0)# #(end)#`

- For training, we combined MSE with `no-problem` and `warning` subsets of
  ArXMLiv.
- For validation, we used part of the `error` subset of ArXMLiv and no data
  from MSE.

## Tokenization {#tokenization}

In our system, we used several tokenizers:

- To tokenize text, we used the BPE tokenizer of the `roberta-base` language
  model [@liu2019roberta].
- To tokenize LaTeX, we trained a BPE tokenizer with a vocabulary of size
  50,000 on our LaTeX dataset.
- To tokenize Tangent-L, we strip leading and trailing hash signs from a
  formula representation and then split it into tokens using pairs of hash
  signs separated by one or more space as the delimiter.
- To tokenize text + LaTeX, we extended the BPE tokenizer of `roberta-base`
  with the `[MATH]` and `[/MATH]` tags and with the tokens recognized by our
  LaTeX tokenizer.

* * *

Text

:    BPE tokenizer of the `roberta-base` language model [@liu2019roberta]

LaTeX

:    BPE tokenizer trained on our LaTeX dataset

Tangent-L

:    Strip leading and trailing `#` and then split using `#\s+#` Perl regex

Text + LaTeX

:   BPE tokenizer of `roberta-base` extended with `[MATH]` and `[/MATH]`
    special tags and with tokens recognized by our LaTeX tokenizer

## Language Modeling {#language-modeling}

In our experiments, we used two different types of language models:

Shallow log-bilinear models

:   We trained shallow `word2vec` language models [@mikolov2013distributed] on
    our text + LaTeX, text, LaTeX, and Tangent-L datasets.

    On text documents, a technique known as *constrained positional weighting*
    has been shown to improve the performance of `word2vec` models on
    analogical reasoning and causal language modeling [@novotny2022when].
    To evaluate the impact of constrained positional weighting on math
    information retrieval, we trained `word2vec` models both with and without
    constrained positional weighting for every dataset.

Deep transformer models

:   To model text, we used pre-trained `roberta-base` model [@liu2019roberta].

    To model text and math in the LaTeX format, we replaced the tokenizer of
    `roberta-base` with our text + LaTeX tokenizer, we randomly initialized
    weights for the new tokens, and we fine-tuned our model on our text + LaTeX
    dataset for one epoch using the masked language modeling objective.
    We called our model MathBERTa and released it to the HF Model Hub.

* * *

Shallow log-bilinear models

:   Word2vec models [@mikolov2013distributed] trained on all our datasets

Deep transformer models

:   RoBERTa model [@liu2019roberta] fine-tuned on our text + LaTeX dataset

## Token Similarity {#token-similarity}

To determine the similarity of text and math tokens, we first extracted their
global representations from our language models:

Shallow log-bilinear models

:   We extracted token vectors from the input and output matrices of our
    `word2vec` models and averaged them to produce global token embeddings.

Deep transformer models

:   Unlike `word2vec`, transformer models do not contain global representations
    of tokens, but produce representations of tokens in the context of
    a sentence. To extract global token embeddings from `roberta-base` and
    MathBERTa, we decontextualized their contextual token embeddings
    [@stefanik2021regemt, Section 3.2] on the sentences from our text + LaTeX
    dataset.

Then, we produced two types of token similarity matrices:

Lexical similarity

:   We used the method of @charlet2017simbow [Section 2.2] to produce
    similarity matrices using the Levenshtein distance between the tokens.

Semantic similarity

:   We used the method of @charlet2017simbow [Section 2.1] to produce
    similarity matrices using the cosine similarity between the global
    token embeddings.

Finally, to produce token similarity matrices that captured both lexical and
semantic similarity between tokens, we combined every semantic similarity
matrix with a corresponding lexical similarity matrix as follows:

/combine_similarity_matrices.tex

* * *

- First, we extracted global representations of tokens from our language models:

    Shallow log-bilinear models

    :   We averaged input and output matrices of Word2vec.

    Deep transformer models

    :   We decontextualized [@stefanik2021regemt] the contextual token embeddings.

- Then, we produced two types of token similarity matrices:

    Lexical similarity

    :   Using the Levenshtein distance between tokens

    Semantic similarity

    :   Using the cosine similarity between global token representations

- Finally, we combined lexical and semantic similarity matrices:

/combine_similarity_matrices.tex

## Soft Vector Space Modeling {#soft-vector-space-modeling}

In order to find answers to math questions, we used sparse retrieval with the soft
vector space model of @sidorov2014soft, using Lucene BM25 [@kamphuis2020bm25,
Table 1] as the vector space and our combined similarity matrices as the token
similarity measure. To address the bimodal nature of math questions and
answers, we used the following two approaches:

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

* * *

Joint models

:   Single soft vector space model represents both text and math.

Interpolated models

:   Two models represent text and math separately:

/interpolate_similarity_scores.tex

- Questions and answers contain title repeated γ times and body text.
