Good afternoon, I am Vítek Novotný. You will recall that I have already spoken
in the previous session, where I described Task 3. Besides helping to organize
Task 3, I have also participated in Task 1 together with my colleague Michal
Štefánik from the Masaryk University in Brno, Czech Republic. In this talk, I
will describe our Task 1 submission.

# Introduction {#introduction}

State-of-the-art systems for finding answers to math questions use sparse
retrieval techniques that can detect exact key word matches with high
precision, but fail to retrieve texts that are semantically similar but use
different terminology. This shortcoming is all the more apparent with
mathematical texts, where the same information can be expressed in two
completely different systems of writing and thought: the natural language and
the language of mathematics.

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
previously explored. ↷

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

↷

- We aim to answer the following four research questions:
    1. Does the soft vector space model outperform sparse information retrieval?
    2. Which math representation works best with the soft vector space model?
    3. Which notion of similarity between keywords and symbols works best?
    4. Is it better to use a single model for both text and math or two models?

# Methods

## Datasets {#datasets}

In our experiments, we used the Math StackExchange and ArXMLiv corpora.
[@mansouri2021overview] [@ginev2020arxmliv]. The *Math StackExchange* corpus
contains 2.5 million posts from the Math StackExchange question answering online
forum. The *ArXMLiv* corpus contains 1.6 million scientific preprints from ArXiv.

From the corpora, we produced a number of datasets in different formats that we
used to train our tokenizers and language models:

- The *Text + LaTeX* datasets contain text and math formulae, where the
  formulae are in the LaTeX format and surrounded by special tags.
- By contrast, the *Text* datasets only contain text with math formulae removed
  and the *LaTeX* datasets only contain formulae in the LaTeX format.
- Finally, the *Tangent-L* datasets contain formulae in the format used by the
  state-of-the-art search engine from the University of Waterloo.

To train our tokenizers and language models, we combined Math StackExchange
with the `no-problem` and `warning` subsets of ArXMLiv. To validate our
language models, we used a small part of the `error` subset of ArXMLiv.

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

- To tokenize text, we used the byte pair encoding tokenizer of the
  `roberta-base` language model [@liu2019roberta].
- To tokenize LaTeX, we trained a byte pair encoding tokenizer with a
  vocabulary of size 50,000 on our LaTeX dataset.
- To tokenize Tangent-L, we stripped leading and trailing hash signs from a
  formula representation and then we split the remainder into tokens
  using pairs of hash signs separated by one or more spaces as the delimiter.
- To tokenize text + LaTeX, we extended the tokenizer of `roberta-base` with
  the tokens recognized by our LaTeX tokenizer. ↷

* * *

Text

:    BPE tokenizer of the `roberta-base` language model [@liu2019roberta]

LaTeX

:    BPE tokenizer trained on our LaTeX dataset

Tangent-L

:    Strip leading and trailing `#` and then split using `#\s+#` Perl regex

Text + LaTeX

:   BPE tokenizer of `roberta-base` extended with `[MATH]` and `[/MATH]`
    special tags and with tokens recognized by our LaTeX tokenizer ↷

## Language Modeling {#language-modeling}

In our experiments, we also used two different types of language models:

1. We trained shallow `word2vec` language models [@mikolov2013distributed] on
    all our datasets.

    A technique known as *constrained positional weighting* has been shown to
    improve the performance of `word2vec` models on analogical reasoning and
    causal language modeling [@novotny2022when].  To evaluate the impact of
    constrained positional weighting on math information retrieval, we trained
    `word2vec` models both with and without constrained positional weighting for
    every dataset.

2. We also trained deep transformer language models.

    To model text, we used a pre-trained `roberta-base` model [@liu2019roberta].

    To model text and math in the LaTeX format, we replaced the tokenizer of
    `roberta-base` with our text + LaTeX tokenizer, we randomly initialized
    weights for the new tokens, and we fine-tuned our model on our text + LaTeX
    dataset for one epoch using the autoregressive masked language modeling
    objective. We called our model MathBERTa and we released it to the Hugging
    Face Model Hub.

* * *

Shallow log-bilinear models

:

- Word2vec models [@mikolov2013distributed] with and without positional
  weighting [@novotny2022when] for all formats

Deep transformer models

:

- Pre-trained `roberta-base` model for text
- Fine-tuned MathBERTa model for text + LaTeX

## Token Similarity {#token-similarity}

To determine the similarity of text and math tokens, we first extracted their
global representations from our language models:

- For our `word2vec` models, we extracted token vectors from the input and
  output matrices of the models. Then, we averaged the input and output vectors
  to produce global token embeddings.

- For our deep transformer models, we *decontextualized* their contextual token
  embeddings [@stefanik2021regemt, Section 3.2] in order to obtain global token
  embeddings. We ded this by taking the average of all contextual embeddings
  for a token in the sentences from our text + LaTeX dataset.

Then, we produced two types of token similarity matrices:

- *Lexical similarity* matrices, where we used the Levenshtein distance
  between tokens, and

- *Semantic similarity* matrices, where we used the cosine similarity between
  the global token embeddings.

Finally, to produce token similarity matrices that captured both lexical and
semantic similarity between tokens, we combined every semantic similarity
matrix with a corresponding lexical similarity matrix. In our experiments, we
only used the combined token similarity matrices. ↷

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

↷

## Soft Vector Space Modeling {#soft-vector-space-modeling}

In order to find answers to math questions, we used sparse retrieval with the soft
vector space model of @sidorov2014soft, using Lucene BM25 [@kamphuis2020bm25,
Table 1] as the vector space and our combined similarity matrices as the token
similarity measure. To address the bimodal nature of math information
retrieval, we used the following two approaches:

Joint models

:   To allow users to query math information using natural language and
    vise versa, we used single soft vector space models to jointly represent
    both text and math.

Interpolated models

:   To properly represent the different frequency distributions of text and
    math tokens, we used separate soft vector space models for text and math.
    The final score of an answer is determined by linear interpolation of the
    scores assigned by the two soft vector space models:

To represent a question in the soft vector space model, we used the tokens in
the title and body text of the question. To represent an answer, we used the
tokens in the title of its parent question and in the body text of the answer.
To give greater weight to tokens in the title, we repeated them several times,
which proved useful in ARQMath-2 [@novotny2021ensembling, Section 3.2].

* * *

Joint models

:   Single soft vector space model represents both text and math.

Interpolated models

:   Two models represent text and math separately:

/interpolate_similarity_scores.tex

- Questions and answers are indexed as title repeated γ times and body text.

# Results {#results}

Using our experimental results, we can answer our research questions as follows:

1. *Does the soft vector space model outperform sparse information retrieval
   baselines on the math information retrieval task?*

   Using the soft vector space model to capture the semantic similarity between
   tokens consistently improves effectiveness compared to sparse retrieval
   baselines, both for just text and for text combined with different math
   representations.

2. *Which math representation works best with the soft vector space model?*

   Among the LaTeX and Tangent-L math representations, our soft vector space
   models using Tangent-L achieve the highest effectiveness.

3. *Which notion of similarity between key words and symbols works best?*

   Among lexical and semantic similarity, all joint models and all
   interpolated models for text reach their highest effectiveness by combining
   both lexical and semantic similarity, but place slightly more weight on
   lexical similarity. The interpolated models for math gave mixed results: The
   model for Tangent-L reaches the highest efficiency by using only semantic
   similarity, whereas the model for LaTeX reaches the highest efficiency by
   using only lexical similarity.

   Among sources of semantic similarity, joint models achieve comparable
   effectiveness with non-positional `word2vec`, positional `word2vec`, and
   MathBERTa, and interpolated models achieve comparable effectiveness with
   non-positional `word2vec` and positional `word2vec`. This may indicate that
   the soft vector space model does not fully exploit the semantic information
   provided by the sources of semantic similarity and therefore does not
   benefit from their improvements after a certain threshold.

4. *Is it better to use a single soft vector space model to represent both
   text and math or to use two separate models?*

   All our interpolated models achieved higher effectiveness on ARQMath-3
   Task 1 than our joint models. This shows that it is generally better
   to use two separate models to represent text and math even at the expense
   of losing the ability to model the similarity between text and math tokens.

* * *

| Model | α₁ | γ₁ | α₂ | γ₂ | β | NDCG' |
|-------|----|----|----|----|---|-------|
| Interpolated text + Tangent-L (positional `word2vec`)     | 0.7 | 2 | 0.0 | 5 | 0.7 | 0.355 |
| Interpolated text + Tangent-L (non-positional `word2vec`) | 0.6 | 2 | 0.0 | 5 | 0.7 | 0.351 |
| Interpolated text + Tangent-L (no token similarities)     |     | 2 |     | 4 | 0.6 | 0.349 |
| Interpolated text + LaTeX (positional `word2vec`)         | 0.7 | 2 | 1.0 | 5 | 0.6 | 0.288 |
| Interpolated text + LaTeX (non-positional `word2vec`)     | 0.6 | 2 | 1.0 | 5 | 0.6 | 0.288 |
| Interpolated text + LaTeX (no token similarities)         |     | 2 |     | 5 | 0.6 | 0.257 |
| Joint text + LaTeX (non-positional `word2vec`)            | 0.6 | 5 |     |   |     | 0.251 |
| Joint text + LaTeX (positional `word2vec`)                | 0.7 | 5 |     |   |     | 0.249 |
| Joint text + LaTeX (MathBERTa)                            | 0.6 | 4 |     |   |     | 0.249 |
| Joint text (`roberta-base`)                               | 0.6 | 2 |     |   |     | 0.247 |
| Joint text (no token similarities)                        |     | 2 |     |   |     | 0.235 |
| Joint text + LaTeX (no token similarities)                |     | 3 |     |   |     | 0.224 |

: Results with optimized values of α (lexical similarity), β (text weight), and γ (title weight), where α₁ and γ₁ are parameters of the text model and  α₂ and γ₂ are parameters of the math model.

# Results {#results-continuation}

Here is another look at the results, which emphasizes the relationships
between our models and highlights the impact of different extensions on the
effectiveness. We can clearly see how far ahead the interpolated Text +
Tangent-L model is compared to the other models. This is likely because
it's the only representation that exposes the structure of math formulae.

We can also see that fuzzy matching is always benefitial compared to sparse
retrieval baselines on the left and that positional `word2vec` always improves
effectiveness of the interpolated models, albeit slightly.

In the bottom part of the figure, we can see that indexing LaTeX tokens in
addition to text tokens has an adverse effect on the effectiveness of the
baseline text. However, we can also see that the soft vector space model can
take advantage of the LaTeX tokens and achieves better effectiveness than both
the baseline text model and the soft vector space model for text.

* * *

/visualization-of-extensions.tex

# Results {#artefacts-and-code-pearls}

Besides reporting our experimental results, we have also released an online
demo of our system. In the demo, you can browse a sample of Task 1 topics
and the top results for these topics. You can also select two documents and
compare them to see which exact and fuzzy keyword matches contributed to their
similarity scores.

Futhermore, we have released our MathBERTa model at the Hugging Face Model hub,
so that you can start using it in your own software in a matter of seconds.
Finally, we have released the full source code of our system at GitHub, so that
you can study it and reuse parts of it in your own systems. One code pearl,
which we are quite proud of, is the GPU-accelerated algorithm for word
embedding decontextualization, which can be used to produce global embeddings
from deep transformer models in all sorts of useful applications.

* * *

## Artefacts

- Online demo of our system is available at <https://witiko.github.io/scm-at-arqmath3>.
- Fine-tuned RoBERTa model is available at <https://huggingface.co/witiko/mathberta>.
- Our system has been open-sourced at <https://github.com/witiko/scm-at-arqmath3>.

## Code Pearls

- GPU-accelerated word embedding decontextualization is available in file
  [`system/extract_decontextualized_word_embeddings.py`][1].

 [1]: https://github.com/Witiko/scm-at-arqmath3/blob/5585ddadf0d911e24c091bfe4c74511dbc24fdec/system/extract_decontextualized_word_embeddings.py#L104-L141

/formula1.jpg
