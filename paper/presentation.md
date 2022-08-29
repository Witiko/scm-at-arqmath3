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
  format surrounded by *[MATH]* and *[/MATH]* tags.
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
