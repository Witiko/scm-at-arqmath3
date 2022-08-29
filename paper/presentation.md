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
