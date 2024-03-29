from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
import csv
from enum import Enum
from itertools import chain, product
import json
import logging
from multiprocessing import get_context
from pathlib import Path
import re
from sys import argv, setrecursionlimit
from typing import Optional, Iterable, List, Dict, Tuple, Union, Set

from arqmath_eval import get_ndcg as _get_ndcg
from gensim.corpora import Dictionary  # type: ignore
from gensim.interfaces import TransformationABC as TermWeightTransformation  # type: ignore
from gensim.models import LuceneBM25Model  # type: ignore
from gensim.similarities import (
    SparseMatrixSimilarity,  # type: ignore
    SoftCosineSimilarity,  # type: ignore
    SparseTermSimilarityMatrix,  # type: ignore
)
from lxml.html import fromstring
from more_itertools import zip_equal
from pv211_utils.arqmath.entities import (
    ArqmathQueryBase as Query,
    ArqmathAnswerBase as Answer,
    ArqmathQuestionBase as Question,
)
from pv211_utils.arqmath.eval import ArqmathEvaluation as Evaluation
from pv211_utils.arqmath.irsystem import ArqmathIRSystemBase as System
from pv211_utils.arqmath.loader import load_queries, load_judgements, ArqmathJudgements as Judgements
from scipy.sparse import dok_matrix
from tokenizers import Tokenizer as _Tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer
from filelock import FileLock

from .prepare_msm_dataset import (
    get_input_text_format,
    get_questions_and_answers as _get_questions_and_answers,
    read_document_latex,
    read_document_tangentl,
    read_document_text,
    read_document_text_latex,
    read_document_text_tangentl,
    TextFormat,
)
from .combine_similarity_matrices import get_combined_term_similarity_matrix, get_alphas, Alpha
from .prepare_levenshtein_similarity_matrix import get_dictionary


LOGGER = logging.getLogger(__name__)


AnswerIndex = int

TermId = int
Weight = float
TermWeight = Tuple[TermId, Weight]
Vector = List[TermWeight]

Gamma = int
Parameters = Tuple[Alpha, Gamma]

Ndcg = float
Interval = Tuple[float, float]
NdcgAndInterval = Tuple[Ndcg, Interval]

Similarity = float

SimilarityIndex = Union[SparseMatrixSimilarity, SoftCosineSimilarity]

Task = str

Token = str
Text = List[Token]
Line = str

Document = Union[Query, Question, Answer]


class Year(Enum):
    ARQMATH_2020 = 2020
    ARQMATH_2021 = 2021
    ARQMATH_2022 = 2022

    @classmethod
    def from_int(cls, year: int) -> 'Year':
        if year == 2020:
            return cls.ARQMATH_2020
        elif year == 2021:
            return cls.ARQMATH_2021
        elif year == 2022:
            return cls.ARQMATH_2022
        else:
            raise ValueError(f'Unknown year {year}')

    def __int__(self) -> int:
        year = self.value
        return year

    def get_number_of_queries(self) -> int:
        year = int(self)
        if year == 2020:
            return 77
        elif year == 2021:
            return 71
        elif year == 2022:
            return 78
        else:
            raise ValueError(f'Unknown year {year}')

    def __len__(self) -> int:
        number_of_queries = self.get_number_of_queries()
        return number_of_queries

    def maybe_get_judgements(self, queries: Iterable[Query],
                             answers: Iterable[Answer]) -> Optional[Judgements]:
        year = int(self)
        if year < 2022:
            queries_dict = OrderedDict()
            for query in queries:
                queries_dict[query.query_id] = query
            answers_dict = OrderedDict()
            for answer in answers:
                answers_dict[answer.document_id] = answer
            judgements = load_judgements(queries_dict, answers_dict, year=year)
        else:
            judgements = None
        return judgements

    def maybe_get_task(self) -> Optional[Task]:
        year = int(self)
        if year <= 2022:
            task = f'task1-{year}'
        else:
            task = None
        return task


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: Line) -> Text:
        pass


class TextTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    def tokenize(self, line: Line) -> Text:
        tokens = self.tokenizer.tokenize(line)
        return tokens


class TextLaTeXTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('./roberta-base-text+latex/',
                                                       add_prefix_space=True)

    def tokenize(self, line: Line) -> Text:
        tokens = self.tokenizer.tokenize(line)
        return tokens


class LaTeXTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = _Tokenizer.from_file('tokenizer-latex.json')

    def tokenize(self, line: Line) -> Text:
        tokens = self.tokenizer.encode(line).tokens
        return tokens


class TextTangentLTokenizer(Tokenizer):
    def __init__(self):
        self.text_tokenizer = TextTokenizer()
        self.tangentl_tokenizer = TangentLTokenizer()

    def tokenize(self, line: Line) -> Text:
        first_text_span, *tangentl_span_heads = re.split(r'\s*\[MATH\]\s*', line)
        tokens = self.text_tokenizer.tokenize(first_text_span)
        for tangentl_span_head in tangentl_span_heads:
            tangentl_span, *text_spans = re.split(r'\s*\[/MATH\]\s*', tangentl_span_head)
            tangentl_tokens = self.tangentl_tokenizer.tokenize(tangentl_span)
            tokens.extend(tangentl_tokens)
            assert len(text_spans) <= 1
            if len(text_spans) == 1:
                text_span, = text_spans
                text_tokens = self.text_tokenizer.tokenize(text_span)
                tokens.extend(text_tokens)
        return tokens


class TangentLTokenizer(Tokenizer):
    def tokenize(self, line: Line) -> Text:
        tokens = line.strip('#').split('# #')
        return tokens


def get_tokenizer(text_format: TextFormat) -> Tokenizer:
    if text_format == 'text':
        tokenizer = TextTokenizer()
    elif text_format == 'text+latex':
        tokenizer = TextLaTeXTokenizer()
    elif text_format == 'latex':
        tokenizer = LaTeXTokenizer()
    elif text_format == 'text+tangentl':
        tokenizer = TextTangentLTokenizer()
    elif text_format == 'tangentl':
        tokenizer = TangentLTokenizer()
    else:
        raise ValueError(f'Unknown text format {text_format}')
    return tokenizer


def get_gammas() -> Iterable[Gamma]:
    gammas = [1, 2, 3, 4, 5]
    return gammas


def get_parameters(input_similarity_matrix_file: Optional[Path]) -> Iterable[Parameters]:
    alphas = get_alphas() if input_similarity_matrix_file is not None else [0.0]
    gammas = get_gammas()
    parameters = product(alphas, gammas)
    return parameters


class Preprocessor:
    BODY_WEIGHT = 1

    def __init__(self, text_format: TextFormat, tokenizer: Tokenizer, questions: Iterable[Question],
                 maybe_gamma: Optional[Gamma]):
        self.text_format = text_format
        self.tokenizer = tokenizer
        self.answer_to_question = {
            answer: question
            for question in questions
            for answer in question.answers
        }
        if maybe_gamma is None:
            self.gamma = None
        else:
            self.gamma = maybe_gamma

    def preprocess_part(self, document_part: str) -> Text:
        tree = fromstring(document_part)
        paragraphs = list()
        for paragraph in tree.xpath('//p'):
            if self.text_format == 'text':
                paragraph_text = read_document_text(paragraph, min_paragraph_length=None)
            elif self.text_format == 'text+latex':
                paragraph_text = read_document_text_latex(paragraph, min_paragraph_length=None)
            elif self.text_format == 'latex':
                paragraph_text = read_document_latex(paragraph, min_math_length=None)
            elif self.text_format == 'text+tangentl':
                paragraph_text = read_document_text_tangentl(paragraph, min_paragraph_length=None)
            elif self.text_format == 'tangentl':
                paragraph_text = read_document_tangentl(paragraph)
            else:
                raise ValueError(f'Unknown text format {self.text_format}')
            paragraphs.extend(paragraph_text)
        texts = [self.tokenizer.tokenize(paragraph) for paragraph in paragraphs]
        text = list(chain(*texts))
        return text

    def preprocess(self, document: Document) -> Text:
        if self.gamma is None:
            LOGGER.warning(f'Unparametrized preprocessor, assuming gamma={self.BODY_WEIGHT}')
            gamma = 1
        else:
            gamma = self.gamma

        texts = []
        if isinstance(document, Answer):
            if document in self.answer_to_question:
                question = self.answer_to_question[document]
                texts += [self.preprocess_part(question.title)] * gamma
            texts += [self.preprocess_part(document.body)] * self.BODY_WEIGHT
        else:
            texts += [self.preprocess_part(document.title)] * gamma
            texts += [self.preprocess_part(document.body)] * self.BODY_WEIGHT
        text = list(chain(*texts))
        return text


def get_preprocessor(text_format: TextFormat, questions: Iterable[Question],
                     gamma: Gamma) -> Preprocessor:
    tokenizer = get_tokenizer(text_format)
    preprocessor = Preprocessor(text_format, tokenizer, questions, gamma)
    return preprocessor


def get_unparametrized_preprocessor(text_format: TextFormat, questions: Iterable[Question]) -> Preprocessor:
    tokenizer = get_tokenizer(text_format)
    preprocessor = Preprocessor(text_format, tokenizer, questions, None)
    return preprocessor


def get_number_of_workers() -> Optional[int]:
    number_of_workers = None
    return number_of_workers


class BulkSearchSystem(System):
    def bulk_search(self, queries: Iterable[Query]) -> Iterable[Tuple[Query, Iterable[Answer]]]:
        for query in queries:
            answers = self.search(queries)
            yield (query, answers)


def get_bm25_model(dictionary: Dictionary):
    bm25_model = LuceneBM25Model(dictionary=dictionary)
    return bm25_model


class JointBM25System(BulkSearchSystem, metaclass=ABCMeta):
    CURRENT_INSTANCE: Optional['JointBM25System'] = None

    def __init__(self, dictionary: Dictionary, preprocessor: Preprocessor, answers: Iterable[Answer]):
        self.dictionary = dictionary
        self.preprocessor = preprocessor
        self.bm25_model = get_bm25_model(self.dictionary)
        self.index_to_answer: Dict[AnswerIndex, Answer] = dict(enumerate(answers))

    @abstractmethod
    def get_similarity_index(self) -> SimilarityIndex:
        pass

    def _document_to_vector(self, document: Document) -> Vector:
        preprocessed_document = self.preprocessor.preprocess(document)
        document_vector = self.dictionary.doc2bow(preprocessed_document)
        if isinstance(document, Answer):
            document_vector = self.bm25_model[document_vector]
        return document_vector

    def _documents_to_vectors(self, documents: Iterable[Document]) -> Iterable[Vector]:
        self.__class__.CURRENT_INSTANCE = self
        number_of_workers = get_number_of_workers()
        with get_context('fork').Pool(number_of_workers) as pool:
            vectors = pool.imap(self.__class__._document_to_vector_helper, documents)
            for vector in vectors:
                yield vector
        self.__class__.CURRENT_INSTANCE = None

    @classmethod
    def _document_to_vector_helper(cls, document: Document) -> Vector:
        return cls.CURRENT_INSTANCE._document_to_vector(document)

    def get_similarities(self, query: Query) -> Dict[Answer, Similarity]:
        similarity_index = self.get_similarity_index()
        query_vector = self._document_to_vector(query)
        similarity_vector = similarity_index[query_vector]
        similarities = {
            self.index_to_answer[answer_index]: similarity
            for answer_index, similarity
            in enumerate(similarity_vector)
        }
        return similarities

    def search(self, query: Query) -> Iterable[Answer]:
        similarities = self.get_similarities(query)
        for answer, _ in sorted(similarities.items(), key=result_sort_key):
            yield answer

    def get_bulk_similarities(self, queries: Iterable[Query]) -> Iterable[Tuple[Query, Dict[Answer, Similarity]]]:
        queries = list(queries)
        similarity_index = self.get_similarity_index()
        query_vectors = list(self._documents_to_vectors(queries))
        bulk_similarities = similarity_index[query_vectors]
        for query, similarity_vector in zip_equal(queries, bulk_similarities):
            similarities = {
                self.index_to_answer[answer_index]: similarity
                for answer_index, similarity
                in enumerate(similarity_vector)
            }
            yield (query, similarities)

    def bulk_search(self, queries: Iterable[Query]) -> Iterable[Tuple[Query, Iterable[Answer]]]:
        bulk_similarities = self.get_bulk_similarities(queries)
        for query, similarities in bulk_similarities:
            answers = (answer for answer, _ in sorted(similarities.items(), key=result_sort_key))
            yield (query, answers)


def result_sort_key(args: Tuple[Answer, Similarity]):
    answer, similarity = args
    sort_key = (-similarity, answer)
    return sort_key


class LuceneBM25System(JointBM25System):
    def __init__(self, dictionary: Dictionary, preprocessor: Preprocessor, answers: Iterable[Answer]):
        answers = list(answers)
        super().__init__(dictionary, preprocessor, list(answers))
        vectors = self._documents_to_vectors(answers)
        vectors = tqdm(vectors, total=len(answers), desc='Indexing answers to BM25')
        self.bm25_index = SparseMatrixSimilarity(
            vectors,
            num_docs=len(answers),
            num_terms=len(self.dictionary),
            normalize_queries=False,
            normalize_documents=False,
        )

    def get_similarity_index(self) -> SimilarityIndex:
        return self.bm25_index


class SCMSystem(JointBM25System):
    def __init__(self, dictionary: Dictionary, similarity_matrix: SparseTermSimilarityMatrix,
                 preprocessor: Preprocessor, answers: Iterable[Answer]):
        answers = list(answers)
        super().__init__(dictionary, preprocessor, list(answers))
        vectors = self._documents_to_vectors(answers)
        vectors = tqdm(vectors, total=len(answers), desc='Indexing answers to SCM')
        self.scm_index = SoftCosineSimilarity(
            vectors,
            similarity_matrix,
            normalized=('maintain', 'maintain'),
        )

    def get_similarity_index(self) -> SimilarityIndex:
        return self.scm_index


def get_system(text_format: TextFormat, questions: Iterable[Question], answers: Iterable[Answer],
               dictionary: Dictionary, parameters: Parameters,
               similarity_matrix: Optional[SparseTermSimilarityMatrix],
               lock_file: Path) -> BulkSearchSystem:
    with FileLock(str(lock_file)):
        _, gamma = parameters
        preprocessor = get_preprocessor(text_format, questions, gamma)
        if similarity_matrix is None:
            system = LuceneBM25System(dictionary, preprocessor, answers)
        else:
            system = SCMSystem(dictionary, similarity_matrix, preprocessor, answers)
        return system


def get_queries(year: Year, output_text_format: TextFormat) -> Iterable[Query]:
    input_text_format = get_input_text_format(output_text_format)
    queries_dict = load_queries(input_text_format, year=int(year))
    queries = queries_dict.values()
    assert len(queries) >= len(year)
    return queries


def get_questions_and_answers(msm_input_directory: Path, output_text_format: str, *args, **kwargs):
    input_text_format = get_input_text_format(output_text_format)
    questions_and_answers = _get_questions_and_answers(msm_input_directory, input_text_format, *args, **kwargs)
    return questions_and_answers


def maybe_get_term_similarity_matrix(input_file: Optional[Path],
                                     parameters: Parameters) -> Optional[SparseTermSimilarityMatrix]:
    alpha, _ = parameters
    if input_file is not None:
        term_similarity_matrix = get_combined_term_similarity_matrix(input_file, alpha)
    else:
        term_similarity_matrix = None
    return term_similarity_matrix


def get_topn() -> int:
    topn = 1000
    return topn


def produce_serp(system: BulkSearchSystem, queries: Iterable[Query], output_run_file: Path,
                 run_name: str) -> None:
    queries = list(queries)
    topn = get_topn()

    output_run_file.parent.mkdir(exist_ok=True)
    with output_run_file.open('wt') as f:
        LOGGER.info(f'Sent a batch of {len(queries)} queries to {system.__class__.__name__}')
        for query, answers in system.bulk_search(queries):
            for rank, answer in enumerate(answers):
                rank = rank + 1
                if rank > topn:
                    break
                score = 1.0 / float(rank)
                query_id = f'A.{query.query_id}'
                answer_id = answer.document_id
                line = f'{query_id}\t{answer_id}\t{rank}\t{score}\t{run_name}'
                print(line, file=f)


def read_tsv_file(input_file: Path) -> Iterable[Tuple[str, str, int, float, str]]:
    with input_file.open('rt', newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for row in csv_reader:
            query_id, answer_id, rank, score, run_id = row
            yield (query_id, answer_id, int(rank), float(score), run_id)


class TSVFileReader(System):
    def __init__(self, input_file: Path, queries: Iterable[Query], answers: Iterable[Answer]):
        queries = {f'A.{query.query_id}': query for query in queries}
        answers = {answer.document_id: answer for answer in answers}
        self.query_answers = defaultdict(lambda: list())
        for query_id, answer_id, rank, *_ in read_tsv_file(input_file):
            query = queries[query_id]
            answer = answers[answer_id]
            self.query_answers[query].append(answer)
            assert len(self.query_answers[query]) == rank

    def search(self, query: Query) -> Iterable[Answer]:
        answers = self.query_answers[query]
        return answers


def produce_document_maps_corpus(input_run_file: Path,
                                 queries: Iterable[Query],
                                 answers: Iterable[Answer],
                                 input_dictionary_file: Path,
                                 input_similarity_matrix_file: Path,
                                 parameters: Parameters,
                                 text_format: TextFormat,
                                 questions: Iterable[Question],
                                 output_document_maps_file: Path) -> None:
    dictionary = get_dictionary(input_dictionary_file)

    similarity_matrix = maybe_get_term_similarity_matrix(input_similarity_matrix_file, parameters)
    assert similarity_matrix is not None

    _, gamma = parameters

    unparametrized_preprocessor = get_unparametrized_preprocessor(text_format, questions)
    preprocessor = get_preprocessor(text_format, questions, gamma)

    bm25_model = get_bm25_model(dictionary)

    _produce_document_maps_corpus(output_run_file, queries, answers,
                                  dictionary, similarity_matrix,
                                  unparametrized_preprocessor, preprocessor,
                                  output_document_maps_file, None, bm25_model)


def get_minimum_allowed_query_id() -> int:
    minimum_allowed_query_id = 301
    return minimum_allowed_query_id


def get_maximum_allowed_query_id() -> int:
    maximum_allowed_query_id = 310
    return maximum_allowed_query_id


def get_document_maps_topn() -> int:
    document_maps_topn = 5
    return document_maps_topn


def _produce_document_maps_corpus(input_run_file: Path,
                                  queries: Iterable[Query],
                                  answers: Iterable[Answer],
                                  dictionary: Dictionary,
                                  similarity_matrix: SparseTermSimilarityMatrix,
                                  unparametrized_preprocessor: Preprocessor,
                                  preprocessor: Preprocessor,
                                  output_document_maps_file: Path,
                                  query_term_weight_transformer: Optional[TermWeightTransformation],
                                  answer_term_weight_transformer: Optional[TermWeightTransformation]) -> None:
    queries = list(queries)
    similarity_matrix = dok_matrix(similarity_matrix.matrix)

    system = TSVFileReader(input_run_file, queries, answers)
    top_results: Dict[Query, List[Answer]] = dict()
    for query in queries:
        if query.query_id < get_minimum_allowed_query_id():
            continue
        if query.query_id > get_maximum_allowed_query_id():
            continue
        topn = get_document_maps_topn()
        answers = system.search(query)
        answers = list(answers)
        answers = answers[:topn]
        top_results[query] = answers

    top_result_terms: Set[Token] = set()
    top_answers: Set[Answer] = set()
    for query, answers in top_results.items():
        for token in unparametrized_preprocessor.preprocess(query):
            top_result_terms.add(token)
        for answer in answers:
            for token in unparametrized_preprocessor.preprocess(answer):
                top_result_terms.add(token)
            top_answers.add(answer)

    corpus = {'version': '1'}

    corpus['results'] = dict()
    for query, answers in top_results.items():
        answer_ids = [answer.document_id for answer in answers]
        corpus['results'][f'Topic A.{query.query_id}'] = answer_ids

    corpus['dictionary'] = dict()
    for term, term_id in dictionary.token2id.items():
        if term not in top_result_terms:
            continue
        corpus['dictionary'][term_id] = term

    corpus['word_similarities'] = defaultdict(lambda: dict())
    term1_ids, term2_ids = similarity_matrix.nonzero()
    term1_ids, term2_ids = map(int, term1_ids), map(int, term2_ids)
    for term1_id, term2_id in zip(term1_ids, term2_ids):
        term1, term2 = dictionary[term1_id], dictionary[term2_id]
        if term1 not in top_result_terms:
            continue
        if term2 not in top_result_terms:
            continue
        if term1_id >= term2_id:
            continue
        word_similarity = similarity_matrix[term1_id, term2_id]
        word_similarity = float(word_similarity)
        corpus['word_similarities'][term1_id][term2_id] = word_similarity

    corpus['texts']: Dict[str, Text] = defaultdict(lambda: list())
    corpus['texts_bow']: Dict[str, Dict[TermId, Weight]] = defaultdict(lambda: dict())

    top_documents = list()
    for query in top_results:
        query_id = f'Topic A.{query.query_id}'
        top_documents.append((query_id, query, query_term_weight_transformer))
    for answer in top_answers:
        answer_id = answer.document_id
        top_documents.append((answer_id, answer, answer_term_weight_transformer))

    for document_id, document, document_term_weight_transformer in top_documents:
        document_tokens = unparametrized_preprocessor.preprocess(document)
        for token in document_tokens:
            if token not in dictionary.token2id:
                continue
            assert token in top_result_terms
            token_id = dictionary.token2id[token]
            corpus['texts'][document_id].append(str(token_id))
        document_tokens = preprocessor.preprocess(document)
        document_vector = dictionary.doc2bow(document_tokens)
        if document_term_weight_transformer is not None:
            document_vector = document_term_weight_transformer[document_vector]
        for term_id, term_weight in document_vector:
            term = dictionary[term_id]
            assert term in top_result_terms
            corpus['texts_bow'][document_id][term_id] = term_weight

    with output_document_maps_file.open('wt') as f:
        json.dump(corpus, f, sort_keys=True, indent=4)


def evaluate_serp_with_map(input_run_file: Path, queries: Iterable[Query], answers: Iterable[Answer],
                           year: Year, output_map_file: Path) -> None:
    queries, answers = list(queries), list(answers)
    judgements = year.maybe_get_judgements(queries, answers)
    if judgements is None:
        return

    queries = list(queries)
    system = TSVFileReader(input_run_file, queries, answers)
    number_of_workers = get_number_of_workers()
    evaluation = Evaluation(system, judgements, num_workers=number_of_workers)
    map_score = evaluation.mean_average_precision(queries)

    output_map_file.parent.mkdir(exist_ok=True)
    with output_map_file.open('wt') as f:
        print(f'{100 * map_score:.2f}%', file=f)


def get_confidence() -> float:
    confidence = 95.0
    return confidence


def maybe_get_ndcg_and_interval(input_run_file: Path, year: Year) -> Optional[NdcgAndInterval]:
    task = year.maybe_get_task()
    if task is None:
        ndcg_and_interval = None
    else:
        parsed_result = defaultdict(lambda: dict())
        for query_id, answer_id, rank, *_ in read_tsv_file(input_run_file):
            score = 1.0 / rank
            parsed_result[query_id][answer_id] = score
        confidence = get_confidence()
        ndcg_score, interval = _get_ndcg(parsed_result, task, 'all', confidence=confidence)
        ndcg_and_interval = (ndcg_score, interval)
    return ndcg_and_interval


def get_ndcg(input_run_file: Path, year: Year) -> Ndcg:
    ndcg_and_interval = maybe_get_ndcg_and_interval(input_run_file, year)
    assert ndcg_and_interval is not None
    ndcg, _ = ndcg_and_interval
    return ndcg


def evaluate_serp_with_ndcg(input_run_file: Path, year: Year, output_ndcg_file: Path) -> None:
    ndcg_and_interval = maybe_get_ndcg_and_interval(input_run_file, year)
    if ndcg_and_interval is None:
        return

    confidence = get_confidence()
    ndcg_score, interval = ndcg_and_interval
    lower_bound, upper_bound = interval

    output_ndcg_file.parent.mkdir(exist_ok=True)
    with output_ndcg_file.open('wt') as f:
        print(f'{ndcg_score:.3f}, {confidence:g}% CI: [{lower_bound:.3f}; {upper_bound:.3f}]', file=f)


def produce_system(msm_input_directory: Path,
                   output_text_format: TextFormat, input_dictionary_file: Path,
                   input_similarity_matrix_file: Optional[Path],
                   parameters: Parameters, lock_file: Path) -> BulkSearchSystem:
    dictionary = get_dictionary(input_dictionary_file)
    similarity_matrix = maybe_get_term_similarity_matrix(input_similarity_matrix_file, parameters)
    questions, answers = get_questions_and_answers(msm_input_directory, output_text_format)
    questions, answers = list(questions), list(answers)
    system = get_system(output_text_format, questions, answers, dictionary, parameters,
                        similarity_matrix, lock_file)
    return system


def get_optimal_parameters(msm_input_directory: Path,
                           output_text_format: TextFormat, input_dictionary_file: Path,
                           input_similarity_matrix_file: Optional[Path], run_name: str,
                           output_run_file: Path, temporary_output_parameter_file: Path,
                           output_parameter_file: Path, lock_file: Path,
                           fixed_gamma: Optional[int] = None) -> Parameters:
    all_parameters = get_parameters(input_similarity_matrix_file)
    all_parameters = sorted(all_parameters)

    try:
        with output_parameter_file.open('rt') as f:
            obj = json.load(f)
        best_alpha, best_gamma = obj['best_alpha'], obj['best_gamma']
        best_parameters = (best_alpha, best_gamma)
        LOGGER.info(f'Loaded optimal alpha and gamma from {output_parameter_file}')
        return best_parameters
    except IOError:
        try:
            with temporary_output_parameter_file.open('rt') as f:
                obj = json.load(f)
            best_ndcg, best_alpha, best_gamma = obj['best_ndcg'], obj['best_alpha'], obj['best_gamma']
            alpha, gamma = obj['alpha'], obj['gamma']
            best_parameters = (best_alpha, best_gamma)
            parameters = (alpha, gamma)
            assert parameters in all_parameters
            all_parameters = all_parameters[all_parameters.index(parameters) + 1:]
            LOGGER.info('Fast-forwarded optimization of alpha and gamma to last tested values '
                        f'from {temporary_output_parameter_file}')
        except IOError:
            best_ndcg, best_parameters = float('-inf'), None

    all_parameters = tqdm(all_parameters, desc='Optimizing alpha and gamma')

    year_2020 = Year.from_int(2020)
    year_2021 = Year.from_int(2021)

    queries_2020 = list(get_queries(year_2020, output_text_format))
    queries_2021 = list(get_queries(year_2021, output_text_format))

    for parameters in all_parameters:
        alpha, gamma = parameters

        if fixed_gamma is not None and gamma != fixed_gamma:
            continue

        system = produce_system(
            msm_input_directory, output_text_format, input_dictionary_file,
            input_similarity_matrix_file, parameters, lock_file)

        produce_serp(system, queries_2020, output_run_file, run_name)
        ndcg_2020 = get_ndcg(output_run_file, year_2020)
        produce_serp(system, queries_2021, output_run_file, run_name)
        ndcg_2021 = get_ndcg(output_run_file, year_2021)

        ndcg = ndcg_2020 * len(year_2020) + ndcg_2021 * len(year_2021)
        ndcg /= len(year_2020) + len(year_2021)

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_parameters = parameters

        best_alpha, best_gamma = best_parameters
        with temporary_output_parameter_file.open('wt') as f:
            json.dump({'alpha': alpha, 'gamma': gamma,
                       'best_alpha': best_alpha, 'best_gamma': best_gamma,
                       'best_ndcg': best_ndcg}, f)

    assert best_parameters is not None

    best_alpha, best_gamma = best_parameters
    with output_parameter_file.open('wt') as f:
        json.dump({'best_alpha': best_alpha, 'best_gamma': best_gamma, 'best_ndcg': best_ndcg}, f)

    temporary_output_parameter_file.unlink()

    return best_parameters


def main(msm_input_directory: Path, output_text_format: TextFormat,
         input_dictionary_file: Path, input_similarity_matrix_file: Optional[Path],
         run_name: str, temporary_output_run_file: Path,
         output_run_file: Path, output_map_file: Path, output_ndcg_file: Path,
         temporary_output_parameter_file: Path, output_parameter_file: Path,
         lock_file: Path, output_document_maps_file: Path) -> None:
    year = Year.from_int(2022)

    queries = None

    def ensure_queries() -> None:
        nonlocal queries
        if queries is None:
            queries = list(get_queries(year, output_text_format))

    optimal_parameters = None

    def ensure_optimal_parameters() -> None:
        nonlocal optimal_parameters
        if optimal_parameters is None:
            optimal_parameters = get_optimal_parameters(
                msm_input_directory, output_text_format, input_dictionary_file,
                input_similarity_matrix_file, run_name,
                temporary_output_run_file, temporary_output_parameter_file,
                output_parameter_file, lock_file)

            try:
                temporary_output_run_file.unlink()
            except FileNotFoundError:
                pass

    if not output_run_file.exists():
        ensure_optimal_parameters()
        system = produce_system(
            msm_input_directory, output_text_format, input_dictionary_file,
            input_similarity_matrix_file, optimal_parameters, lock_file)

        ensure_queries()
        produce_serp(system, queries, output_run_file, run_name)

    questions, answers = None, None

    def ensure_questions_and_answers() -> None:
        nonlocal questions, answers
        if any([questions is None, answers is None]):
            questions, answers = get_questions_and_answers(msm_input_directory, output_text_format)
            questions, answers = list(questions), list(answers)

    if not output_map_file.exists():
        ensure_queries()
        ensure_questions_and_answers()
        evaluate_serp_with_map(output_run_file, queries, answers, year, output_map_file)

    if not output_ndcg_file.exists():
        evaluate_serp_with_ndcg(output_run_file, year, output_ndcg_file)

    if input_similarity_matrix_file is not None and not output_document_maps_file.exists():
        ensure_queries()
        ensure_questions_and_answers()
        ensure_optimal_parameters()
        produce_document_maps_corpus(output_run_file, queries, answers,
                                     input_dictionary_file, input_similarity_matrix_file,
                                     optimal_parameters, output_text_format, questions,
                                     output_document_maps_file)


if __name__ == '__main__':
    setrecursionlimit(15000)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    assert len(argv) == 14

    msm_input_directory = Path(argv[1])
    text_format = argv[2]
    input_dictionary_file = Path(argv[3])
    input_similarity_matrix_file = Path(argv[4]) if argv[4] != 'none' else None
    run_name = argv[5]
    temporary_output_run_file = Path(argv[6])
    output_run_file = Path(argv[7])
    output_map_file = Path(argv[8])
    output_ndcg_file = Path(argv[9])
    temporary_output_parameter_file = Path(argv[10])
    output_parameter_file = Path(argv[11])
    lock_file = Path(argv[12])
    output_document_maps_file = Path(argv[13])

    main(msm_input_directory, text_format, input_dictionary_file, input_similarity_matrix_file,
         run_name, temporary_output_run_file, output_run_file, output_map_file, output_ndcg_file,
         temporary_output_parameter_file, output_parameter_file, lock_file,
         output_document_maps_file)
