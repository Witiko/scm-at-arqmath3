from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
import csv
from enum import Enum, auto
from itertools import chain, product
import json
import logging
from multiprocessing import get_context
from pathlib import Path
from sys import argv, setrecursionlimit
from typing import Optional, Iterable, List, Dict, Tuple, Union

from arqmath_eval import get_ndcg as _get_ndcg
from gensim.corpora import Dictionary  # type: ignore
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
from tokenizers import Tokenizer as _Tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

from .prepare_msm_dataset import (
    Document,
    get_input_text_format,
    get_questions_and_answers as _get_questions_and_answers,
    read_document_latex,
    read_document_tangentl,
    read_document_text,
    read_document_text_latex,
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

Year = int


class RunType(Enum):
    ARQMATH_2020 = auto()
    ARQMATH_2021 = auto()
    ARQMATH_2022 = auto()

    def get_year(self) -> Year:
        if self == RunType.ARQMATH_2020:
            return 2020
        elif self == RunType.ARQMATH_2021:
            return 2021
        elif self == RunType.ARQMATH_2022:
            return 2022
        else:
            raise ValueError(f'Unknown run type {self}')

    def get_number_of_queries(self) -> int:
        if self == RunType.ARQMATH_2020:
            return 77
        elif self == RunType.ARQMATH_2021:
            return 100
        elif self == RunType.ARQMATH_2022:
            return 100
        else:
            raise ValueError(f'Unknown run type {self}')

    def __len__(self) -> int:
        number_of_queries = self.get_number_of_queries()
        return number_of_queries

    def maybe_get_judgements(self, queries: Iterable[Query],
                             answers: Iterable[Answer]) -> Optional[Judgements]:
        year = self.get_year()
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
        year = self.get_year()
        if year < 2022:
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
                 gamma: Gamma):
        self.text_format = text_format
        self.tokenizer = tokenizer
        self.answer_to_question = {
            answer: question
            for question in questions
            for answer in question.answers
        }
        self.gamma = gamma

    def _preprocess_part(self, document_part: str) -> Text:
        tree = fromstring(document_part)
        paragraphs = list()
        for paragraph in tree.xpath('//p'):
            if self.text_format == 'text':
                paragraph_text = read_document_text(paragraph, min_paragraph_length=None)
            elif self.text_format == 'text+latex':
                paragraph_text = read_document_text_latex(paragraph, min_paragraph_length=None)
            elif self.text_format == 'latex':
                paragraph_text = read_document_latex(paragraph, min_math_length=None)
            elif self.text_format == 'tangentl':
                paragraph_text = read_document_tangentl(paragraph)
            else:
                raise ValueError(f'Unknown text format {self.text_format}')
            paragraphs.extend(paragraph_text)
        texts = [self.tokenizer.tokenize(paragraph) for paragraph in paragraphs]
        text = list(chain(*texts))
        return text

    def preprocess(self, document: Document) -> Text:
        texts = []
        if isinstance(document, Answer):
            texts += [self._preprocess_part(document.body)] * self.BODY_WEIGHT
            if document in self.answer_to_question:
                question = self.answer_to_question[document]
                texts += [self._preprocess_part(question.title)] * self.gamma
        else:
            texts += [self._preprocess_part(document.body)] * self.BODY_WEIGHT
            texts += [self._preprocess_part(document.title)] * self.gamma
        text = list(chain(*texts))
        return text


def get_preprocessor(text_format: TextFormat, questions: Iterable[Question], gamma: Gamma) -> Preprocessor:
    tokenizer = get_tokenizer(text_format)
    preprocessor = Preprocessor(text_format, tokenizer, questions, gamma)
    return preprocessor


def get_number_of_workers() -> Optional[int]:
    number_of_workers = None
    return number_of_workers


class BulkSearchSystem(System):
    def bulk_search(self, queries: Iterable[Query]) -> Iterable[Tuple[Query, Iterable[Answer]]]:
        for query in queries:
            answers = self.search(queries)
            yield (query, answers)


class JointBM25System(BulkSearchSystem, metaclass=ABCMeta):
    CURRENT_INSTANCE: Optional['JointBM25System'] = None

    def __init__(self, dictionary: Dictionary, preprocessor: Preprocessor, answers: Iterable[Answer]):
        self.dictionary = dictionary
        self.preprocessor = preprocessor
        self.bm25_model = LuceneBM25Model(dictionary=self.dictionary)
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
               similarity_matrix: Optional[SparseTermSimilarityMatrix]) -> BulkSearchSystem:
    _, gamma = parameters
    preprocessor = get_preprocessor(text_format, questions, gamma)
    if similarity_matrix is None:
        system = LuceneBM25System(dictionary, preprocessor, answers)
    else:
        system = SCMSystem(dictionary, similarity_matrix, preprocessor, answers)
    return system


def get_queries(run_type: RunType, output_text_format: TextFormat) -> Iterable[Query]:
    input_text_format = get_input_text_format(output_text_format)
    year = run_type.get_year()
    queries_dict = load_queries(input_text_format, year=year)
    queries = queries_dict.values()
    assert len(queries) == len(run_type)
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


def evaluate_serp_with_map(input_run_file: Path, queries: Iterable[Query], answers: Iterable[Answer],
                           run_type: RunType, output_map_file: Path) -> None:
    queries, answers = list(queries), list(answers)
    judgements = run_type.maybe_get_judgements(queries, answers)
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


def maybe_get_ndcg_and_interval(input_run_file: Path, run_type: RunType) -> Optional[NdcgAndInterval]:
    task = run_type.maybe_get_task()
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


def get_ndcg(input_run_file: Path, run_type: RunType) -> Ndcg:
    ndcg_and_interval = maybe_get_ndcg_and_interval(input_run_file, run_type)
    assert ndcg_and_interval is not None
    ndcg, _ = ndcg_and_interval
    return ndcg


def evaluate_serp_with_ndcg(input_run_file: Path, run_type: RunType, output_ndcg_file: Path) -> None:
    ndcg_and_interval = maybe_get_ndcg_and_interval(input_run_file, run_type)
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
                   parameters: Parameters) -> BulkSearchSystem:
    dictionary = get_dictionary(input_dictionary_file)
    similarity_matrix = maybe_get_term_similarity_matrix(input_similarity_matrix_file, parameters)
    questions, answers = get_questions_and_answers(msm_input_directory, output_text_format)
    questions, answers = list(questions), list(answers)
    system = get_system(output_text_format, questions, answers, dictionary, parameters,
                        similarity_matrix)
    return system


def get_optimal_parameters(msm_input_directory: Path,
                           output_text_format: TextFormat, input_dictionary_file: Path,
                           input_similarity_matrix_file: Optional[Path], run_name: str,
                           output_run_file: Path, temporary_output_parameter_file: Path,
                           output_parameter_file: Path) -> Parameters:
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

    run_type_2020 = RunType.ARQMATH_2020
    run_type_2021 = RunType.ARQMATH_2021

    queries_2020 = list(get_queries(run_type_2020, output_text_format))
    queries_2021 = list(get_queries(run_type_2021, output_text_format))

    for parameters in all_parameters:
        system = produce_system(
            msm_input_directory, output_text_format, input_dictionary_file,
            input_similarity_matrix_file, parameters)

        produce_serp(system, queries_2020, output_run_file, run_name)
        ndcg_2020 = get_ndcg(output_run_file, run_type_2020)
        produce_serp(system, queries_2021, output_run_file, run_name)
        ndcg_2021 = get_ndcg(output_run_file, run_type_2021)

        ndcg = ndcg_2020 * len(run_type_2020) + ndcg_2021 * len(run_type_2021)
        ndcg /= len(run_type_2020) + len(run_type_2021)

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_parameters = parameters

        alpha, gamma = parameters
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
         temporary_output_parameter_file: Path, output_parameter_file: Path) -> None:
    run_type = RunType.ARQMATH_2022

    if not output_run_file.exists():
        optimal_parameters = get_optimal_parameters(
            msm_input_directory, output_text_format, input_dictionary_file, input_similarity_matrix_file,
            run_name, temporary_output_run_file, temporary_output_parameter_file, output_parameter_file)

        temporary_output_run_file.unlink()

        system = produce_system(
            msm_input_directory, output_text_format, input_dictionary_file,
            input_similarity_matrix_file, optimal_parameters)

        queries = list(get_queries(run_type, output_text_format))
        produce_serp(system, queries, output_run_file, run_name)

    if not output_map_file.exists():
        questions, answers = get_questions_and_answers(msm_input_directory, output_text_format)
        questions, answers = list(questions), list(answers)
        evaluate_serp_with_map(output_run_file, queries, answers, run_type, output_map_file)

    if not output_ndcg_file.exists():
        evaluate_serp_with_ndcg(output_run_file, run_type, output_ndcg_file)


if __name__ == '__main__':
    setrecursionlimit(15000)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    assert len(argv) == 12

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

    main(msm_input_directory, text_format, input_dictionary_file, input_similarity_matrix_file,
         run_name, temporary_output_run_file, output_run_file, output_map_file, output_ndcg_file,
         temporary_output_parameter_file, output_parameter_file)
