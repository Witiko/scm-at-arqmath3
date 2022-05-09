from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
import csv
from datetime import datetime
from itertools import chain
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from sys import argv, setrecursionlimit
from typing import Optional, Iterable, List, Dict, Tuple, Union

from arqmath_eval import get_ndcg
from gensim.corpora import Dictionary  # type: ignore
from gensim.models import LuceneBM25Model  # type: ignore
from gensim.similarities import (
    SparseMatrixSimilarity,  # type: ignore
    SoftCosineSimilarity,  # type: ignore
    SparseTermSimilarityMatrix,  # type: ignore
)
from lxml.html import fromstring
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
    TextFormat,
    get_questions_and_answers,
    get_input_text_format,
    read_document_text,
    read_document_text_latex,
    read_document_latex,
    read_document_tangentl,
)
from .combine_similarity_matrices import get_term_similarity_matrix
from .prepare_levenshtein_similarity_matrix import get_dictionary


AnswerIndex = int

TermId = int
Weight = float
TermWeight = Tuple[TermId, Weight]
Vector = List[TermWeight]

Similarity = float

SimilarityIndex = Union[SparseMatrixSimilarity, SoftCosineSimilarity]

RunType = str

Task = str

Token = str
Text = List[Token]
Line = str


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


class Preprocessor:
    BODY_WEIGHT = 1
    TITLE_WEIGHT = 5

    def __init__(self, text_format: TextFormat, tokenizer: Tokenizer, questions: Iterable[Question]):
        self.text_format = text_format
        self.tokenizer = tokenizer
        self.answer_to_question = {
            answer: question
            for question in questions
            for answer in question.answers
        }

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
                texts += [self._preprocess_part(question.title)] * self.TITLE_WEIGHT
        else:
            texts += [self._preprocess_part(document.body)] * self.BODY_WEIGHT
            texts += [self._preprocess_part(document.title)] * self.TITLE_WEIGHT
        text = list(chain(*texts))
        return text


def get_preprocessor(text_format: TextFormat, questions: Iterable[Question]) -> Preprocessor:
    tokenizer = get_tokenizer(text_format)
    preprocessor = Preprocessor(text_format, tokenizer, questions)
    return preprocessor


class TimedSystem(System):
    def __init__(self, system: System):
        self.system = system
        self.durations = dict()

    def search(self, query: Query) -> Iterable[Answer]:
        before = datetime.now()
        results = list(self.system.search(query))
        after = datetime.now()
        duration = after - before
        self.durations[query] = duration.total_seconds()
        return results

    def __str__(self) -> str:
        assert self.durations
        mean_duration = mean(self.durations.values())
        min_query, min_duration = None, float('inf')
        max_query, max_duration = None, float('-inf')
        for query, duration in self.durations.items():
            if duration < min_duration:
                min_query, min_duration = query, duration
            if duration > max_duration:
                max_query, max_duration = query, duration
        assert min_query is not None
        assert max_query is not None
        return '\n'.join([
            f'Mean duration: {mean_duration:.2f}',
            f'Shortest duration: {min_duration:.2f} (A.{min_query.query_id})',
            f'Longest duration: {max_duration:.2f} (A.{max_query.query_id})',
        ])


def get_number_of_workers() -> Optional[int]:
    number_of_workers = None
    return number_of_workers


class JointBM25System(System, metaclass=ABCMeta):
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
        query_vector = self._document_to_vector(query)
        similarity_index = self.get_similarity_index()
        similarities = {
            self.index_to_answer[answer_index]: similarity
            for answer_index, similarity
            in enumerate(similarity_index[query_vector])
        }
        return similarities

    def search(self, query: Query) -> Iterable[Answer]:
        similarities = self.get_similarities(query)
        for answer, _ in sorted(similarities.items(), key=result_sort_key):
            yield answer


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
               dictionary: Dictionary, similarity_matrix: Optional[SparseTermSimilarityMatrix]) -> System:
    preprocessor = get_preprocessor(text_format, questions)
    if similarity_matrix is None:
        system = LuceneBM25System(dictionary, preprocessor, answers)
    else:
        system = SCMSystem(dictionary, similarity_matrix, preprocessor, answers)
    return system


def get_queries(run_type: RunType, text_format: TextFormat) -> Iterable[Query]:
    if run_type == 'submission2020':
        queries = load_queries(text_format, year=2020)
    elif run_type == 'submission2021':
        queries = load_queries(text_format, year=2021)
    elif run_type == 'submission2022':
        queries = load_queries(text_format, year=2022)
    else:
        raise ValueError(f'Unknown run type {run_type}')
    return queries.values()


def maybe_get_judgements(run_type: RunType, queries: Iterable[Query],
                         answers: Iterable[Answer]) -> Optional[Judgements]:
    if run_type == 'submission2020' or run_type == 'submission2021':
        queries_dict = OrderedDict()
        for query in queries:
            queries_dict[query.query_id] = query
        answers_dict = OrderedDict()
        for answer in answers:
            answers_dict[answer.document_id] = answer
        if run_type == 'submission2020':
            judgements = load_judgements(queries_dict, answers_dict, year=2020)
        elif run_type == 'submission2021':
            judgements = load_judgements(queries_dict, answers_dict, year=2021)
    elif run_type == 'submission2022':
        judgements = None
    else:
        raise ValueError(f'Unknown run type {run_type}')
    return judgements


def maybe_get_task(run_type: RunType) -> Optional[Task]:
    if run_type == 'submission2020':
        task = 'task1-2020'
    elif run_type == 'submission2021':
        task = 'task1-2021'
    elif run_type == 'submission2022':
        task = None
    else:
        raise ValueError(f'Unknown run type {run_type}')
    return task


def maybe_get_term_similarity_matrix(input_file: Optional[Path]) -> Optional[SparseTermSimilarityMatrix]:
    if input_file is not None:
        term_similarity_matrix = get_term_similarity_matrix(input_file)
    else:
        term_similarity_matrix = None
    return term_similarity_matrix


def get_topn() -> int:
    topn = 1000
    return topn


def produce_serp(system: System, queries: Iterable[Query], output_run_file: Path,
                 output_timer_file: Path, run_name: str) -> None:
    queries = list(queries)
    timed_system = TimedSystem(system)
    topn = get_topn()

    output_run_file.parent.mkdir(exist_ok=True)
    with output_run_file.open('wt') as f:
        for query in tqdm(queries, desc='Querying the system'):
            answers = timed_system.search(query)
            for rank, answer in enumerate(answers):
                rank = rank + 1
                if rank > topn:
                    break
                score = 1.0 / float(rank)
                query_id = f'A.{query.query_id}'
                answer_id = answer.document_id
                line = f'{query_id}\t{answer_id}\t{rank}\t{score}\t{run_name}'
                print(line, file=f)

    output_timer_file.parent.mkdir(exist_ok=True)
    with output_timer_file.open('wt') as f:
        print(timed_system, file=f)


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
                           judgements: Optional[Judgements], output_map_file: Path) -> None:
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


def evaluate_serp_with_ndcg(input_run_file: Path, run_type: RunType, output_ndcg_file: Path) -> None:
    task = maybe_get_task(run_type)
    if task is None:
        return

    parsed_result = defaultdict(lambda: dict())
    for query_id, answer_id, rank, *_ in read_tsv_file(input_run_file):
        score = 1.0 / rank
        parsed_result[query_id][answer_id] = score
    confidence = get_confidence()
    ndcg_score, interval = get_ndcg(parsed_result, task, 'all', confidence=confidence)
    lower_bound, upper_bound = interval

    output_ndcg_file.parent.mkdir(exist_ok=True)
    with output_ndcg_file.open('wt') as f:
        print(f'{ndcg_score:.3f}, {confidence:g}% CI: [{lower_bound:.3f}; {upper_bound:.3f}]', file=f)


def main(run_type: RunType, msm_input_directory: Path, output_text_format: TextFormat,
         input_dictionary_file: Path, input_similarity_matrix_file: Optional[Path], run_name: str,
         output_run_file: Path, output_timer_file: Path, output_map_file: Path,
         output_ndcg_file: Path) -> None:

    input_text_format = get_input_text_format(output_text_format)
    dictionary = get_dictionary(input_dictionary_file)
    similarity_matrix = maybe_get_term_similarity_matrix(input_similarity_matrix_file)
    questions, answers = get_questions_and_answers(msm_input_directory, input_text_format)
    questions, answers = list(questions), list(answers)
    system = get_system(output_text_format, questions, answers, dictionary, similarity_matrix)

    queries = list(get_queries(run_type, input_text_format))
    produce_serp(system, queries, output_run_file, output_timer_file, run_name)

    judgements = maybe_get_judgements(run_type, queries, answers)
    evaluate_serp_with_map(output_run_file, queries, answers, judgements, output_map_file)
    evaluate_serp_with_ndcg(output_run_file, run_type, output_ndcg_file)


if __name__ == '__main__':
    setrecursionlimit(15000)

    run_type = argv[1]
    msm_input_directory = Path(argv[2])
    text_format = argv[3]
    input_dictionary_file = Path(argv[4])
    input_similarity_matrix_file = Path(argv[5]) if argv[5] != 'none' else None
    run_name = argv[6]
    output_run_file = Path(argv[7])
    output_timer_file = Path(argv[8])
    output_map_file = Path(argv[9])
    output_ndcg_file = Path(argv[10])

    main(run_type, msm_input_directory, text_format, input_dictionary_file,
         input_similarity_matrix_file, run_name, output_run_file, output_timer_file,
         output_map_file, output_ndcg_file)
