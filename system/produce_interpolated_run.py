from collections import defaultdict
from copy import deepcopy
import json
import logging
from pathlib import Path
from sys import argv, setrecursionlimit
from typing import Optional, Dict, Iterable, Tuple

from gensim.corpora import Dictionary  # type: ignore
from gensim.interfaces import TransformationABC as TermWeightTransformation  # type: ignore
from gensim.similarities import SparseTermSimilarityMatrix  # type: ignore
from gensim.utils import is_corpus  # type: ignore

from pv211_utils.arqmath.entities import (
    ArqmathQueryBase as Query,
    ArqmathAnswerBase as Answer,
    ArqmathQuestionBase as Question,
)
from scipy.sparse import dok_matrix
from tqdm import tqdm

from .produce_joint_run import (
    BulkSearchSystem,
    evaluate_serp_with_map,
    evaluate_serp_with_ndcg,
    get_bm25_model,
    get_dictionary,
    get_optimal_parameters,
    get_queries,
    get_questions_and_answers,
    get_ndcg,
    get_preprocessor,
    get_unparametrized_preprocessor,
    JointBM25System,
    maybe_get_term_similarity_matrix,
    Parameters,
    _produce_document_maps_corpus,
    produce_serp,
    produce_system,
    result_sort_key,
    Year,
)
from .prepare_msm_dataset import TextFormat


LOGGER = logging.getLogger(__name__)


Beta = float


def get_betas() -> Iterable[Beta]:
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return betas


class InterpolatedTermWeightTransformation(TermWeightTransformation):
    def __init__(self, previous_transformation: Optional[TermWeightTransformation],
                 first_dictionary: Dictionary, second_dictionary: Dictionary, beta: Beta):
        self.previous_transformation = previous_transformation
        self.first_dictionary = first_dictionary
        self.second_dictionary = second_dictionary
        self.beta = beta

    def __getitem__(self, weights):
        are_weights_corpus, weights = is_corpus(weights)
        if are_weights_corpus:
            return self._apply(weights)

        transformed_weights = weights
        if self.previous_transformation is not None:
            transformed_weights = self.previous_transformation[transformed_weights]
        transformed_weights = [
            (term_id, weight * (self.beta if term_id in self.first_dictionary else 1.0 - self.beta))
            for term_id, weight
            in transformed_weights
        ]

        return transformed_weights


class InterpolatedBM25System(BulkSearchSystem):
    FIRST_COEFFICIENT = 0.5
    SECOND_COEFFICIENT = 0.5

    def __init__(self, first_system: JointBM25System, second_system: JointBM25System,
                 first_queries: Iterable[Query], second_queries: Iterable[Query],
                 beta: Beta):
        self.first_system = first_system
        self.second_system = second_system
        self.beta = beta

        second_queries_by_id = {query.query_id: query for query in second_queries}
        self.first_queries_to_second_queries: Dict[Query, Query] = {
            first_query: second_queries_by_id[first_query.query_id]
            for first_query
            in first_queries
        }

    def search(self, first_query: Query) -> Iterable[Answer]:
        second_query = self.first_queries_to_second_queries[first_query]
        first_similarities = self.first_system.get_similarities(first_query)
        second_similarities = self.second_system.get_similarities(second_query)

        interpolated_similarities = defaultdict(lambda: 0.0)
        for answer, similarity in first_similarities.items():
            interpolated_similarities[answer] += self.beta * similarity
        for answer, similarity in second_similarities.items():
            interpolated_similarities[answer] += (1.0 - self.beta) * similarity

        for answer, _ in sorted(interpolated_similarities.items(), key=result_sort_key):
            yield answer

    def bulk_search(self, first_queries: Iterable[Query]) -> Iterable[Tuple[Query, Iterable[Answer]]]:
        first_queries = list(first_queries)
        second_queries = [
            self.first_queries_to_second_queries[first_query]
            for first_query
            in first_queries
        ]

        first_bulk_similarities = dict(self.first_system.get_bulk_similarities(first_queries))
        second_bulk_similarities = dict(self.second_system.get_bulk_similarities(second_queries))

        for query in first_queries:
            first_similarities = first_bulk_similarities[query]
            second_similarities = second_bulk_similarities[query]
            interpolated_similarities = defaultdict(lambda: 0.0)
            for answer, similarity in first_similarities.items():
                interpolated_similarities[answer] += self.beta * similarity
            for answer, similarity in second_similarities.items():
                interpolated_similarities[answer] += (1.0 - self.beta) * similarity

            answers = (answer for answer, _ in sorted(interpolated_similarities.items(), key=result_sort_key))
            yield (query, answers)


def interpolate_text_formats(first_text_format: TextFormat,
                             second_text_format: TextFormat) -> TextFormat:
    assert first_text_format == 'text'
    assert second_text_format in ('latex', 'tangentl')
    interpolated_text_format = f'{first_text_format}+{second_text_format}'
    return interpolated_text_format


def interpolate_similarity_matrices(first_dictionary: Dictionary, second_dictionary: Dictionary,
                                    maybe_first_similarity_matrix: Optional[SparseTermSimilarityMatrix],
                                    maybe_second_similarity_matrix: Optional[SparseTermSimilarityMatrix],
                                    dtype: type = float) -> Tuple[Dictionary, SparseTermSimilarityMatrix]:
    dictionary = deepcopy(first_dictionary)
    dictionary_transformer = dictionary.merge_with(second_dictionary)

    shape = (len(dictionary), len(dictionary))
    similarity_matrix = dok_matrix(shape, dtype)

    if maybe_first_similarity_matrix is not None:
        first_similarity_matrix = dok_matrix(maybe_first_similarity_matrix.matrix)
        term1_ids, term2_ids = first_similarity_matrix.nonzero()
        term1_ids, term2_ids = map(int, term1_ids), map(int, term2_ids)
        for term1_id, term2_id in zip(term1_ids, term2_ids):
            first_word_similarity = first_similarity_matrix[term1_id, term2_id]
            first_word_similarity = float(first_word_similarity)
            similarity_matrix[term1_id, term2_id] = first_word_similarity

    if maybe_second_similarity_matrix is not None:
        second_similarity_matrix = dok_matrix(maybe_second_similarity_matrix.matrix)
        old_term1_ids, old_term2_ids = second_similarity_matrix.nonzero()
        old_term1_ids, old_term2_ids = map(int, old_term1_ids), map(int, old_term2_ids)
        for old_term1_id, old_term2_id in zip(old_term1_ids, old_term2_ids):
            second_word_similarity = second_similarity_matrix[old_term1_id, old_term2_id]
            second_word_similarity = float(second_word_similarity)
            term1_id = dictionary_transformer.old2new[old_term1_id]
            term2_id = dictionary_transformer.old2new[old_term2_id]
            similarity_matrix[term1_id, term2_id] = second_word_similarity

    similarity_matrix = SparseTermSimilarityMatrix(similarity_matrix)

    return (dictionary, similarity_matrix)


def produce_document_maps_corpus(input_run_file: Path,
                                 queries: Iterable[Query],
                                 answers: Iterable[Answer],
                                 first_input_dictionary_file: Path,
                                 second_input_dictionary_file: Path,
                                 first_input_similarity_matrix_file: Optional[Path],
                                 second_input_similarity_matrix_file: Optional[Path],
                                 first_parameters: Parameters, second_parameters: Parameters,
                                 first_text_format: TextFormat, second_text_format: TextFormat,
                                 questions: Iterable[Question],
                                 output_document_maps_file: Path, beta: Beta) -> None:
    first_dictionary = get_dictionary(first_input_dictionary_file)
    second_dictionary = get_dictionary(second_input_dictionary_file)

    maybe_first_similarity_matrix = maybe_get_term_similarity_matrix(
        first_input_similarity_matrix_file, first_parameters)
    maybe_second_similarity_matrix = maybe_get_term_similarity_matrix(
        second_input_similarity_matrix_file, second_parameters)
    assert maybe_first_similarity_matrix is not None or maybe_second_similarity_matrix is not None

    dictionary, similarity_matrix = interpolate_similarity_matrices(
        first_dictionary, second_dictionary, maybe_first_similarity_matrix, maybe_second_similarity_matrix)

    _, first_gamma = first_parameters
    _, second_gamma = second_parameters
    gamma = max(first_gamma, second_gamma)

    bm25_model = get_bm25_model(dictionary)
    query_term_weight_transformer = InterpolatedTermWeightTransformation(
        None, first_dictionary, second_dictionary, beta)
    answer_term_weight_transformer = InterpolatedTermWeightTransformation(
        bm25_model, first_dictionary, second_dictionary, beta)

    text_format = interpolate_text_formats(first_text_format, second_text_format)
    unparametrized_preprocessor = get_unparametrized_preprocessor(text_format, questions)
    preprocessor = get_preprocessor(text_format, questions, gamma)

    _produce_document_maps_corpus(output_run_file, queries, answers,
                                  dictionary, similarity_matrix,
                                  unparametrized_preprocessor, preprocessor,
                                  output_document_maps_file,
                                  query_term_weight_transformer, answer_term_weight_transformer)


def get_interpolated_system(first_system: JointBM25System, second_system: JointBM25System,
                            first_queries: Iterable[Query], second_queries: Iterable[Query],
                            beta: Beta) -> InterpolatedBM25System:
    interpolated_system = InterpolatedBM25System(
        first_system, second_system, first_queries, second_queries, beta)
    return interpolated_system


def get_optimal_beta(first_system: JointBM25System, second_system: JointBM25System,
                     first_output_text_format: TextFormat, second_output_text_format: TextFormat,
                     run_name: str, output_run_file: Path, temporary_output_beta_file: Path,
                     output_beta_file: Path) -> Beta:
    betas = get_betas()
    betas = sorted(betas)

    try:
        with output_beta_file.open('rt') as f:
            obj = json.load(f)
        best_beta = obj['best_beta']
        LOGGER.info(f'Loaded optimal beta from {output_beta_file}')
        return best_beta
    except IOError:
        try:
            with temporary_output_beta_file.open('rt') as f:
                obj = json.load(f)
            best_ndcg, best_beta, beta = obj['best_ndcg'], obj['best_beta'], obj['beta']
            assert beta in betas
            betas = betas[betas.index(beta) + 1:]
            LOGGER.info('Fast-forwarded optimization of beta to last tested value '
                        f'from {temporary_output_beta_file}')
        except IOError:
            best_ndcg, best_beta = float('-inf'), None

    betas = tqdm(betas, desc='Optimizing beta')

    year_2020 = Year.from_int(2020)
    year_2021 = Year.from_int(2021)

    first_queries_2020 = list(get_queries(year_2020, first_output_text_format))
    first_queries_2021 = list(get_queries(year_2021, first_output_text_format))

    second_queries_2020 = list(get_queries(year_2020, second_output_text_format))
    second_queries_2021 = list(get_queries(year_2021, second_output_text_format))

    for beta in betas:
        system_2020 = get_interpolated_system(
            first_system, second_system, first_queries_2020, second_queries_2020, beta)
        produce_serp(system_2020, first_queries_2020, output_run_file, run_name)
        ndcg_2020 = get_ndcg(output_run_file, year_2020)

        system_2021 = get_interpolated_system(
            first_system, second_system, first_queries_2021, second_queries_2021, beta)
        produce_serp(system_2021, first_queries_2021, output_run_file, run_name)
        ndcg_2021 = get_ndcg(output_run_file, year_2021)

        ndcg = ndcg_2020 * len(year_2020) + ndcg_2021 * len(year_2021)
        ndcg /= len(year_2020) + len(year_2021)

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_beta = beta

        with temporary_output_beta_file.open('wt') as f:
            json.dump({'beta': beta, 'best_beta': best_beta, 'best_ndcg': best_ndcg}, f)

    assert best_beta is not None

    with output_beta_file.open('wt') as f:
        json.dump({'best_beta': best_beta, 'best_ndcg': best_ndcg}, f)

    temporary_output_beta_file.unlink()

    return best_beta


def main(msm_input_directory: Path,
         first_output_text_format: TextFormat, first_input_dictionary_file: Path,
         first_input_similarity_matrix_file: Optional[Path],
         first_temporary_output_parameter_file: Path, first_output_parameter_file: Path,
         second_output_text_format: TextFormat, second_input_dictionary_file: Path,
         second_input_similarity_matrix_file: Optional[Path],
         second_temporary_output_parameter_file: Path, second_output_parameter_file: Path,
         run_name: str, temporary_output_run_file: Path, output_run_file: Path, output_map_file: Path,
         output_ndcg_file: Path, temporary_output_beta_file: Path, output_beta_file: Path,
         lock_file: Path, output_document_maps_file: Path) -> None:
    year = Year.from_int(2022)

    first_queries, second_queries = None, None

    def ensure_queries() -> None:
        nonlocal first_queries, second_queries
        if any([first_queries is None, second_queries is None]):
            first_queries = list(get_queries(year, first_output_text_format))
            second_queries = list(get_queries(year, second_output_text_format))

    first_optimal_parameters, second_optimal_parameters, optimal_beta = None, None, None
    first_system, second_system = None, None

    def ensure_optimal_parameters() -> None:
        nonlocal first_optimal_parameters, second_optimal_parameters, optimal_beta
        nonlocal first_system, second_system
        if any([
                    first_optimal_parameters is None,
                    second_optimal_parameters is None,
                    optimal_beta is None,
                ]):
            first_optimal_parameters = get_optimal_parameters(
                msm_input_directory, first_output_text_format,
                first_input_dictionary_file, first_input_similarity_matrix_file,
                run_name, temporary_output_run_file,
                first_temporary_output_parameter_file, first_output_parameter_file,
                lock_file)

            second_optimal_parameters = get_optimal_parameters(
                msm_input_directory, second_output_text_format,
                second_input_dictionary_file, second_input_similarity_matrix_file,
                run_name, temporary_output_run_file,
                second_temporary_output_parameter_file, second_output_parameter_file,
                lock_file)

            first_system = produce_system(
                msm_input_directory, first_output_text_format,
                first_input_dictionary_file, first_input_similarity_matrix_file,
                first_optimal_parameters, lock_file)

            second_system = produce_system(
                msm_input_directory, second_output_text_format,
                second_input_dictionary_file, second_input_similarity_matrix_file,
                second_optimal_parameters, lock_file)

            optimal_beta = get_optimal_beta(
                first_system, second_system, first_output_text_format,
                second_output_text_format, run_name, temporary_output_run_file,
                temporary_output_beta_file, output_beta_file)

            try:
                temporary_output_run_file.unlink()
            except FileNotFoundError:
                pass

    if not output_run_file.exists():
        ensure_optimal_parameters()
        ensure_queries()
        system = get_interpolated_system(first_system, second_system, first_queries, second_queries, optimal_beta)
        produce_serp(system, first_queries, output_run_file, run_name)

    if not output_map_file.exists():
        first_questions, first_answers = get_questions_and_answers(msm_input_directory, first_output_text_format)
        first_questions, first_answers = list(first_questions), list(first_answers)
        ensure_queries()
        evaluate_serp_with_map(output_run_file, first_queries, first_answers, year, output_map_file)

    if not output_ndcg_file.exists():
        ensure_optimal_parameters()
        evaluate_serp_with_ndcg(output_run_file, year, output_ndcg_file)

    if (first_input_similarity_matrix_file is not None or second_input_similarity_matrix_file is not None) \
            and not output_document_maps_file.exists():
        ensure_optimal_parameters()
        output_text_format = interpolate_text_formats(first_output_text_format, second_output_text_format)
        queries = list(get_queries(year, output_text_format))
        questions, answers = get_questions_and_answers(msm_input_directory, output_text_format)
        questions, answers = list(questions), list(answers)
        produce_document_maps_corpus(output_run_file, queries, answers,
                                     first_input_dictionary_file, second_input_dictionary_file,
                                     first_input_similarity_matrix_file,
                                     second_input_similarity_matrix_file,
                                     first_optimal_parameters, second_optimal_parameters,
                                     first_output_text_format, second_output_text_format,
                                     questions, output_document_maps_file, optimal_beta)


if __name__ == '__main__':
    setrecursionlimit(15000)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    assert len(argv) == 21

    msm_input_directory = Path(argv[1])
    first_text_format = argv[2]
    first_input_dictionary_file = Path(argv[3])
    first_input_similarity_matrix_file = Path(argv[4]) if argv[4] != 'none' else None
    first_temporary_output_parameter_file = Path(argv[5])
    first_output_parameter_file = Path(argv[6])
    second_text_format = argv[7]
    second_input_dictionary_file = Path(argv[8])
    second_input_similarity_matrix_file = Path(argv[9]) if argv[9] != 'none' else None
    second_temporary_output_parameter_file = Path(argv[10])
    second_output_parameter_file = Path(argv[11])
    run_name = argv[12]
    temporary_output_run_file = Path(argv[13])
    output_run_file = Path(argv[14])
    output_map_file = Path(argv[15])
    output_ndcg_file = Path(argv[16])
    temporary_output_beta_file = Path(argv[17])
    output_beta_file = Path(argv[18])
    lock_file = Path(argv[19])
    output_document_maps_file = Path(argv[20])

    main(msm_input_directory, first_text_format, first_input_dictionary_file,
         first_input_similarity_matrix_file,
         first_temporary_output_parameter_file, first_output_parameter_file,
         second_text_format, second_input_dictionary_file,
         second_input_similarity_matrix_file,
         second_temporary_output_parameter_file, second_output_parameter_file,
         run_name, temporary_output_run_file, output_run_file, output_map_file,
         output_ndcg_file, temporary_output_beta_file, output_beta_file,
         lock_file, output_document_maps_file)
