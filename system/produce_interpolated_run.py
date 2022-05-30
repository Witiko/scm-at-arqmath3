from collections import defaultdict
import json
import logging
from pathlib import Path
from sys import argv, setrecursionlimit
from typing import Optional, Dict, Iterable, Tuple

from pv211_utils.arqmath.entities import (
    ArqmathQueryBase as Query,
    ArqmathAnswerBase as Answer,
)
from tqdm import tqdm

from .produce_joint_run import (
    BulkSearchSystem,
    evaluate_serp_with_map,
    evaluate_serp_with_ndcg,
    get_optimal_parameters,
    get_queries,
    get_questions_and_answers,
    get_ndcg,
    JointBM25System,
    produce_serp,
    produce_system,
    result_sort_key,
    RunType,
)
from .prepare_msm_dataset import TextFormat


LOGGER = logging.getLogger(__name__)


Beta = float


def get_betas() -> Iterable[Beta]:
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return betas


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

    run_type_2020 = RunType.ARQMATH_2020
    run_type_2021 = RunType.ARQMATH_2021

    first_queries_2020 = list(get_queries(run_type_2020, first_output_text_format))
    first_queries_2021 = list(get_queries(run_type_2021, first_output_text_format))

    second_queries_2020 = list(get_queries(run_type_2020, second_output_text_format))
    second_queries_2021 = list(get_queries(run_type_2021, second_output_text_format))

    for beta in betas:
        system_2020 = get_interpolated_system(
            first_system, second_system, first_queries_2020, second_queries_2020, beta)
        produce_serp(system_2020, first_queries_2020, output_run_file, run_name)
        ndcg_2020 = get_ndcg(output_run_file, run_type_2020)

        system_2021 = get_interpolated_system(
            first_system, second_system, first_queries_2021, second_queries_2021, beta)
        produce_serp(system_2021, first_queries_2021, output_run_file, run_name)
        ndcg_2021 = get_ndcg(output_run_file, run_type_2021)

        ndcg = ndcg_2020 * len(run_type_2020) + ndcg_2021 * len(run_type_2021)
        ndcg /= len(run_type_2020) + len(run_type_2021)

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
         output_ndcg_file: Path, temporary_output_beta_file: Path, output_beta_file: Path) -> None:
    first_optimal_parameters = get_optimal_parameters(
        msm_input_directory, first_output_text_format,
        first_input_dictionary_file, first_input_similarity_matrix_file,
        run_name, temporary_output_run_file,
        first_temporary_output_parameter_file, first_output_parameter_file)
    first_system = produce_system(
        msm_input_directory, first_output_text_format,
        first_input_dictionary_file, first_input_similarity_matrix_file,
        first_optimal_parameters, silent=False)

    second_optimal_parameters = get_optimal_parameters(
        msm_input_directory, second_output_text_format,
        second_input_dictionary_file, second_input_similarity_matrix_file,
        run_name, temporary_output_run_file,
        second_temporary_output_parameter_file, second_output_parameter_file)
    second_system = produce_system(
        msm_input_directory, second_output_text_format,
        second_input_dictionary_file, second_input_similarity_matrix_file,
        second_optimal_parameters, silent=False)

    optimal_beta = get_optimal_beta(
        first_system, second_system, first_output_text_format,
        second_output_text_format, run_name, temporary_output_run_file,
        temporary_output_beta_file, output_beta_file)

    temporary_output_run_file.unlink()

    run_type = RunType.ARQMATH_2022
    first_queries = list(get_queries(run_type, first_output_text_format))
    second_queries = list(get_queries(run_type, second_output_text_format))
    system = get_interpolated_system(first_system, second_system, first_queries, second_queries, optimal_beta)
    produce_serp(system, first_queries, output_run_file, run_name)

    first_questions, first_answers = get_questions_and_answers(msm_input_directory, first_output_text_format)
    first_questions, first_answers = list(first_questions), list(first_answers)
    evaluate_serp_with_map(output_run_file, first_queries, first_answers, run_type, output_map_file)
    evaluate_serp_with_ndcg(output_run_file, run_type, output_ndcg_file)


if __name__ == '__main__':
    setrecursionlimit(15000)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    assert len(argv) == 19

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

    main(msm_input_directory, first_text_format, first_input_dictionary_file,
         first_input_similarity_matrix_file,
         first_temporary_output_parameter_file, first_output_parameter_file,
         second_text_format, second_input_dictionary_file,
         second_input_similarity_matrix_file,
         second_temporary_output_parameter_file, second_output_parameter_file,
         run_name, temporary_output_run_file, output_run_file, output_map_file,
         output_ndcg_file, temporary_output_beta_file, output_beta_file)
