from collections import defaultdict
import logging
from pathlib import Path
from sys import argv, setrecursionlimit
from typing import Optional, Dict, Iterable, Tuple

from pv211_utils.arqmath.entities import (
    ArqmathQueryBase as Query,
    ArqmathAnswerBase as Answer,
)

from .produce_joint_run import (
    BulkSearchSystem,
    evaluate_serp_with_map,
    evaluate_serp_with_ndcg,
    get_dictionary,
    get_queries,
    get_system,
    JointBM25System,
    maybe_get_judgements,
    maybe_get_term_similarity_matrix,
    produce_serp,
    result_sort_key,
    RunType,
)
from .prepare_msm_dataset import get_questions_and_answers, get_input_text_format, TextFormat


class InterpolatedBM25System(BulkSearchSystem):
    FIRST_COEFFICIENT = 0.5
    SECOND_COEFFICIENT = 0.5

    def __init__(self, first_system: JointBM25System, second_system: JointBM25System,
                 first_queries: Iterable[Query], second_queries: Iterable[Query]):
        self.first_system = first_system
        self.second_system = second_system

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
            interpolated_similarities[answer] += self.FIRST_COEFFICIENT * similarity
        for answer, similarity in second_similarities.items():
            interpolated_similarities[answer] += self.SECOND_COEFFICIENT * similarity

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
                interpolated_similarities[answer] += self.FIRST_COEFFICIENT * similarity
            for answer, similarity in second_similarities.items():
                interpolated_similarities[answer] += self.SECOND_COEFFICIENT * similarity

            answers = (answer for answer, _ in sorted(interpolated_similarities.items(), key=result_sort_key))
            yield (query, answers)


def get_interpolated_system(first_system: JointBM25System, second_system: JointBM25System,
                            first_queries: Iterable[Query], second_queries: Iterable[Query]) -> InterpolatedBM25System:
    interpolated_system = InterpolatedBM25System(first_system, second_system, first_queries, second_queries)
    return interpolated_system


def main(run_type: RunType, msm_input_directory: Path,
         first_output_text_format: TextFormat, first_input_dictionary_file: Path,
         first_input_similarity_matrix_file: Optional[Path],
         second_output_text_format: TextFormat, second_input_dictionary_file: Path,
         second_input_similarity_matrix_file: Optional[Path],
         run_name: str, output_run_file: Path, output_map_file: Path, output_ndcg_file: Path) -> None:

    first_input_text_format = get_input_text_format(first_output_text_format)
    first_dictionary = get_dictionary(first_input_dictionary_file)
    first_similarity_matrix = maybe_get_term_similarity_matrix(first_input_similarity_matrix_file)
    first_questions, first_answers = get_questions_and_answers(msm_input_directory, first_input_text_format)
    first_questions, first_answers = list(first_questions), list(first_answers)
    first_system = get_system(first_output_text_format, first_questions, first_answers,
                              first_dictionary, first_similarity_matrix)

    second_input_text_format = get_input_text_format(second_output_text_format)
    second_dictionary = get_dictionary(second_input_dictionary_file)
    second_similarity_matrix = maybe_get_term_similarity_matrix(second_input_similarity_matrix_file)
    second_questions, second_answers = get_questions_and_answers(msm_input_directory, second_input_text_format)
    second_questions, second_answers = list(second_questions), list(second_answers)
    second_system = get_system(second_output_text_format, second_questions, second_answers,
                               second_dictionary, second_similarity_matrix)

    first_queries = list(get_queries(run_type, first_input_text_format))
    second_queries = list(get_queries(run_type, second_input_text_format))
    system = get_interpolated_system(first_system, second_system, first_queries, second_queries)
    produce_serp(system, first_queries, output_run_file, run_name)

    queries, answers = first_queries, first_answers
    judgements = maybe_get_judgements(run_type, queries, answers)
    evaluate_serp_with_map(output_run_file, first_queries, first_answers, judgements, output_map_file)
    evaluate_serp_with_ndcg(output_run_file, run_type, output_ndcg_file)


if __name__ == '__main__':
    setrecursionlimit(15000)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    run_type = argv[1]
    msm_input_directory = Path(argv[2])
    first_text_format = argv[3]
    first_input_dictionary_file = Path(argv[4])
    first_input_similarity_matrix_file = Path(argv[5]) if argv[5] != 'none' else None
    second_text_format = argv[6]
    second_input_dictionary_file = Path(argv[7])
    second_input_similarity_matrix_file = Path(argv[8]) if argv[8] != 'none' else None
    run_name = argv[9]
    output_run_file = Path(argv[10])
    output_map_file = Path(argv[11])
    output_ndcg_file = Path(argv[12])

    main(run_type, msm_input_directory, first_text_format, first_input_dictionary_file,
         first_input_similarity_matrix_file, second_text_format, second_input_dictionary_file,
         second_input_similarity_matrix_file, run_name, output_run_file, output_map_file,
         output_ndcg_file)
