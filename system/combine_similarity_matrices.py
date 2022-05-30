from pathlib import Path
from sys import argv
from typing import Iterable

from gensim.similarities import SparseTermSimilarityMatrix  # type: ignore


Alpha = float


def get_term_similarity_matrix(input_file: Path) -> SparseTermSimilarityMatrix:
    term_similarity_matrix = SparseTermSimilarityMatrix.load(str(input_file))
    return term_similarity_matrix


def combine_term_similarity_matrices(
            alpha: Alpha,
            levenshtein_term_similarity_matrix: SparseTermSimilarityMatrix,
            word_embedding_term_similarity_matrix: SparseTermSimilarityMatrix,
        ) -> SparseTermSimilarityMatrix:
    assert alpha >= 0.0
    assert alpha <= 1.0
    combined_term_similarity_matrix = SparseTermSimilarityMatrix(
        alpha * levenshtein_term_similarity_matrix.matrix +
        (1.0 - alpha) * word_embedding_term_similarity_matrix.matrix
    )
    return combined_term_similarity_matrix


def get_alphas() -> Iterable[Alpha]:
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return alphas


def get_combined_term_similarity_matrix_file(alpha: Alpha, output_directory: Path) -> Path:
    combined_term_similarity_matrix_filename = f'alpha={alpha}'
    combined_term_similarity_matrix_file = output_directory / combined_term_similarity_matrix_filename
    return combined_term_similarity_matrix_file


def get_combined_term_similarity_matrix(alpha: Alpha, output_directory: Path) -> SparseTermSimilarityMatrix:
    combined_term_similarity_matrix_file = get_combined_term_similarity_matrix_file(
        alpha, output_directory)
    combined_term_similarity_matrix = SparseTermSimilarityMatrix.load(
        str(combined_term_similarity_matrix_file))
    return combined_term_similarity_matrix


def combine_all_term_similarity_matrices(
            levenshtein_term_similarity_matrix: SparseTermSimilarityMatrix,
            word_embedding_term_similarity_matrix: SparseTermSimilarityMatrix,
            output_directory: Path,
        ) -> None:
    output_directory.mkdir(exist_ok=True)
    alphas = get_alphas()
    for alpha in alphas:
        combined_term_similarity_matrix = combine_term_similarity_matrices(
            alpha, levenshtein_term_similarity_matrix, word_embedding_term_similarity_matrix)
        combined_term_similarity_matrix_file = get_combined_term_similarity_matrix_file(
            alpha, output_directory)
        combined_term_similarity_matrix.save(str(combined_term_similarity_matrix_file))


def main(input_levenshtein_term_similarity_matrix_file: Path,
         input_word_embedding_term_similarity_matrix_file: Path, output_directory: Path) -> None:
    levenshtein_term_similarity_matrix = get_term_similarity_matrix(
        input_levenshtein_term_similarity_matrix_file)
    word_embedding_term_similarity_matrix = get_term_similarity_matrix(
        input_word_embedding_term_similarity_matrix_file)
    combine_all_term_similarity_matrices(
        levenshtein_term_similarity_matrix, word_embedding_term_similarity_matrix, output_directory)


if __name__ == '__main__':
    input_levenshtein_term_similarity_matrix_file = Path(argv[1])
    input_word_embedding_term_similarity_matrix_file = Path(argv[2])
    output_directory = Path(argv[3])
    main(input_levenshtein_term_similarity_matrix_file, input_word_embedding_term_similarity_matrix_file,
         output_directory)
