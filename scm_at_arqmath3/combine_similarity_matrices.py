from pathlib import Path
from sys import argv

from gensim.similarities import SparseTermSimilarityMatrix  # type: ignore


def get_term_similarity_matrix(input_file: Path) -> SparseTermSimilarityMatrix:
    term_similarity_matrix = SparseTermSimilarityMatrix.load(str(input_file))
    return term_similarity_matrix


def combine_term_similarity_matrices(
            levenshtein_term_similarity_matrix: SparseTermSimilarityMatrix,
            word_embedding_term_similarity_matrix: SparseTermSimilarityMatrix,
        ) -> SparseTermSimilarityMatrix:
    combined_term_similarity_matrix = SparseTermSimilarityMatrix(
        0.9 * levenshtein_term_similarity_matrix.matrix +
        0.1 * word_embedding_term_similarity_matrix.matrix
    )
    return combined_term_similarity_matrix


def main(input_levenshtein_term_similarity_matrix_file: Path,
         input_word_embedding_term_similarity_matrix_file: Path, output_file: Path) -> None:
    levenshtein_term_similarity_matrix = get_term_similarity_matrix(
        input_levenshtein_term_similarity_matrix_file)
    word_embedding_term_similarity_matrix = get_term_similarity_matrix(
        input_word_embedding_term_similarity_matrix_file)
    combined_term_similarity_matrix = combine_term_similarity_matrices(
        levenshtein_term_similarity_matrix, word_embedding_term_similarity_matrix)
    combined_term_similarity_matrix.save(str(output_file))


if __name__ == '__main__':
    input_levenshtein_term_similarity_matrix_file = Path(argv[1])
    input_word_embedding_term_similarity_matrix_file = Path(argv[2])
    output_file = Path(argv[3])
    main(input_levenshtein_term_similarity_matrix_file, input_word_embedding_term_similarity_matrix_file, output_file)
