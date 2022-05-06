from pathlib import Path
from sys import argv

from gensim.corpora import Dictionary  # type: ignore
from gensim.models import TfidfModel  # type: ignore
from gensim.similarities import LevenshteinSimilarityIndex, SparseTermSimilarityMatrix  # type: ignore


def get_dictionary(input_file: Path) -> Dictionary:
    dictionary = Dictionary.load(input_file)
    return dictionary


def get_tfidf_model(dictionary: Dictionary) -> TfidfModel:
    tfidf_model = TfidfModel(dictionary=dictionary)
    return tfidf_model


def get_term_similarity_index(dictionary: Dictionary) -> LevenshteinSimilarityIndex:
    term_similarity_index = LevenshteinSimilarityIndex(dictionary, alpha=1.8, beta=5.0, max_distance=3)
    return term_similarity_index


def get_term_similarity_matrix(input_file: Path) -> SparseTermSimilarityMatrix:
    dictionary = get_dictionary(input_file)
    tfidf_model = get_tfidf_model(dictionary)
    term_similarity_index = get_term_similarity_index(dictionary)
    term_similarity_matrix = SparseTermSimilarityMatrix(term_similarity_index, dictionary, tfidf_model,
                                                        symmetric=True, dominant=True, nonzero_limit=100)
    return term_similarity_matrix


def main(input_file: Path, output_file: Path) -> None:
    term_similarity_matrix = get_term_similarity_matrix(input_file)
    term_similarity_matrix.save(str(output_file))


if __name__ == '__main__':
    input_file = Path(argv[1])
    output_file = Path(argv[2])
    main(input_file, output_file)
