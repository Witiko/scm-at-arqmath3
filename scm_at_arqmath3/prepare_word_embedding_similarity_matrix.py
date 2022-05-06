from pathlib import Path
from sys import argv

from gensim.models import KeyedVectors  # type: ignore
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix  # type: ignore
from gensim.similarities.annoy import AnnoyIndexer  # type: ignore

from .prepare_levenshtein_similarity_matrix import get_dictionary, get_tfidf_model


def get_word_embeddings(input_word_embedding_file: Path) -> KeyedVectors:
    word_embeddings = KeyedVectors.load_word2vec_format(str(input_word_embedding_file))
    return word_embeddings


def get_term_similarity_index(word_embeddings: KeyedVectors) -> WordEmbeddingSimilarityIndex:
    annoy_indexer = AnnoyIndexer(word_embeddings, num_trees=1)
    term_similarity_index = WordEmbeddingSimilarityIndex(word_embeddings, threshold=-1.0, exponent=4.0,
                                                         kwargs={'indexer': annoy_indexer})
    return term_similarity_index


def get_term_similarity_matrix(input_dictionary_file: Path,
                               input_word_embedding_file: Path) -> SparseTermSimilarityMatrix:
    word_embeddings = get_word_embeddings(input_word_embedding_file)
    term_similarity_index = get_term_similarity_index(word_embeddings)
    dictionary = get_dictionary(input_dictionary_file)
    tfidf_model = get_tfidf_model(dictionary)
    term_similarity_matrix = SparseTermSimilarityMatrix(term_similarity_index, dictionary, tfidf_model,
                                                        symmetric=True, dominant=True, nonzero_limit=100)
    return term_similarity_matrix


def main(input_dictionary_file: Path, input_word_embedding_file: Path, output_file: Path) -> None:
    term_similarity_matrix = get_term_similarity_matrix(input_dictionary_file, input_word_embedding_file)
    term_similarity_matrix.save(str(output_file))


if __name__ == '__main__':
    input_dictionary_file = Path(argv[1])
    input_word_embedding_file = Path(argv[2])
    output_file = Path(argv[3])
    main(input_dictionary_file, input_word_embedding_file, output_file)
