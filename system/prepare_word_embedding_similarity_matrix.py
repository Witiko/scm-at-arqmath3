from pathlib import Path
from sys import argv

from gensim.models import FastText, KeyedVectors  # type: ignore
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix  # type: ignore
from gensim.similarities.annoy import AnnoyIndexer  # type: ignore

from .prepare_levenshtein_similarity_matrix import get_dictionary, get_tfidf_model


def get_word_embeddings(input_word_embedding_file_or_directory: Path) -> KeyedVectors:
    if input_word_embedding_file_or_directory.is_dir():
        input_model_directory, = (input_word_embedding_file_or_directory / 'model').glob('*/')
        input_model_file = input_model_directory / 'model'
        language_model = FastText.load(str(input_model_file))
        word_embeddings = language_model.wv
    else:
        word_embeddings = KeyedVectors.load(str(input_word_embedding_file_or_directory))
    return word_embeddings


def get_term_similarity_index(word_embeddings: KeyedVectors) -> WordEmbeddingSimilarityIndex:
    annoy_indexer = AnnoyIndexer(word_embeddings, num_trees=1)
    term_similarity_index = WordEmbeddingSimilarityIndex(word_embeddings, threshold=-1.0, exponent=4.0,
                                                         kwargs={'indexer': annoy_indexer})
    return term_similarity_index


def get_term_similarity_matrix(input_dictionary_file: Path,
                               input_word_embedding_file_or_directory: Path,
                               symmetric: bool, dominant: bool, nonzero_limit: int) -> SparseTermSimilarityMatrix:
    word_embeddings = get_word_embeddings(input_word_embedding_file_or_directory)
    term_similarity_index = get_term_similarity_index(word_embeddings)
    dictionary = get_dictionary(input_dictionary_file)
    tfidf_model = get_tfidf_model(dictionary)
    term_similarity_matrix = SparseTermSimilarityMatrix(term_similarity_index, dictionary, tfidf_model,
                                                        symmetric=symmetric, dominant=dominant,
                                                        nonzero_limit=nonzero_limit)
    return term_similarity_matrix


def main(input_dictionary_file: Path, input_word_embedding_file_or_directory: Path, output_file: Path,
         symmetric: bool, dominant: bool, nonzero_limit: int) -> None:
    term_similarity_matrix = get_term_similarity_matrix(input_dictionary_file, input_word_embedding_file_or_directory,
                                                        symmetric, dominant, nonzero_limit)
    term_similarity_matrix.save(str(output_file))


if __name__ == '__main__':
    input_dictionary_file = Path(argv[1])
    input_word_embedding_file_or_directory = Path(argv[2])
    output_file = Path(argv[3])
    assert argv[4] in ('True', 'False')
    symmetric = argv[4] == 'True'
    assert argv[5] in ('True', 'False')
    dominant = argv[5] == 'True'
    nonzero_limit = int(argv[6])
    main(input_dictionary_file, input_word_embedding_file_or_directory, output_file,
         symmetric, dominant, nonzero_limit)
