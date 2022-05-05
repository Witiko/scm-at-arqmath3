from pathlib import Path
from sys import argv

from gensim.corpora import Dictionary

from .train_word2vec_model import get_corpus, Corpus


def get_dictionary(corpus: Corpus) -> Dictionary:
    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(keep_n=None, no_below=2)
    return dictionary


def main(text_format: str, input_file: Path, output_file: Path) -> None:
    corpus = get_corpus(text_format, input_file)
    dictionary = get_dictionary(corpus)
    dictionary.save(str(output_file))


if __name__ == '__main__':
    text_format = argv[1]
    input_file = Path(argv[2])
    output_file = Path(argv[3])
    main(text_format, input_file, output_file)
