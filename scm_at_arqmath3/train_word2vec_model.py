from pathlib import Path
from sys import argv
from typing import Iterable, Iterator, List

from pine import LanguageModel
from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .train_extended_tokenizer import get_math_tokenizer, get_extended_tokenizer


Line = List[str]
Corpus = Iterable[Line]


class TextCorpus():
    def __init__(self, input_file: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.input_file = input_file
        self.number_of_lines = count_lines(input_file)

    def __iter__(self) -> Iterator[Line]:
        with self.input_file.open('rt') as f:
            sentences = tqdm(f, desc=f'Reading {self.input_file}', total=self.number_of_lines)
            for sentence in sentences:
                tokens = self.tokenizer.tokenize(sentence)
                yield tokens


class TextLaTeXCorpus():
    def __init__(self, input_file: Path) -> None:
        math_tokenizer = get_math_tokenizer(Path('tokenizer-latex.json'))
        self.tokenizer = get_extended_tokenizer('roberta-base', math_tokenizer)
        self.input_file = input_file
        self.number_of_lines = count_lines(input_file)

    def __iter__(self) -> Iterator[Line]:
        with self.input_file.open('rt') as f:
            sentences = tqdm(f, desc=f'Reading {self.input_file}', total=self.number_of_lines)
            for sentence in sentences:
                tokens = self.tokenizer.tokenize(sentence)
                yield tokens


class LaTeXCorpus():
    def __init__(self, input_file: Path) -> None:
        self.tokenizer = Tokenizer.from_file('tokenizer-latex.json')
        self.input_file = input_file
        self.number_of_lines = count_lines(input_file)

    def __iter__(self) -> Iterator[Line]:
        with self.input_file.open('rt') as f:
            sentences = tqdm(f, desc=f'Reading {self.input_file}', total=self.number_of_lines)
            for sentence in sentences:
                tokens = self.tokenizer.encode(sentence).tokens
                yield tokens


class TangentLCorpus():
    def __init__(self, input_file: Path) -> None:
        self.input_file = input_file
        self.number_of_lines = count_lines(input_file)

    def __iter__(self) -> Iterator[Line]:
        with self.input_file.open('rt') as f:
            sentences = tqdm(f, desc=f'Reading {self.input_file}', total=self.number_of_lines)
            for sentence in sentences:
                tokens = sentence.strip('#').split('# #')
                yield tokens


def count_lines(input_file: Path):
    with input_file.open('rt') as f:
        num_lines = sum(1 for _ in tqdm(f, desc=f'Counting lines in {input_file}'))
    return num_lines


def get_number_of_epochs(text_format: str) -> int:
    if text_format == 'text':
        num_epochs = 2
    elif text_format == 'text+latex':
        num_epochs = 1
    elif text_format == 'latex':
        num_epochs = 5
    elif text_format == 'tangentl':
        num_epochs = 2
    else:
        raise ValueError(f'Unknown text format {text_format}')
    return num_epochs


def get_corpus(text_format: str, input_file: Path) -> Corpus:
    if text_format == 'text':
        corpus = TextCorpus(input_file)
    elif text_format == 'text+latex':
        corpus = TextLaTeXCorpus(input_file)
    elif text_format == 'latex':
        corpus = LaTeXCorpus(input_file)
    elif text_format == 'tangentl':
        corpus = TangentLCorpus(input_file)
    else:
        raise ValueError(f'Unknown text format {text_format}')
    return corpus


def get_language_model(text_format: str, positions: bool, input_file: Path, output_file: Path) -> LanguageModel:
    corpus = get_corpus(text_format, input_file)
    num_epochs = get_number_of_epochs(text_format)
    language_model = LanguageModel(corpus, output_file, subwords=False,
                                   positions='constrained' if positions else False,
                                   extra_fasttext_parameters={'epochs': num_epochs})
    return language_model


def main(text_format: str, positions: bool, input_file: Path, output_file: Path) -> None:
    language_model = get_language_model(text_format, positions, input_file, output_file)
    _ = language_model.model


if __name__ == '__main__':
    text_format = argv[1]
    positions = argv[2] == 'positional'
    input_file = Path(argv[3])
    output_file = Path(argv[4])
    main(text_format, positions, input_file, output_file)