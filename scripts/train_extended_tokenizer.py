from pathlib import Path
from sys import argv

from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModel


def get_math_tokenizer(input_file: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(input_file))
    return tokenizer


def get_extended_model(pretrained_identifier: str, extended_tokenizer: AutoTokenizer) -> AutoModel:
    extended_model = AutoModel.from_pretrained(pretrained_identifier)
    extended_model.resize_token_embeddings(len(extended_tokenizer))
    return extended_model


def get_extended_tokenizer(pretrained_identifier: str, math_tokenizer: Tokenizer) -> AutoTokenizer:
    extended_tokenizer = AutoTokenizer.from_pretrained(pretrained_identifier, add_prefix_space=True)
    extended_tokenizer.add_special_tokens({'additional_special_tokens': [' [MATH] ', ' [/MATH]']})
    extended_tokenizer.add_tokens(list(math_tokenizer.get_vocab()))
    return extended_tokenizer


def main(pretrained_identifier: str, input_file: Path, output_path: Path) -> None:
    math_tokenizer = get_math_tokenizer(input_file)

    extended_tokenizer = get_extended_tokenizer(pretrained_identifier, math_tokenizer)
    extended_tokenizer.save_pretrained(str(output_path))

    extended_model = get_extended_model(pretrained_identifier, extended_tokenizer)
    extended_model.save_pretrained(str(output_path))


if __name__ == '__main__':
    pretrained_identifier = argv[1]
    input_file = Path(argv[2])
    output_directory = Path(argv[3])
    main(pretrained_identifier, input_file, output_directory)
