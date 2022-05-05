from pathlib import Path
from sys import argv

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def get_tokenizer(input_file: Path) -> Tokenizer:
    model = BPE(unk_token='[UNK]')
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.normalizer = normalizers.Sequence([normalizers.Strip()])
    tokenizer_trainer = BpeTrainer(special_tokens=['[UNK]'])
    tokenizer.train(['dataset-latex.txt'], tokenizer_trainer)
    return tokenizer


def main(input_file: Path, output_file: Path) -> None:
    tokenizer = get_tokenizer(input_file)
    _ = tokenizer.save(str(output_file))


if __name__ == '__main__':
    input_file = Path(argv[1])
    output_file = Path(argv[2])
    main(input_file, output_file)
