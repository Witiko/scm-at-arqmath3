from itertools import chain
from os import devnull
from pathlib import Path
from sys import argv
from typing import Iterable, Tuple

from transformers import AutoTokenizer
from tqdm import tqdm

from .train_extended_tokenizer import get_math_tokenizer, get_extended_tokenizer
from .finetune_transformer import get_adaptation_arguments, get_adapter, PathOrIdentifier


def get_validation_loss(input_validation_dataset_file: Path,
                        checkpoint: PathOrIdentifier,
                        extended_tokenizer: AutoTokenizer) -> float:
    adaptation_arguments = get_adaptation_arguments(Path(devnull))
    adapter = get_adapter(Path(devnull), input_validation_dataset_file, checkpoint,
                          extended_tokenizer, adaptation_arguments)
    evaluation = adapter.evaluate()
    validation_loss = evaluation['eval_loss']
    return validation_loss


def get_checkpoints(objective_directory: Path) -> Iterable[Tuple[int, PathOrIdentifier]]:

    def get_checkpoint_number(checkpoint_directory: Path) -> int:
        *_, checkpoint_number = checkpoint_directory.name.split('-')
        return int(checkpoint_number)

    baseline = (0, 'roberta-base')
    nonbaseline_checkpoints: Iterable[Tuple[int, str]] = (
        (get_checkpoint_number(checkpoint), str(checkpoint / 'MaskedLanguageModeling'))
        for checkpoint
        in objective_directory.glob('checkpoint-*')
    )
    checkpoints = chain([baseline], nonbaseline_checkpoints)
    return checkpoints


def main(pretrained_identifier: str,
         input_validation_dataset_file: Path,
         input_math_tokenizer_file: Path,
         objective_directory: Path,
         output_file: Path) -> None:
    math_tokenizer = get_math_tokenizer(input_math_tokenizer_file)
    extended_tokenizer = get_extended_tokenizer(pretrained_identifier, math_tokenizer)
    checkpoints = sorted(get_checkpoints(objective_directory))
    with output_file.open('wt') as f:
        for checkpoint_number, checkpoint in tqdm(checkpoints):
            validation_loss = get_validation_loss(input_validation_dataset_file,
                                                  checkpoint, extended_tokenizer)
            print(f'{checkpoint_number}\t{validation_loss}', file=f, flush=True)


if __name__ == '__main__':
    pretrained_identifier = argv[1]
    input_validation_dataset_file = Path(argv[2])
    input_math_tokenizer_file = Path(argv[3])
    objective_directory = Path(argv[4])
    output_file = Path(argv[5])
    main(pretrained_identifier, input_validation_dataset_file, input_math_tokenizer_file,
         objective_directory, output_file)
