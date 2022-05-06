from math import ceil
from pathlib import Path
from sys import argv
from typing import Union

from adaptor.lang_module import LangModule
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.schedules import SequentialSchedule
from adaptor.adapter import Adapter
from adaptor.utils import AdaptationArguments, StoppingStrategy
from transformers import AutoTokenizer

from .train_extended_tokenizer import get_math_tokenizer, get_extended_tokenizer


PathOrIdentifier = Union[Path, str]


def get_batch_size() -> int:
    batch_size = 48
    return batch_size


def get_effective_batch_size() -> int:
    effective_batch_size = 256
    return effective_batch_size


def get_gradient_accumulation_steps() -> float:
    effective_batch_size = 256
    batch_size = get_batch_size()
    gradient_accumulation_steps = int(ceil(1.0 * effective_batch_size / batch_size))
    return gradient_accumulation_steps


def get_adaptation_arguments(objective_directory: Path) -> AdaptationArguments:
    gradient_accumulation_steps = get_gradient_accumulation_steps()
    number_of_training_epochs = 1
    adaptation_arguments = AdaptationArguments(
        output_dir=str(objective_directory),
        stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_CONVERGED, stopping_patience=2,
        evaluation_strategy='steps', eval_steps=1000,
        save_strategy='steps', save_steps=1000,
        logging_strategy='steps', logging_steps=1000,
        do_train=True, do_eval=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=number_of_training_epochs + 1,
    )
    return adaptation_arguments


def get_adapter(input_training_dataset_file: Path,
                input_validation_dataset_file: Path,
                input_model_directory: PathOrIdentifier,
                extended_tokenizer: AutoTokenizer,
                adaptation_arguments: AdaptationArguments) -> Adapter:
    kwargs = {'tokenizer': extended_tokenizer} if isinstance(input_model_directory, Path) else dict()
    language_module = LangModule(str(input_model_directory), **kwargs)
    batch_size = get_batch_size()
    objectives = MaskedLanguageModeling(language_module, batch_size=batch_size,
                                        texts_or_path=str(input_training_dataset_file),
                                        val_texts_or_path=str(input_validation_dataset_file))
    schedule = SequentialSchedule([objectives], adaptation_arguments)
    adapter = Adapter(language_module, schedule, adaptation_arguments)
    return adapter


def main(pretrained_identifier: str,
         input_training_dataset_file: Path,
         input_validation_dataset_file: Path,
         input_model_directory: Path,
         input_math_tokenizer_file: Path,
         objective_directory: Path,
         output_model_directory: Path) -> None:
    math_tokenizer = get_math_tokenizer(input_math_tokenizer_file)
    extended_tokenizer = get_extended_tokenizer(pretrained_identifier, math_tokenizer)
    adaptation_arguments = get_adaptation_arguments(objective_directory)
    adapter = get_adapter(input_training_dataset_file, input_validation_dataset_file,
                          input_model_directory, extended_tokenizer, adaptation_arguments)
    adapter.train()
    adapter.save_model(str(output_model_directory))


if __name__ == '__main__':
    pretrained_identifier = argv[1]
    input_training_dataset_file = Path(argv[2])
    input_validation_dataset_file = Path(argv[3])
    input_model_directory = Path(argv[4])
    input_math_tokenizer_file = Path(argv[5])
    objective_directory = Path(argv[6])
    output_model_directory = Path(argv[7])
    main(pretrained_identifier, input_training_dataset_file, input_validation_dataset_file,
         input_model_directory, input_math_tokenizer_file, objective_directory,
         output_model_directory)
