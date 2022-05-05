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


BATCH_SIZE: int = 48
EFFECTIVE_BATCH_SIZE: int = 256
LOGGING_STEPS: int = 1000
EVALUATION_STEPS: int = 1000
NUMBER_OF_TRANING_EPOCHS: int = 1
STOPPING_PATIENCE: int = 2


def get_gradient_accumulation_steps() -> float:
    gradient_accumulation_steps = int(ceil(1.0 * EFFECTIVE_BATCH_SIZE / BATCH_SIZE))
    return gradient_accumulation_steps


def get_adaptation_arguments(objective_directory: Path) -> AdaptationArguments:
    gradient_accumulation_steps = get_gradient_accumulation_steps()
    adaptation_arguments = AdaptationArguments(
        output_dir=str(objective_directory),
        stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_CONVERGED,
        stopping_patience=STOPPING_PATIENCE,
        evaluation_strategy='steps',
        do_train=True,
        do_eval=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVALUATION_STEPS,
        num_train_epochs=NUMBER_OF_TRANING_EPOCHS + 1,
    )
    return adaptation_arguments


def get_adapter(input_training_dataset_file: Path,
                input_validation_dataset_file: Path,
                input_model_directory: PathOrIdentifier,
                extended_tokenizer: AutoTokenizer,
                adaptation_arguments: AdaptationArguments) -> Adapter:
    kwargs = {'tokenizer': extended_tokenizer} if isinstance(input_model_directory, Path) else dict()
    language_module = LangModule(str(input_model_directory), **kwargs)
    objectives = MaskedLanguageModeling(language_module, batch_size=BATCH_SIZE,
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
