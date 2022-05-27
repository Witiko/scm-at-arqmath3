from math import ceil
from pathlib import Path
from sys import argv
from typing import Optional

from adaptor.lang_module import LangModule
from adaptor.objectives.objective_base import Objective
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.schedules import SequentialSchedule
from adaptor.adapter import Adapter
from adaptor.utils import AdaptationArguments, StoppingStrategy

from .extract_decontextualized_word_embeddings import PathOrIdentifier, get_tokenizer


def get_batch_size() -> int:
    batch_size = 54
    return batch_size


def get_stopping_patience() -> int:
    stopping_patience = 2
    return stopping_patience


def get_effective_batch_size() -> int:
    effective_batch_size = 256
    return effective_batch_size


def get_gradient_accumulation_steps() -> float:
    effective_batch_size = 256
    batch_size = get_batch_size()
    gradient_accumulation_steps = int(ceil(1.0 * effective_batch_size / batch_size))
    return gradient_accumulation_steps


def get_adaptation_arguments(objective_directory: Optional[Path] = None) -> AdaptationArguments:
    gradient_accumulation_steps = get_gradient_accumulation_steps()
    adaptation_arguments = AdaptationArguments(
        output_dir=str(objective_directory) if objective_directory is not None else '.',
        stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_CONVERGED,
        evaluation_strategy='steps', eval_steps=1000,
        save_strategy='steps', save_steps=1000,
        logging_strategy='steps', logging_steps=1000,
        do_train=True, do_eval=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1000,
        remove_unused_columns=False,
        fp16=True, fp16_full_eval=True,
    )
    return adaptation_arguments


class LangModuleWithSpacePrefixingTokenizer(LangModule):
    def __init__(self, input_model_directory: PathOrIdentifier) -> None:
        super().__init__(str(input_model_directory))
        self.tokenizer = get_tokenizer(input_model_directory)


def get_language_module(input_model_directory: PathOrIdentifier) -> LangModule:
    language_module = LangModuleWithSpacePrefixingTokenizer(input_model_directory)
    return language_module


def get_objective(input_training_dataset_file: Path,
                  input_validation_dataset_file: Path,
                  language_module: LangModule) -> Objective:
    batch_size = get_batch_size()
    objective = MaskedLanguageModeling(language_module, batch_size=batch_size,
                                       texts_or_path=str(input_training_dataset_file),
                                       val_texts_or_path=str(input_validation_dataset_file))
    return objective


def get_adapter(input_training_dataset_file: Path,
                input_validation_dataset_file: Path,
                input_model_directory: PathOrIdentifier,
                adaptation_arguments: AdaptationArguments) -> Adapter:
    language_module = get_language_module(input_model_directory)
    objective = get_objective(input_training_dataset_file, input_validation_dataset_file, language_module)
    schedule = SequentialSchedule([objective], adaptation_arguments)
    adapter = Adapter(language_module, schedule, adaptation_arguments)
    return adapter


def main(pretrained_identifier: str,
         input_training_dataset_file: Path,
         input_validation_dataset_file: Path,
         input_model_directory: Path,
         input_math_tokenizer_file: Path,
         objective_directory: Path,
         output_model_directory: Path) -> None:
    adaptation_arguments = get_adaptation_arguments(objective_directory)
    adapter = get_adapter(input_training_dataset_file, input_validation_dataset_file,
                          input_model_directory, adaptation_arguments)
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
