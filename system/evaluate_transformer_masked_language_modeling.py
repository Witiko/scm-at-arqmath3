import json
from pathlib import Path
from sys import argv
from typing import Iterable, Tuple

from tqdm import tqdm

from .finetune_transformer import get_language_module, get_objective


Step = int
Epoch = float
Loss = float


def get_checkpoint_step(input_checkpoint_directory: Path) -> Step:
    _, checkpoint_step = input_checkpoint_directory.name.split('-')
    checkpoint_step = int(checkpoint_step)
    return checkpoint_step


def get_checkpoint_epoch(input_checkpoint_directory: Path) -> Epoch:
    checkpoint_step = get_checkpoint_step(input_checkpoint_directory)
    trainer_state_file = input_checkpoint_directory / 'trainer_state.json'
    with trainer_state_file.open('rt') as f:
        trainer_state = json.load(f)
    for trainer_state_item in trainer_state['log_history']:
        step = int(trainer_state_item['step'])
        epoch = float(trainer_state_item['epoch'])
        if step == checkpoint_step:
            return epoch
    raise ValueError(f'File {trainer_state_file} contains no information about step {checkpoint_step}')


def get_checkpoint_directories(input_objective_directory: Path) -> Iterable[Path]:
    checkpoint_directories = input_objective_directory.glob('checkpoint-*')
    checkpoint_directories = sorted(checkpoint_directories, key=get_checkpoint_step)
    return checkpoint_directories


def get_validation_loss(input_dataset_file: Path, checkpoint_directory: Path) -> Loss:
    input_model_directory = checkpoint_directory / 'MaskedLanguageModeling'
    language_module = get_language_module(input_model_directory)
    objective = get_objective(input_dataset_file, input_dataset_file, language_module)
    evaluation = objective.evaluate()
    loss = evaluation['loss']
    return loss


def get_validation_losses(input_dataset_file: Path,
                          input_objective_directory: Path) -> Iterable[Tuple[Step, Epoch, Loss]]:
    checkpoint_directories = get_checkpoint_directories(input_objective_directory)
    checkpoint_directories = tqdm(list(checkpoint_directories))
    for checkpoint_directory in checkpoint_directories:
        step = get_checkpoint_step(checkpoint_directory)
        epoch = get_checkpoint_epoch(checkpoint_directory)
        loss = get_validation_loss(input_dataset_file, checkpoint_directory)
        yield (step, epoch, loss)


def main(input_dataset_file: Path, input_objective_directory: Path, output_file: Path) -> None:
    validation_losses = list()
    for step, epoch, loss in get_validation_losses(input_dataset_file, input_objective_directory):
        validation_loss = {'step': step, 'epoch': epoch, 'loss': loss}
        validation_losses.append(validation_loss)
        with output_file.open('wt') as f:
            json.dump(validation_losses, f)


if __name__ == '__main__':
    input_dataset_file = Path(argv[1])
    input_objective_directory = Path(argv[2])
    output_file = Path(argv[3])
    main(input_dataset_file, input_objective_directory, output_file)
