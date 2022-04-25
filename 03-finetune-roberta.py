from math import ceil
from pathlib import Path

from adaptor.lang_module import LangModule
from adaptor.objectives.MLM import MaskedLanguageModeling
from adaptor.schedules import SequentialSchedule
from adaptor.adapter import Adapter
from adaptor.utils import AdaptationArguments, StoppingStrategy


def main(batch_size: int = 24, effective_batch_size: int = 256,
         dataset: Path = Path('dataset-text+latex.txt'),
         input_model_dir: Path = Path('./roberta-base-text+latex/'),
         objective_dir: Path = Path('./tuned-roberta-base-text+latex.MLM-objective/'),
         output_model_dir: Path = Path('./tuned-roberta-base-text+latex/')) -> None:
    gradient_accumulation_steps = int(ceil(1.0 * effective_batch_size / batch_size))

    lang_module = LangModule(str(input_model_dir))
    objectives = MaskedLanguageModeling(lang_module, batch_size=batch_size,
                                        texts_or_path=str(dataset))
    training_arguments = AdaptationArguments(output_dir=str(objective_dir),
                                             stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_CONVERGED,
                                             evaluation_strategy='steps',
                                             do_train=True,
                                             do_eval=True,
                                             gradient_accumulation_steps=gradient_accumulation_steps,
                                             logging_steps=1000,
                                             eval_steps=1000,
                                             num_train_epochs=100)
    schedule = SequentialSchedule([objectives], training_arguments)
    adapter = Adapter(lang_module, schedule, training_arguments)
    adapter.train()  # wait a few hours here
    adapter.save_model(str(model_dir))


if __name__ == '__main__':
    main()
