import logging
from multiprocessing import Pool
from pathlib import Path
from sys import argv
from typing import Tuple

from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import AutoTokenizer


LOGGER = logging.getLogger(__name__)

LATEX_TOKENIZER = Tokenizer.from_file('tokenizer-latex.json')

TEXT_LATEX_TOKENIZER = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
TEXT_LATEX_TOKENIZER.add_special_tokens({'additional_special_tokens': [' [MATH] ', ' [/MATH]']})
TEXT_LATEX_TOKENIZER.add_tokens(list(LATEX_TOKENIZER.get_vocab()))


def measure_line_length(line: str) -> Tuple[str, int]:
    return line, len(TEXT_LATEX_TOKENIZER.encode(line))


def main(input_file: Path, output_file: Path, max_num_tokens: int = 510,
         chunksize: int = 50_000) -> None:
    with input_file.open('rt') as rf:
        lines = [line.rstrip('\r\n') for line in rf]
    with output_file.open('wt') as wf, Pool(None) as pool:
        line_lengths = pool.imap(measure_line_length, lines, chunksize)
        for line, line_length in tqdm(line_lengths, total=len(lines)):
            if line_length > max_num_tokens:
                continue
            print(line, file=wf)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    input_file = Path(argv[1])
    output_file = Path(argv[2])
    main(input_file, output_file)
