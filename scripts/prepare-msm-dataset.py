from itertools import chain, repeat
import logging
from multiprocessing import Pool
from pathlib import Path
import re
from sys import argv
from typing import Iterable, Union, Tuple, List

from lxml.html import fromstring, Element
from tqdm import tqdm
from pv211_utils.arqmath.entities import ArqmathQuestionBase, ArqmathAnswerBase
from pv211_utils.arqmath.loader import load_questions, load_answers


LOGGER = logging.getLogger(__name__)


Document = Union[ArqmathQuestionBase, ArqmathAnswerBase]
Line = str
TextFormat = str


def get_documents(msm_input_directory: Path,
                  text_format: str = 'xhtml+latex',
                  question_filename: str = 'arqmath2020_questions_xhtml+latex.json.gz',
                  answer_filename: str = 'arqmath2020_answers_xhtml+latex.json.gz') -> Iterable[Document]:
    question_filename: Path = msm_input_directory / question_filename
    answer_filename: Path = msm_input_directory / answer_filename
    answers = load_answers(text_format, cache_download=answer_filename)
    questions = load_questions(text_format, answers, cache_download=question_filename)
    documents = chain(questions.values(), answers.values())
    return documents


def iterate_math_elements(paragraph: Element) -> Iterable[Element]:
    math_xpath = '@class="math-container" or contains(@class, " math-container") or contains(@class, "math-container ")'
    for math in paragraph.xpath(f'.//span[{math_xpath}]'):
        yield math


def read_document_text_latex(paragraph: Element, min_paragraph_length: int = 250) -> Iterable[Line]:
    for math in iterate_math_elements(paragraph):
        math.text = f' [MATH] {math.text} [/MATH] '
    paragraph_text = re.sub(r'\s+', ' ', paragraph.text_content().strip())
    paragraph_text = re.sub(r' \[/MATH\] \[MATH\] ', ' ', paragraph_text)
    paragraph_text = re.sub(r' \[/MATH\]$', '', paragraph_text)
    if len(paragraph_text) >= min_paragraph_length:
        yield paragraph_text


def read_document_latex(paragraph: Element, min_math_length: int = 20) -> Iterable[Line]:
    for math in iterate_math_elements(paragraph):
        if not math.text:
            continue
        math_text = re.sub(r'\s+', ' ', math.text.strip())
        if len(math_text) >= min_math_length:
            yield math_text


def _read_document_helper(args: Tuple[Document, TextFormat]) -> Tuple[Document, List[Line]]:
    return read_document(*args)


def read_document(document: Document, text_format: TextFormat) -> Tuple[Document, List[Line]]:
    tree = fromstring(document.body)
    paragraphs = list()
    for paragraph in tree.xpath('//p'):
        if text_format == 'text+latex':
            paragraph_text = read_document_text_latex(paragraph)
        elif text_format == 'latex':
            paragraph_text = read_document_latex(paragraph)
        else:
            raise ValueError(f'Unknown text format {text_format}')
        paragraphs.extend(paragraph_text)
    return document, paragraphs


def main(text_format: TextFormat, msm_input_directory: Path, output_file: Path) -> None:
    documents = list(get_documents(msm_input_directory))
    with output_file.open('wt') as f:
        with Pool(None) as pool:
            for document, paragraphs in tqdm(pool.imap_unordered(_read_document_helper,
                                             zip(documents, repeat(text_format))),
                                             total=len(documents)):
                for paragraph in paragraphs:
                    print(paragraph, file=f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    text_format = argv[1]
    msm_input_directory = Path(argv[2])
    output_file = Path(argv[3])
    main(text_format, msm_input_directory, output_file)
