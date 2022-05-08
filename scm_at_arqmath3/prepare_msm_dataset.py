from contextlib import contextmanager
from itertools import chain, repeat
import logging
from multiprocessing import Pool, TimeoutError
from pathlib import Path
import re
import signal
from sys import argv, setrecursionlimit
from typing import Iterable, Union, Tuple, List

from lxml.html import fromstring, Element
from tqdm import tqdm
from mathtuples.exceptions import UnknownTagException
from mathtuples.mathsymbol import MathSymbol
from mathtuples.convert import check_node, expand_nodes_with_location, format_node, START_TAG, END_TAG
from pv211_utils.arqmath.entities import ArqmathQuestionBase, ArqmathAnswerBase
from pv211_utils.arqmath.loader import load_questions, load_answers


LOGGER = logging.getLogger(__name__)


Document = Union[ArqmathQuestionBase, ArqmathAnswerBase]
Line = str
TextFormat = str


def get_documents(msm_input_directory: Path,
                  text_format: str,
                  question_filename: str = 'arqmath2020_questions_{text_format}.json.gz',
                  answer_filename: str = 'arqmath2020_answers_{text_format}.json.gz') -> Iterable[Document]:
    question_filename: Path = msm_input_directory / question_filename.format(text_format=text_format)
    answer_filename: Path = msm_input_directory / answer_filename.format(text_format=text_format)
    answers = load_answers(text_format, cache_download=answer_filename)
    questions = load_questions(text_format, answers, cache_download=question_filename)
    documents = chain(questions.values(), answers.values())
    return documents


def iterate_math_elements(paragraph: Element) -> Iterable[Element]:
    math_xpath = '@class="math-container" or contains(@class, " math-container") or contains(@class, "math-container ")'
    for math in paragraph.xpath(f'.//math | .//span[{math_xpath}]'):
        yield math


def read_document_text(paragraph: Element, min_paragraph_length: int = 100) -> Iterable[Line]:
    for math in iterate_math_elements(paragraph):
        math.text = ' '
    paragraph_text = re.sub(r'\s+', ' ', paragraph.text_content().strip())
    if len(paragraph_text) >= min_paragraph_length:
        yield paragraph_text


def read_document_text_latex(paragraph: Element, min_paragraph_length: int = 250) -> Iterable[Line]:
    for math in iterate_math_elements(paragraph):
        math.text = f' [MATH] {math.text} [/MATH] '
    paragraph_text = re.sub(r'\s+', ' ', paragraph.text_content().rstrip())
    if not paragraph_text.startswith(' [MATH]'):
        paragraph_text = paragraph_text.lstrip()
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


@contextmanager
def timeout(duration: int):
    def timeout_handler(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def read_document_tangentl(paragraph: Element, maximum_duration: int = 10) -> Iterable[Line]:
    for math in iterate_math_elements(paragraph):
        try:
            with timeout(maximum_duration):
                try:
                    tree_root = MathSymbol.parse_from_mathml(math)
                except AttributeError:
                    continue
                if tree_root is None:
                    continue
                pairs = tree_root.get_pairs('', 1, eol=False, symbol_pairs=True,
                                            compound_symbols=True, terminal_symbols=True,
                                            edge_pairs=False, unbounded=False,
                                            repetitions=True, repDict=dict(), shortened=True)
                node_list = [node for node in pairs if check_node(node)]
                nodes_payloads = expand_nodes_with_location(node_list)
                node_list = [format_node(node) for node in nodes_payloads]
                node_list = [START_TAG] + node_list + [END_TAG]
                line = ' '.join(node_list)
                yield line
        except (UnknownTagException, TimeoutError):
            pass


def _read_document_helper(args: Tuple[Document, TextFormat]) -> Tuple[Document, List[Line]]:
    return read_document(*args)


def read_document(document: Document, text_format: TextFormat) -> Tuple[Document, List[Line]]:
    tree = fromstring(document.body)
    paragraphs = list()
    for paragraph in tree.xpath('//p'):
        if text_format == 'text':
            paragraph_text = read_document_text(paragraph)
        elif text_format == 'text+latex':
            paragraph_text = read_document_text_latex(paragraph)
        elif text_format == 'latex':
            paragraph_text = read_document_latex(paragraph)
        elif text_format == 'tangentl':
            paragraph_text = read_document_tangentl(paragraph)
        else:
            raise ValueError(f'Unknown text format {text_format}')
        paragraphs.extend(paragraph_text)
    return document, paragraphs


def main(output_text_format: TextFormat, msm_input_directory: Path, output_file: Path) -> None:
    input_text_format = 'xhtml+pmml' if output_text_format == 'tangentl' else 'xhtml+latex'
    documents = list(get_documents(msm_input_directory, input_text_format))
    with output_file.open('wt') as f:
        with Pool(None) as pool:
            for document, paragraphs in tqdm(pool.imap_unordered(_read_document_helper,
                                                                 zip(documents, repeat(output_text_format))),
                                             total=len(documents)):
                for paragraph in paragraphs:
                    print(paragraph, file=f)


if __name__ == '__main__':
    setrecursionlimit(15000)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    text_format = argv[1]
    msm_input_directory = Path(argv[2])
    output_file = Path(argv[3])
    main(text_format, msm_input_directory, output_file)
