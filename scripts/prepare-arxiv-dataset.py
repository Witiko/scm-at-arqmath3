from contextlib import contextmanager
from io import TextIOWrapper
from itertools import repeat
import logging
from multiprocessing import Pool, TimeoutError
from pathlib import Path
import re
import signal
from sys import argv, setrecursionlimit
from typing import Iterable, Tuple, List, Set
from xml.etree.ElementTree import ParseError
from zipfile import ZipFile

from lxml import etree
from lxml.html import HTMLParser, parse, Element
from mathtuples.exceptions import UnknownTagException
from mathtuples.math_extractor import MathExtractor
from mathtuples.mathsymbol import MathSymbol
from mathtuples.convert import check_node, expand_nodes_with_location, format_node, START_TAG, END_TAG
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


DocumentIdentifier = Path
ZipPath = Tuple[Path, Path]
Document = ZipPath
Severity = str
Line = str
TextFormat = str


def get_severities(arxiv_input_directory: Path) -> Iterable[Tuple[DocumentIdentifier, Severity]]:
    with ZipFile(arxiv_input_directory / 'meta' / 'grouped_by_severity.zip') as zf:
        for document_list in zf.namelist():
            match = re.match(r'(.*)-tasks.txt', document_list)
            assert match, document_list
            severity = match.group(1)
            with TextIOWrapper(zf.open(document_list, 'r')) as f:
                for document in f:
                    document_identifier = Path(document.strip())
                    yield (document_identifier, severity)


def get_documents(arxiv_input_directory: Path,
                  allowed_severities: Set[Severity]) -> Iterable[Document]:
    num_documents_total = 0
    num_documents_with_unknown_severity = 0
    num_documents_with_allowed_severity = 0
    severities = dict(get_severities(arxiv_input_directory))
    for zip_archive in (arxiv_input_directory / 'data').glob('*.zip'):
        with ZipFile(zip_archive, 'r') as zf:
            for document in zf.namelist():
                if zf.getinfo(document).is_dir():
                    continue
                document = Path(document)
                document_identifier = Path(document.name)
                num_documents_total += 1
                if document_identifier not in severities:
                    num_documents_with_unknown_severity += 1
                    continue
                if severities[document_identifier] not in allowed_severities:
                    continue
                num_documents_with_allowed_severity += 1
                yield zip_archive, document

    if num_documents_with_unknown_severity > 0:
        message = 'Skipped {} out of {} ({:g}%) documents with unknown severity of conversion errors.'
        message = message.format(num_documents_with_unknown_severity, num_documents_total,
                                 100.0 * num_documents_with_unknown_severity / num_documents_total)
        LOGGER.warning(message)

    message = 'Found {} out of {} ({:g}%) documents with severities in {}.'
    message = message.format(num_documents_with_allowed_severity, num_documents_total,
                             100.0 * num_documents_with_allowed_severity / num_documents_total,
                             allowed_severities)
    LOGGER.info(message)


def iterate_math_elements(paragraph: Element) -> Iterable[Element]:
    for math in paragraph.xpath('.//math'):
        yield math


def read_document_text(paragraph: Element, min_paragraph_length: int = 100) -> Iterable[Line]:
    for math in iterate_math_elements(paragraph):
        replacement = Element('span')
        replacement.text = ' '
        replacement.tail = math.tail
        math.getparent().replace(math, replacement)
    paragraph_text = re.sub(r'\s+', ' ', paragraph.text_content().rstrip())
    if len(paragraph_text) >= min_paragraph_length:
        yield paragraph_text


def read_document_text_latex(paragraph: Element, min_paragraph_length: int = 250) -> Iterable[Line]:
    for math in iterate_math_elements(paragraph):
        replacement = Element('span')
        try:
            replacement.text = f' [MATH] {math.attrib["alttext"]} [/MATH] ' if 'alttext' in math.attrib else ''
        except ValueError as e:
            LOGGER.warning(f'Encountered {e} when setting element text')
            return
        replacement.tail = math.tail
        math.getparent().replace(math, replacement)
    paragraph_text = re.sub(r'\s+', ' ', paragraph.text_content().rstrip())
    if not paragraph_text.startswith(' [MATH]'):
        paragraph_text = paragraph_text.lstrip()
    paragraph_text = re.sub(r' \[/MATH\] \[MATH\] ', ' ', paragraph_text)
    paragraph_text = re.sub(r' \[/MATH\]$', '', paragraph_text)
    if len(paragraph_text) >= min_paragraph_length:
        yield paragraph_text


def read_document_latex(paragraph: Element, min_math_length: int = 20) -> Iterable[Line]:
    for math in iterate_math_elements(paragraph):
        if 'alttext' not in math.attrib:
            continue
        math_text = math.attrib['alttext']
        math_text = re.sub(r'\s+', ' ', math_text.strip())
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
                math_tokens = etree.tostring(math, encoding='utf-8').decode('utf-8')
                try:
                    presentation_math = MathExtractor.isolate_pmml(math_tokens)
                except ParseError:
                    continue
                try:
                    tree_root = MathSymbol.parse_from_mathml(presentation_math)
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
    with ZipFile(document[0], 'r') as zf:
        with zf.open(str(document[1])) as f:
            parser = HTMLParser()
            tree = parse(f, parser)
    tree_root = tree.getroot()
    paragraphs = list()
    xpath = '//*[@class="ltx_para" or contains(@class, " ltx_para") or contains(@class, "ltx_para ")]'
    for paragraph in tree_root.xpath(xpath):
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


def main(text_format: TextFormat, allowed_severities: Set[Severity], arxiv_input_directory: Path,
         output_file: Path) -> None:
    documents = list(get_documents(arxiv_input_directory, allowed_severities))
    with output_file.open('wt') as f:
        with Pool(None) as pool:
            for document, paragraphs in tqdm(pool.imap_unordered(_read_document_helper,
                                                                 zip(documents, repeat(text_format))),
                                             total=len(documents)):
                for paragraph in paragraphs:
                    print(paragraph, file=f)


if __name__ == '__main__':
    setrecursionlimit(15000)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    text_format = argv[1]
    allowed_severities = set(argv[2].split(','))
    arxiv_input_directory = Path(argv[3])
    output_file = Path(argv[4])
    main(text_format, allowed_severities, arxiv_input_directory, output_file)
