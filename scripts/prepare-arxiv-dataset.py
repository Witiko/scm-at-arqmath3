from io import TextIOWrapper
from itertools import repeat
import logging
from multiprocessing import Pool
from pathlib import Path
import re
from sys import argv
from typing import Iterable, Tuple, List
from zipfile import ZipFile

from lxml.html import HTMLParser, parse, Element
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
                  allowed_severities={'no-problem'}) -> Iterable[Document]:
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


def read_document_text_latex(paragraph: Element, min_paragraph_length: int = 250) -> Iterable[Line]:
    for math in paragraph.xpath('.//math'):
        replacement = Element('span')
        replacement.text = f' [MATH] {math.attrib["alttext"]} [/MATH] ' if 'alttext' in math.attrib else ''
        math.getparent().replace(math, replacement)
    paragraph_text = re.sub(r'\s+', ' ', paragraph.text_content().rstrip())
    if not paragraph_text.startswith(' [MATH]'):
        paragraph_text = paragraph_text.lstrip()
    paragraph_text = re.sub(r' \[/MATH\] \[MATH\] ', ' ', paragraph_text)
    paragraph_text = re.sub(r' \[/MATH\]$', '', paragraph_text)
    if len(paragraph_text) >= min_paragraph_length:
        yield paragraph_text


def read_document_latex(paragraph: Element, min_math_length: int = 20) -> Iterable[Line]:
    for math in paragraph.xpath('.//math'):
        if 'alttext' not in math.attrib:
            continue
        math_text = math.attrib['alttext']
        math_text = re.sub(r'\s+', ' ', math_text.strip())
        if len(math_text) >= min_math_length:
            yield math_text


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
        if text_format == 'text+latex':
            paragraph_text = read_document_text_latex(paragraph)
        elif text_format == 'latex':
            paragraph_text = read_document_latex(paragraph)
        else:
            raise ValueError(f'Unknown text format {text_format}')
        paragraphs.extend(paragraph_text)
    return document, paragraphs


def main(text_format: TextFormat, arxiv_input_directory: Path, output_file: Path) -> None:
    documents = list(get_documents(arxiv_input_directory))
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
    arxiv_input_directory = Path(argv[2])
    output_file = Path(argv[3])
    main(text_format, arxiv_input_directory, output_file)
