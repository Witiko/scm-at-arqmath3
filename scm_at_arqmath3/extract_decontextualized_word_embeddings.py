from collections import defaultdict
from pathlib import Path
from sys import argv
from typing import Iterable, Tuple, List

from gensim.models.keyedvectors import KeyedVectors, _add_word_to_kv  # type: ignore
from more_itertools import chunked, zip_equal
import numpy as np
from torch import no_grad
from torch.nn import ModuleList
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from .finetune_transformer import PathOrIdentifier
from .train_extended_tokenizer import get_math_tokenizer, get_extended_tokenizer
from .train_word2vec_model import Token, count_lines


Device = str
Embedding = np.ndarray
Line = str
Dataset = Iterable[Line]


def get_tokenizer(input_model: PathOrIdentifier) -> AutoTokenizer:
    if isinstance(input_model, Path):
        math_tokenizer = get_math_tokenizer(Path('tokenizer-latex.json'))
        extended_tokenizer = get_extended_tokenizer('roberta-base', math_tokenizer)
    else:
        extended_tokenizer = AutoTokenizer.from_pretrained(input_model)
    return extended_tokenizer


def get_device() -> Device:
    device = 'cuda'
    return device


def get_top_layer_number() -> int:
    top_layer_number = 10
    return top_layer_number


def get_model(input_model: PathOrIdentifier) -> AutoModel:
    model = AutoModel.from_pretrained(str(input_model))

    top_layer_number = get_top_layer_number()
    assert 0 <= top_layer_number <= len(model.encoder.layer)
    model.encoder.layer = ModuleList([layer for layer in model.encoder.layer[:top_layer_number]])

    device = get_device()
    model.to(device)
    model.eval()

    return model


def get_batch_size() -> int:
    batch_size = 500
    return batch_size


def get_dataset(input_file: Path) -> Dataset:
    number_of_lines = count_lines(input_file)
    with input_file.open('rt') as f:
        for line in tqdm(f, desc=f'Reading {input_file}', total=number_of_lines):
            line = line.rstrip('\r\n')
            yield line


def tokenize_and_embed_dataset(tokenizer: AutoTokenizer, model: AutoModel,
                               dataset: Dataset) -> Iterable[List[Tuple[Token, Embedding]]]:
    device = get_device()
    batch_size = get_batch_size()
    for lines_batch in chunked(dataset, batch_size):
        inputs_batch = tokenizer(lines_batch, return_tensors='pt', padding=True, truncation=True)
        tokens_batch = [list(map(tokenizer.decode, inputs)) for inputs in inputs_batch['input_ids']]
        inputs_batch.to(device)
        with no_grad():
            outputs_batch = model(**inputs_batch)
        embeddings_batch = outputs_batch[0].detach().cpu().numpy()
        for tokens, embeddings in zip_equal(tokens_batch, embeddings_batch):

            def filter_tokens_and_embeddings() -> Iterable[Tuple[Token, Embedding]]:
                for token, embedding in zip_equal(tokens, embeddings):
                    if token in tokenizer.all_special_tokens:
                        continue
                    yield token, embedding

            filtered_tokens_and_embeddings = list(filter_tokens_and_embeddings())
            yield filtered_tokens_and_embeddings


def get_embedding_size(model: AutoModel) -> int:
    embedding_size = model.config.hidden_size
    return embedding_size


def get_decontextualized_word_embeddings(tokenizer: AutoTokenizer, model: AutoModel,
                                         dataset: Dataset) -> KeyedVectors:
    contextual_word_embeddings = defaultdict(lambda: list())
    for tokens_and_embeddings in tokenize_and_embed_dataset(tokenizer, model, dataset):
        for token, embedding in tokens_and_embeddings:
            contextual_word_embeddings[token].append(embedding)

    number_of_tokens = len(contextual_word_embeddings)
    embedding_size = get_embedding_size(model)
    decontextualized_word_embeddings = KeyedVectors(embedding_size, number_of_tokens, dtype=float)
    for token, embeddings in tqdm(contextual_word_embeddings.items(),
                                  desc='Decontextualizing word embeddings'):
        decontextualized_word_embedding = np.mean(embeddings, axis=0)
        _add_word_to_kv(decontextualized_word_embeddings, None, token,
                        decontextualized_word_embedding, number_of_tokens)

    return decontextualized_word_embeddings


def main(input_path_or_identifier: PathOrIdentifier, input_dataset_file: Path,
         output_file: Path) -> None:
    tokenizer = get_tokenizer(input_path_or_identifier)
    model = get_model(input_path_or_identifier)
    dataset = get_dataset(input_dataset_file)
    decontextualized_word_embeddings = get_decontextualized_word_embeddings(tokenizer, model, dataset)
    decontextualized_word_embeddings.save(str(output_file))


if __name__ == '__main__':
    input_path_or_identifier = argv[1]
    if Path(input_path_or_identifier).exists():
        input_path_or_identifier = Path(input_path_or_identifier)
    input_dataset_file = Path(argv[2])
    output_file = Path(argv[3])
    main(input_path_or_identifier, input_dataset_file, output_file)
