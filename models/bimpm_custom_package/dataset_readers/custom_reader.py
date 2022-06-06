import csv
import logging
from typing import Optional, Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer

try:
    from bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
except ModuleNotFoundError:
    from models.bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

import numpy as np

logger = logging.getLogger(__name__)


class CustomDataReader(DatasetReader):
    """
    # Parameters
    tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the premise and hypothesis into words or other kinds of tokens.
        Defaults to `WhitespaceTokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input token representations. Defaults to `{"tokens":
        SingleIdTokenIndexer()}`.
    """

    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            combine_input_fields: Optional[bool] = None,
            **kwargs) -> None:

        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        #         if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
        #             assert not self._tokenizer._add_special_tokens

        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)

    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        file_path = cached_path(file_path)
        with open(file_path, "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter="\t")
            for row in tsv_in:
                if len(row) == 6:
                    yield self.text_to_instance(premise=row[1], hypothesis=row[2], label=row[0],
                                                same_sentence=row[3], same_paragraph=row[4])

    def text_to_instance(
            self,  # type: ignore
            premise: str,
            hypothesis: str,
            label: str,
            same_sentence: str,
            same_paragraph: str,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        tokenized_premise = self._tokenizer.tokenize(premise)
        tokenized_hypothesis = self._tokenizer.tokenize(hypothesis)

        if self._combine_input_fields:
            tokens = self._tokenizer.add_special_tokens(tokenized_premise, tokenized_hypothesis)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        else:
            tokenized_premise = self._tokenizer.add_special_tokens(tokenized_premise)
            tokenized_hypothesis = self._tokenizer.add_special_tokens(tokenized_hypothesis)
            fields["premise"] = TextField(tokenized_premise, self._token_indexers)
            fields["hypothesis"] = TextField(tokenized_hypothesis, self._token_indexers)

        _same_sentence = list(map(list, zip(*same_sentence)))
        _same_paragraph = list(map(list, zip(*same_paragraph)))
        fields["same_sentence"] = ArrayField(np.array(_same_sentence).astype(np.float32))
        fields["same_paragraph"] = ArrayField(np.array(_same_paragraph).astype(np.float32))

        if label is not None:
            fields["label"] = LabelField(label)

        return Instance(fields)
