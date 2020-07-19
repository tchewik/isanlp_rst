
import csv
import logging
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides

try:
    from bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
except ModuleNotFoundError:
    from models.bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

import numpy as np

logger = logging.getLogger(__name__)


@DatasetReader.register("custom_pairs_reader")
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
            self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None,
            lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        with open(cached_path(file_path), "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter="\t")
            for row in tsv_in:
                if len(row) == 6:
                    yield self.text_to_instance(premise=row[1], hypothesis=row[2], label=row[0], 
                                                same_sentence=row[3], same_paragraph=row[4])

    @overrides
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
        fields["premise"] = TextField(tokenized_premise, self._token_indexers)
        fields["hypothesis"] = TextField(tokenized_hypothesis, self._token_indexers)
        _same_sentence = list(map(list, zip(*same_sentence)))
        _same_paragraph = list(map(list, zip(*same_paragraph)))
        #additional_features = list(map(list, zip(*same_sentence)))
        fields["same_sentence"] = ArrayField(np.array(_same_sentence))
        fields["same_paragraph"] = ArrayField(np.array(_same_paragraph))
        if label is not None:
            fields["label"] = LabelField(label)

        return Instance(fields)
