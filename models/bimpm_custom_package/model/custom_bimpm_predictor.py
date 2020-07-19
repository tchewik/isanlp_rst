
from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.predictors.decomposable_attention import DecomposableAttentionPredictor
from overrides import overrides

from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers import Token, Tokenizer, CharacterTokenizer, WordTokenizer
from overrides import overrides
from typing import Dict, List, Tuple

try:
    from bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
except ModuleNotFoundError:
    from models.bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

# You need to name your predictor and register so that `allennlp` command can recognize it
# Note that you need to use "@Predictor.register", not "@Model.register"!
@Predictor.register("custom_bimpm_predictor")
class CustomBiMPMPredictor(DecomposableAttentionPredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = WhitespaceTokenizer()

    def predict(self, premise: str, hypothesis: str, same_sentence: str, same_paragraph: str) -> JsonDict:
        return self.predict_json({"premise": premise, "hypothesis": hypothesis, 
                                  "same_sentence": same_sentence, "same_paragraph": same_paragraph})
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"premise": "...", "hypothesis": "...", "metadata": "..."}`.
        """
        premise_text = json_dict["premise"]
        hypothesis_text = json_dict["hypothesis"]
        same_sentence = json_dict["same_sentence"]
        same_paragraph = json_dict["same_paragraph"]
        #metadata = json_dict["metadata"]
        #same_sentence, same_paragraph = metadata.split('\t')
        return self._dataset_reader.text_to_instance(premise_text, 
                                                     hypothesis_text, 
                                                     label=None, 
                                                     same_sentence=same_sentence,
                                                     same_paragraph=same_paragraph)
