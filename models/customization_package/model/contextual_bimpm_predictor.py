
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
    from customization_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
except ModuleNotFoundError:
    from models.customization_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

# You need to name your predictor and register so that `allennlp` command can recognize it
# Note that you need to use "@Predictor.register", not "@Model.register"!
@Predictor.register("contextual_bimpm_predictor")
class ContextualBiMpmPredictor(DecomposableAttentionPredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = WhitespaceTokenizer()

    def predict(self, premise: str, hypothesis: str, left_context: str, right_context: str, metadata: str) -> JsonDict:
        return self.predict_json({"premise": premise, "hypothesis": hypothesis, 
                                  "left_context": left_context, "right_context": right_context,
                                  "metadata": metadata})
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"premise": "...", "hypothesis": "...", "metadata": "...", 
                                       "right_context": "...", "left_context": "..."}`.
        """
        premise_text = json_dict["premise"]
        hypothesis_text = json_dict["hypothesis"]
        left_context_text = json_dict["left_context"]
        left_context_text = json_dict["right_context"]
        same_sentence = json_dict["metadata"]
        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text, label=None, same_sentence=same_sentence)
