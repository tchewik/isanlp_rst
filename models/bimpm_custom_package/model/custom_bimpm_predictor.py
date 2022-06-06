from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.predictors.decomposable_attention import DecomposableAttentionPredictor

from overrides import overrides

try:
    from bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
except ModuleNotFoundError:
    from models.bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


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
        Expects JSON that looks like `
        {"premise": "...", "hypothesis": "...", "same_sentence": "...", "same_paragraph": "..."}`.
        """
        premise_text = json_dict["premise"]
        hypothesis_text = json_dict["hypothesis"]
        same_sentence = json_dict["same_sentence"]
        same_paragraph = json_dict["same_paragraph"]

        return self._dataset_reader.text_to_instance(premise_text,
                                                     hypothesis_text,
                                                     label=None,
                                                     same_sentence=same_sentence,
                                                     same_paragraph=same_paragraph)
