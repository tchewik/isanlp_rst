import json
import os

import razdel
import torch
import json
import os

import razdel
import torch
from isanlp.annotation import Token
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

from isanlp_rst.dmrst_parser.du_converter import DUConverter
from isanlp_rst.dmrst_parser.src.parser.data import Data
from isanlp_rst.dmrst_parser.src.parser.parsing_net import ParsingNet
from isanlp_rst.dmrst_parser.trainer import Trainer
from isanlp_rst.parsing_net_ensemble import ParsingNetEnsemble


class EnsemblePredictor:
    def __init__(self, predictor_list, aggregation_method="average"):
        """
        Initializes the ensemble predictor with a list of individual predictors.

        :param predictor_list: List of Predictor objects (instances of the Predictor class).
        :param aggregation_method: The method used to aggregate the predictions from individual models.
                                   Options are 'average', 'vote', or 'weighted_average'.
        """
        self.predictors = predictor_list  # List of individual Predictor objects
        self.aggregation_method = aggregation_method
        self.ensemble = ParsingNetEnsemble([predictor.model for predictor in self.predictors])

    def tokenize_input(self, text, model_idx=0):
        """
        Tokenizes the input text using the tokenizer from one of the models.

        :param text: The input text to be tokenized.
        :param model_idx: Index of the model in the ensemble to use for tokenization.
        :return: Tokenized input.
        """
        tokenizer = self.predictors[model_idx].tokenizer
        tokenized_output = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return tokenized_output

    def parse_rst(self, text):
        """
        Parse the input text using the ensemble of models.

        :param text: The input text to be parsed.
        :return: Aggregated output (tokens and tree structure).
        """

        # Preprocess the text
        _text = text.replace('-', ' - ').replace('—', ' — ').replace('  ', ' ')
        _text = _text.replace('...', '…').replace('_', ' ')
        tokenized_text = [token.text for token in razdel.tokenize(_text)]
        data = {
            'input_sentences': [tokenized_text],
            'edu_breaks': [[]],  # Placeholder for EDU breaks
            'decoder_input': [[]],  # Placeholder for decoder input
            'relation_label': [[]],  # Placeholder for relation labels
            'parsing_breaks': [[]],  # Placeholder for parsing breaks
            'golden_metric': [[]],  # Placeholder for evaluation metrics
        }
        input_data = Data(**data)

        # Tokenize the input and prepare it for batching
        batch = self.predictors[0].tokenize(input_data)

        # Perform inference using the ensemble
        with torch.no_grad():
            result = self.ensemble(batch, generate_tree=False)

        return result
