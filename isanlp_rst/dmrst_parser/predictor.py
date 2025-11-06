import json
import os
import razdel
import torch
from bisect import bisect_right
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Sequence, Tuple

from isanlp_rst.base_predictor import BasePredictor
from isanlp_rst.utils.du_converter import DUConverter
from .src.parser.data import Data
from .src.parser.parsing_net import ParsingNet


def str2bool(value):
    if type(value) == bool:
        return value

    if type(value) == str:
        return value.lower() == 'true'


class PredictorDMRST(BasePredictor):
    def __init__(self,
                 model_dir: str = None,
                 hf_model_name: str = None,
                 hf_model_version: str = None,
                 cuda_device: int = -1):

        self.mode = None
        if hf_model_name is not None:
            self.mode = 'hf'
        if model_dir is not None:
            self.mode = 'local'

        assert self.mode is not None

        _file_model = 'best_weights.pt'
        _file_config = 'config.json'
        _file_relation_table = 'relation_table.txt'

        if self.mode == 'local':
            self.model_file = os.path.join(model_dir, _file_model)
            self.config_path = os.path.join(model_dir, _file_config)
            self.relation_table = open(os.path.join(model_dir, _file_relation_table), 'r').read().splitlines()

        elif self.mode == 'hf':
            self.hf_model_name = hf_model_name
            self.hf_model_version = hf_model_version
            self.model_file = hf_hub_download(repo_id=self.hf_model_name,
                                              filename=_file_model,
                                              revision=self.hf_model_version)
            self.config_path = hf_hub_download(repo_id=self.hf_model_name,
                                               filename=_file_config,
                                               revision=self.hf_model_version)
            self.relation_table = open(hf_hub_download(repo_id=self.hf_model_name,
                                                       filename=_file_relation_table,
                                                       revision=self.hf_model_version), 'r').read().splitlines()

        self.config = json.load(open(self.config_path))
        self._cuda_device = torch.device('cpu' if cuda_device == -1 else f'cuda:{cuda_device}')

        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['transformer']['model_name'],
            use_fast=True,
        )
        self.tokenizer.model_max_length = int(
            1e9)  # The parser relies on a sliding window encoding, so we'll suppress the max_len warning this way.

        transformer_config = AutoConfig.from_pretrained(self.config['model']['transformer']['model_name'])
        transformer = AutoModel.from_config(transformer_config).to(self._cuda_device)

        self.tokenizer.add_tokens(['<P>'])
        transformer.resize_token_embeddings(len(self.tokenizer))

        model_config = {
            'relation_table': self.relation_table,
            'classes_number': len(self.relation_table),
            'transformer': transformer,
            'emb_dim': int(self.config['model']['transformer']['emb_size']),
            'cuda_device': self._cuda_device
        }

        model_config.update(self._get_model_configs())
        self.model = ParsingNet(**model_config).to(self._cuda_device)
        self.model.load_state_dict(torch.load(self.model_file, map_location=self._cuda_device))
        self.model.eval()

    def _get_model_configs(self):
        config = {}

        if 'normalize' in self.config['model']['transformer']:
            config['normalize_embeddings'] = self.config['model']['transformer'].get('normalize')

        if 'hidden_size' in self.config['model']:
            hidden_size = int(self.config['model'].get('hidden_size'))
            config['hidden_size'] = hidden_size
            config['decoder_input_size'] = hidden_size
            config['classifier_input_size'] = hidden_size
            config['classifier_hidden_size'] = hidden_size

        if 'type' in self.config['model']['segmenter']:
            config['segmenter_type'] = self.config['model']['segmenter'].get('type')

        if 'hidden_dim' in self.config['model']['segmenter']:
            config['segmenter_hidden_dim'] = int(self.config['model']['segmenter'].get('hidden_dim'))

        if 'lstm_num_layers' in self.config['model']['segmenter']:
            config['segmenter_lstm_num_layers'] = self.config['model']['segmenter'].get('lstm_num_layers')

        if 'lstm_dropout' in self.config['model']['segmenter']:
            config['segmenter_lstm_dropout'] = self.config['model']['segmenter'].get('lstm_dropout')

        if 'lstm_bidirectional' in self.config['model']['segmenter']:
            config['segmenter_lstm_bidirectional'] = str2bool(
                self.config['model']['segmenter'].get('lstm_bidirectional'))

        if 'use_crf' in self.config['model']['segmenter']:
            config['segmenter_use_crf'] = str2bool(self.config['model']['segmenter'].get('use_crf'))

        if 'use_log_crf' in self.config['model']['segmenter']:
            config['segmenter_use_log_crf'] = str2bool(self.config['model']['segmenter'].get('use_log_crf'))

        if 'if_edu_start_loss' in self.config['model']['segmenter']:
            config['segmenter_if_edu_start_loss'] = str2bool(self.config['model']['segmenter'].get('if_edu_start_loss'))

        if 'edu_encoding_kind' in self.config['model']:
            config['edu_encoding_kind'] = self.config['model'].get('edu_encoding_kind')

        if 'du_encoding_kind' in self.config['model']:
            config['du_encoding_kind'] = self.config['model'].get('du_encoding_kind')

        if 'rel_classification_kind' in self.config['model']:
            config['rel_classification_kind'] = self.config['model'].get('rel_classification_kind')

        if 'token_bilstm_hidden' in self.config['model']:
            config['token_bilstm_hidden'] = int(self.config['model'].get('token_bilstm_hidden'))

        return config

    def tokenize(self, data):
        """ Takes data with word level tokenization, run current transformer tokenizer and recount EDU boundaries."""

        def get_offset_mappings(input_ids):
            subwords_str = self.tokenizer.convert_ids_to_tokens(input_ids)

            start, end = 0, 0
            result = []
            for subword in subwords_str:
                if subword.startswith('▁'):
                    if subword != '▁':
                        start += 1

                if subword == '<P>' and start > 0:
                    start += 1
                    end += 1

                end += len(subword)
                result.append((start, end))
                start = end
            return result

        # (word_start_char, word_end_char+1) for each token
        word_offsets = []
        for document in data.input_sentences:
            doc_word_offsets = []
            cur_char = 0
            for word in document:
                doc_word_offsets.append((cur_char, cur_char + len(word)))
                cur_char += len(word) + 1
            word_offsets.append(doc_word_offsets)

        texts = [' '.join(line).strip() for line in data.input_sentences]
        tokens = self.tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
        tokens['entity_ids'] = None
        tokens['entity_position_ids'] = None

        # recount edu_breaks for subwords
        subword_edu_breaks = []
        for doc_word_offsets, doc_subword_offsets, edu_breaks in zip(
                word_offsets, tokens['offset_mapping'], data.edu_breaks):
            subword_edu_breaks.append(self._recount_spans(doc_word_offsets, doc_subword_offsets, edu_breaks))

        return Data(
            input_sentences=tokens['input_ids'],
            entity_ids=tokens['entity_ids'],
            entity_position_ids=tokens['entity_position_ids'],
            sent_breaks=None,
            edu_breaks=subword_edu_breaks,
            decoder_input=data.decoder_input,
            relation_label=data.relation_label,
            parsing_breaks=data.parsing_breaks,
            golden_metric=data.golden_metric,
            parents_index=data.parents_index,
            sibling=data.sibling
        )

    def get_batches(self, data: Data, size: int):
        """ Splits a batch into multiple smaller with given size. """

        if len(data.input_sentences) < size:
            return [data]

        _input_sentences = list(self.divide_chunks(data.input_sentences, size))
        _edu_breaks = list(self.divide_chunks(data.edu_breaks, size))
        _decoder_input = list(self.divide_chunks(data.decoder_input, size))
        _relation_label = list(self.divide_chunks(data.relation_label, size))
        _parsing_breaks = list(self.divide_chunks(data.parsing_breaks, size))
        _golden_metric = list(self.divide_chunks(data.golden_metric, size))

        batches = []
        for (input_sentences, edu_breaks, decoder_input,
             relation_label, parsing_breaks, golden_metric
             ) in tqdm(zip(_input_sentences, _edu_breaks, _decoder_input,
                           _relation_label, _parsing_breaks, _golden_metric), total=len(_input_sentences)):
            batches.append(
                Data(
                    input_sentences=input_sentences,
                    entity_ids=None,
                    entity_position_ids=None,
                    sent_breaks=None,
                    edu_breaks=edu_breaks,
                    decoder_input=decoder_input,
                    relation_label=relation_label,
                    parsing_breaks=parsing_breaks,
                    golden_metric=golden_metric,
                    parents_index=None,
                    sibling=None
                )
            )

        return batches

    def _validate_edus(self, edus: Sequence[str]) -> List[str]:
        if edus is None:
            raise ValueError('`edus` must be provided for parsing.')

        if isinstance(edus, (str, bytes)):
            raise TypeError('`edus` must be a sequence of strings, not a single string.')

        if not isinstance(edus, Sequence):
            raise TypeError('`edus` must be a sequence of strings.')

        if not edus:
            raise ValueError('`edus` must contain at least one EDU.')

        normalized = []
        for idx, edu in enumerate(edus):
            if not isinstance(edu, str):
                raise TypeError(f'EDU at position {idx} must be a string.')
            if not edu:
                raise ValueError(f'EDU at position {idx} is empty.')
            normalized.append(edu)

        return normalized

    @staticmethod
    def _compute_edu_char_spans(edus: Sequence[str]) -> Tuple[str, List[Tuple[int, int]]]:
        text = ' '.join(edus)
        spans: List[Tuple[int, int]] = []
        cursor = 0

        for idx, edu in enumerate(edus):
            start = cursor
            end = start + len(edu)
            if text[start:end] != edu:
                raise ValueError(f'EDU at position {idx} does not align after concatenation.')
            spans.append((start, end))
            if idx < len(edus) - 1:
                cursor = end + 1
            else:
                cursor = end

        return text, spans

    @staticmethod
    def _char_spans_to_token_breaks(tokens, spans: List[Tuple[int, int]]) -> List[int]:
        if not tokens:
            raise ValueError('Unable to derive token boundaries from the provided EDUs.')

        token_stops = [token.stop for token in tokens]
        edu_breaks: List[int] = []
        token_idx = -1

        for span_idx, (_, edu_end) in enumerate(spans):
            while token_idx + 1 < len(token_stops) and token_stops[token_idx + 1] <= edu_end:
                token_idx += 1

            if token_idx == -1 or token_stops[token_idx] != edu_end:
                raise ValueError(
                    f'EDU at position {span_idx} does not align with tokenizer boundaries.'
                )

            edu_breaks.append(token_idx)

        if edu_breaks[-1] != len(token_stops) - 1:
            raise ValueError('EDU boundaries do not cover the entire tokenized text.')

        return edu_breaks

    def parse_rst(self, text: str):
        """
        Parses the given text to generate a tree of rhetorical structure.

        Args:
            text (str): The input text to be parsed.

        Returns:
            dict: Tokens and a tree representing the rhetorical structure based on the input text.
        """

        # Prepare the input data
        razdel_tokens = list(razdel.tokenize(text))
        tokenized_text = [token.text for token in razdel_tokens]
        offset_positions, original_offsets = self.build_offset_converter_from_razdel(razdel_tokens)
        data = {
            'input_sentences': [tokenized_text],
            'edu_breaks': [[]],
            'decoder_input': [[]],
            'relation_label': [[]],
            'parsing_breaks': [[]],
            'golden_metric': [[]],
        }

        if len(tokenized_text) < 3:
            tree = DUConverter.dummy_tree(tokenized_text)
            self.remap_tree_offsets(tree, offset_positions, original_offsets, text)

            return {
                'rst': [tree]
            }

        # Initialize predictions dictionary
        input_data = Data(**data)

        predictions = {
            'tokens': [],
            'spans': [],
            'edu_breaks': [],
            'true_spans': [],
            'true_edu_breaks': []
        }

        # Tokenize the input for the transformer
        batch = self.tokenize(input_data)

        # Perform forward pass
        with torch.inference_mode():
            loss_tree_batch, loss_label_batch, \
                span_batch, label_tuple_batch, predict_edu_breaks = self.model.testing_loss(
                batch.input_sentences, batch.sent_breaks, batch.entity_ids, batch.entity_position_ids,
                batch.edu_breaks, batch.relation_label, batch.parsing_breaks,
                generate_tree=True, use_pred_segmentation=True)

        # Update predictions dictionary
        predictions['tokens'] += [self.tokenizer.convert_ids_to_tokens(text) for text in
                                  batch.input_sentences]
        predictions['spans'] += span_batch
        predictions['edu_breaks'] += predict_edu_breaks
        predictions['true_spans'] += batch.golden_metric
        predictions['true_edu_breaks'] += batch.edu_breaks

        # Convert predictions to a tree structure
        tree = DUConverter(predictions, tokenization_type='default').collect()[0]
        self.remap_tree_offsets(tree, offset_positions, original_offsets, text)

        return {
            'rst': [tree]
        }

    def parse_from_edus(self, edus: Sequence[str]):
        """Parse a document using predefined EDU boundaries."""

        normalized_edus = self._validate_edus(edus)
        text, spans = self._compute_edu_char_spans(normalized_edus)

        razdel_tokens = list(razdel.tokenize(text))
        tokenized_text = [token.text for token in razdel_tokens]
        offset_positions, original_offsets = self.build_offset_converter_from_razdel(razdel_tokens)

        if not tokenized_text:
            raise ValueError('Unable to tokenize text derived from the provided EDUs.')

        if len(normalized_edus) == 1:
            tree = DUConverter.dummy_tree(tokenized_text)
            self.remap_tree_offsets(tree, offset_positions, original_offsets, text)
            leaves: List[str] = []
            self._collect_leaf_texts(tree, leaves)
            if leaves != normalized_edus:
                raise ValueError('Failed to align the provided EDU with the parser output.')
            return {
                'rst': [tree]
            }

        edu_breaks = self._char_spans_to_token_breaks(razdel_tokens, spans)

        num_edus = len(edu_breaks)
        relation_placeholder = [[0] * max(num_edus - 1, 0)]
        parsing_placeholder = [[0] * max(num_edus - 1, 0)]

        data = Data(
            input_sentences=[tokenized_text],
            edu_breaks=[edu_breaks],
            decoder_input=[[]],
            relation_label=relation_placeholder,
            parsing_breaks=parsing_placeholder,
            golden_metric=[[]],
        )

        input_data = data

        predictions = {
            'tokens': [],
            'spans': [],
            'edu_breaks': [],
            'true_spans': [],
            'true_edu_breaks': []
        }

        batch = self.tokenize(input_data)

        with torch.inference_mode():
            loss_tree_batch, loss_label_batch, \
                span_batch, label_tuple_batch, predict_edu_breaks = self.model.testing_loss(
                batch.input_sentences, batch.sent_breaks, batch.entity_ids, batch.entity_position_ids,
                batch.edu_breaks, batch.relation_label, batch.parsing_breaks,
                generate_tree=True, use_pred_segmentation=False)

        predictions['tokens'] += [self.tokenizer.convert_ids_to_tokens(text) for text in
                                  batch.input_sentences]
        predictions['spans'] += span_batch
        predictions['edu_breaks'] += predict_edu_breaks
        predictions['true_spans'] += batch.golden_metric
        predictions['true_edu_breaks'] += batch.edu_breaks

        tree = DUConverter(predictions, tokenization_type='default').collect()[0]

        self.remap_tree_offsets(tree, offset_positions, original_offsets, text)

        leaves: List[str] = []
        self._collect_leaf_texts(tree, leaves)
        if leaves != normalized_edus:
            raise ValueError('The produced segmentation does not match the provided EDUs.')

        return {
            'rst': [tree]
        }
