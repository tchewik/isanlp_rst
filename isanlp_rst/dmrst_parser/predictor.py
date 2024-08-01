import json
import os

import razdel
import torch
from isanlp.annotation import Token
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

from .du_converter import DUConverter
from .src.parser.data import Data
from .src.parser.parsing_net import ParsingNet
from .trainer import Trainer


def str2bool(value):
    if type(value) == bool:
        return value

    if type(value) == str:
        return value.lower() == 'true'


class Predictor:
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
            self.relation_table = open(_file_relation_table, 'r').read().splitlines()

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['transformer']['model_name'], use_fast=True)
        transformer = AutoModel.from_pretrained(self.config['model']['transformer']['model_name']).to(self._cuda_device)

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
            subword_edu_breaks.append(Trainer.recount_spans(doc_word_offsets, doc_subword_offsets, edu_breaks))

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

    @staticmethod
    def divide_chunks(_list, n):
        if _list:
            for i in range(0, len(_list), n):
                yield _list[i:min(i + n, len(_list))]
        else:
            yield _list

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

    def parse_rst(self, text: str):
        """
        Parses the given text to generate a tree of rhetorical structure.

        Args:
            text (str): The input text to be parsed.

        Returns:
            The tree representing the rhetorical structure based on the input text.
        """

        #current_hash = self._get_hash(text)

        ## Check if the parsed data for the given text already exists
        #if self._rst_data:
        #    if current_hash in self._rst_data:
        #        return self._rst_data[current_hash]

        # Preprocess the text
        _text = text.replace('-', ' - ').replace('—', ' — ').replace('  ', ' ')
        _text = _text.replace('...', '…').replace('_', ' ')

        # Prepare the input data
        data = {
            'input_sentences': [[token.text for token in razdel.tokenize(_text)]],
            'edu_breaks': [[]],
            'decoder_input': [[]],
            'relation_label': [[]],
            'parsing_breaks': [[]],
            'golden_metric': [[]],
        }

        if len(data['input_sentences'][0]) < 3:
            return DUConverter.dummy_tree(data['input_sentences'][0])

        # Initialize predictions dictionary
        input = Data(**data)

        predictions = {
            'tokens': [],
            'spans': [],
            'edu_breaks': [],
            'true_spans': [],
            'true_edu_breaks': []
        }

        # Tokenize the input for the transformer
        batch = self.tokenize(input)

        # Perform forward pass
        with torch.no_grad():
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
        duc = DUConverter(predictions, tokenization_type='default')
        tree = duc.collect()[0]

        tokens = []
        begin = 0
        for token in tree.text.split(' '):
            tokens.append(Token(text=token, begin=begin, end=begin + len(token)))
            begin += len(token) + 1

        return {
            'tokens': tokens,
            'rst': [tree]
        }
