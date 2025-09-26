import ast
import json
import os
import pickle
import sys
import types
from bisect import bisect_right
from importlib import import_module
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import razdel
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from isanlp.annotation import Token
from .data_manager import DataManager  # noqa: F401 - ensure module is registered for pickle
from .du_converter import DUConverter
from .src.parser.data import Data
from .src.parser.parsing_net import ParsingNet
from .src.parser.parsing_net_bottom_up import ParsingNetBottomUp
from .trainer import Trainer


def str2bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() == 'true'

    return bool(value)


class Predictor:
    _MODULE_ALIASES = {
        'src.universal_parser.data_manager': 'isanlp_rst.universal_parser.data_manager',
        'src.universal_parser.du_converter': 'isanlp_rst.universal_parser.du_converter',
        'src.universal_parser.trainer': 'isanlp_rst.universal_parser.trainer',
        'src.universal_parser.src.corpus.binary_tree': 'isanlp_rst.universal_parser.src.corpus.binary_tree',
        'src.universal_parser.src.corpus.data': 'isanlp_rst.universal_parser.src.corpus.data',
        'src.universal_parser.src.parser.data': 'isanlp_rst.universal_parser.src.parser.data',
        'src.universal_parser.src.parser.modules': 'isanlp_rst.universal_parser.src.parser.modules',
        'src.universal_parser.src.parser.segmenters': 'isanlp_rst.universal_parser.src.parser.segmenters',
        'src.universal_parser.src.parser.parsing_net': 'isanlp_rst.universal_parser.src.parser.parsing_net',
        'src.universal_parser.src.parser.parsing_net_bottom_up': 'isanlp_rst.universal_parser.src.parser.parsing_net_bottom_up',
        'src.universal_parser.src.parser.metrics': 'isanlp_rst.universal_parser.src.parser.metrics',
        'src.universal_parser.src.parser.training_manager': 'isanlp_rst.universal_parser.src.parser.training_manager',
    }
    _aliases_registered = False

    def __init__(
        self,
        model_dir: Optional[str] = None,
        hf_model_name: Optional[str] = None,
        hf_model_version: Optional[str] = None,
        relinventory: str = None,
        relinventory_idx: int = 0,
        cuda_device: int = -1,
    ) -> None:
        self._ensure_module_aliases()

        self.mode: Optional[str] = None
        if hf_model_name is not None:
            self.mode = 'hf'
        if model_dir is not None:
            self.mode = 'local'

        if self.mode is None:
            raise ValueError('Either model_dir or hf_model_name must be provided.')

        self.model_dir = model_dir
        self.hf_model_name = hf_model_name
        self.hf_model_version = hf_model_version

        model_filename = 'best_weights.pt'
        config_filename = 'config.json'

        if self.mode == 'local':
            self.model_file = os.path.join(self.model_dir, model_filename)
            self.config_path = os.path.join(self.model_dir, config_filename)
        else:
            self.model_file = hf_hub_download(
                repo_id=self.hf_model_name,
                filename=model_filename,
                revision=self.hf_model_version,
            )
            self.config_path = hf_hub_download(
                repo_id=self.hf_model_name,
                filename=config_filename,
                revision=self.hf_model_version,
            )

        with open(self.config_path, 'r', encoding='utf8') as f:
            self.config = json.load(f)

        print(self.config)

        corpora = self.config['data']['corpora']
        if isinstance(corpora, str):
            corpora = ast.literal_eval(corpora)
        self.dataset_names = list(corpora)

        self.relinventory = relinventory
        if self.relinventory is None:
            self.relinventory_idx = relinventory_idx
        else:
            self.relinventory_idx = self.dataset_names.index(self.relinventory.strip().lower())

        self.data_managers: List[Optional[object]] = []
        self.relation_tables: List[Sequence[str]] = []
        for corpus_name in self.dataset_names:
            data_manager = self._load_data_manager(corpus_name)
            self.data_managers.append(data_manager)
            if data_manager is not None:
                self.relation_tables.append(data_manager.relation_table)
            else:
                relation_table = self._load_relation_table(corpus_name)
                if relation_table is None:
                    raise FileNotFoundError(
                        f"Could not find relation inventory for corpus '{corpus_name}'. "
                        'Ensure that relation_table files or data manager pickles are packaged with the model.'
                    )
                self.relation_tables.append(relation_table)

        self._cuda_device = torch.device('cpu' if cuda_device == -1 else f'cuda:{cuda_device}')

        self._load_model()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    @classmethod
    def _ensure_module_aliases(cls) -> None:
        if cls._aliases_registered:
            return

        cls._aliases_registered = True
        for alias, target in cls._MODULE_ALIASES.items():
            cls._register_alias(alias, target)

    @staticmethod
    def _register_alias(alias: str, target: str) -> None:
        module = import_module(target)
        sys.modules[alias] = module
        parent_name, _, child_name = alias.rpartition('.')
        if parent_name:
            parent = Predictor._ensure_parent_module(parent_name)
            setattr(parent, child_name, module)

    @staticmethod
    def _ensure_parent_module(name: str):
        if name in sys.modules:
            return sys.modules[name]

        module = types.ModuleType(name)
        sys.modules[name] = module
        parent_name, _, child_name = name.rpartition('.')
        if parent_name:
            parent = Predictor._ensure_parent_module(parent_name)
            setattr(parent, child_name, module)
        return module

    def _resolve_resource(self, relative_path: str) -> Optional[str]:
        if os.path.isabs(relative_path) and os.path.exists(relative_path):
            return relative_path

        if self.mode == 'local':
            if self.model_dir is None:
                return None
            path = os.path.join(self.model_dir, relative_path)
            if os.path.exists(path):
                return path
            return None

        try:
            return hf_hub_download(
                repo_id=self.hf_model_name,
                filename=relative_path,
                revision=self.hf_model_version,
            )
        except Exception:
            return None

    def _corpus_variants(self, corpus_name: str) -> List[str]:
        lower = corpus_name.lower()
        variants = {lower}
        variants.add(lower.replace('.', '_'))
        variants.add(lower.replace('-', '_'))
        if lower.endswith('-tr'):
            variants.add(lower[:-3])
        if lower.endswith('_tr'):
            variants.add(lower[:-3])
        if lower in {'rst-dt-tr', 'rst_dt_tr'}:
            variants.add('rst-dt')
            variants.add('rst_dt')
        if lower in {'gum10-tr', 'gum10_tr'}:
            variants.add('gum10')
            variants.add('gum')
        return [variant for variant in variants if variant]

    def _load_data_manager(self, corpus_name: str):
        candidates = []
        for variant in self._corpus_variants(corpus_name):
            filename = f'data_manager_{variant}.pickle'
            candidates.append(filename)
            candidates.append(os.path.join('data', filename))
            candidates.append(os.path.join('data', 'dms', filename))

        for rel_path in dict.fromkeys(candidates):  # preserve order, drop duplicates
            resolved = self._resolve_resource(rel_path)
            if not resolved:
                continue
            try:
                with open(resolved, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                continue
        return None

    def _load_relation_table(self, corpus_name: str) -> Optional[List[str]]:
        lower = corpus_name.lower()
        if lower == 'rst-dt-tr':
            lower = 'rst-dt'
        elif lower == 'gum10-tr':
            lower = 'gum'
        elif lower == 'gum10_tr':
            lower = 'gum'

        filename = f'relation_table_{lower}.txt'
        resolved = self._resolve_resource(filename)
        if not resolved:
            return None

        with open(resolved, 'r', encoding='utf8') as f:
            return [line.strip() for line in f if line.strip()]

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['transformer']['model_name'],
            use_fast=True,
        )
        transformer = AutoModel.from_pretrained(
            self.config['model']['transformer']['model_name']
        ).to(self._cuda_device)

        self.tokenizer.add_tokens(['<P>'])
        transformer.resize_token_embeddings(len(self.tokenizer))

        rel_tables = self.relation_tables
        has_data_managers = all(dm is not None for dm in self.data_managers)
        use_union = (
            bool(self.config['model'].get('use_union_relations'))
            and len(rel_tables) > 1
        )

        if use_union:
            union_table: List[str] = []
            label2id: Dict[str, int] = {}
            dataset_masks: List[List[bool]] = []
            label_maps: List[List[int]] = []

            for table in rel_tables:
                for lbl in table:
                    key = lbl.lower()
                    if key not in label2id:
                        label2id[key] = len(union_table)
                        union_table.append(key)

            for table in rel_tables:
                mask = [False] * len(union_table)
                mapping_tbl = []
                for lbl in table:
                    uid = label2id[lbl.lower()]
                    mask[uid] = True
                    mapping_tbl.append(uid)
                dataset_masks.append(mask)
                label_maps.append(mapping_tbl)

            self.label_maps = label_maps
            model_relation_tables = rel_tables
            classes_numbers = [len(union_table)]
            dataset2classifier = list(range(len(rel_tables)))
            model_specific_config = {
                'relation_tables': model_relation_tables,
                'relation_vocab': union_table,
                'dataset_masks': dataset_masks,
                'classes_numbers': classes_numbers,
                'dataset2classifier': dataset2classifier,
            }
        else:
            unique_tables: List[Sequence[str]] = []
            mapping: List[int] = []
            for table in rel_tables:
                for idx, unique in enumerate(unique_tables):
                    if list(table) == list(unique):
                        mapping.append(idx)
                        break
                else:
                    mapping.append(len(unique_tables))
                    unique_tables.append(table)

            self.label_maps = None
            model_relation_tables = unique_tables
            classes_numbers = [len(t) for t in unique_tables]
            dataset2classifier = mapping
            model_specific_config = {
                'relation_tables': model_relation_tables,
                'classes_numbers': classes_numbers,
                'dataset2classifier': dataset2classifier,
            }

        model_config = {
            'transformer': transformer,
            'emb_dim': int(self.config['model']['transformer']['emb_size']),
            'cuda_device': self._cuda_device,
        }
        model_config.update(model_specific_config)
        model_config.update(self._get_model_configs())

        parser_type = self.config['model'].get('parser_type', 'top-down')
        model_cls = ParsingNet if parser_type == 'top-down' else ParsingNetBottomUp

        self.model = model_cls(**model_config).to(self._cuda_device)
        self.model.load_state_dict(torch.load(self.model_file, map_location=self._cuda_device))
        self.model.eval()

    def _get_model_configs(self) -> dict:
        config: dict = {}

        transformer_cfg = self.config['model'].get('transformer', {})
        segmenter_cfg = self.config['model'].get('segmenter', {})
        model_cfg = self.config.get('model', {})

        if 'normalize' in transformer_cfg:
            config['normalize_embeddings'] = transformer_cfg.get('normalize')

        if 'window_size' in transformer_cfg:
            config['window_size'] = int(transformer_cfg.get('window_size'))

        if 'window_padding' in transformer_cfg:
            config['window_padding'] = int(transformer_cfg.get('window_padding'))

        if 'hidden_size' in model_cfg:
            hidden_size = int(model_cfg.get('hidden_size'))
            config['hidden_size'] = hidden_size
            config['decoder_input_size'] = hidden_size
            config['classifier_input_size'] = hidden_size
            config['classifier_hidden_size'] = hidden_size

        if 'type' in segmenter_cfg:
            config['segmenter_type'] = segmenter_cfg.get('type')

        if 'hidden_dim' in segmenter_cfg:
            config['segmenter_hidden_dim'] = int(segmenter_cfg.get('hidden_dim'))

        if 'lstm_num_layers' in segmenter_cfg:
            config['segmenter_lstm_num_layers'] = segmenter_cfg.get('lstm_num_layers')

        if 'lstm_dropout' in segmenter_cfg:
            config['segmenter_lstm_dropout'] = segmenter_cfg.get('lstm_dropout')

        if 'lstm_bidirectional' in segmenter_cfg:
            config['segmenter_lstm_bidirectional'] = str2bool(segmenter_cfg.get('lstm_bidirectional'))

        if 'use_crf' in segmenter_cfg:
            config['segmenter_use_crf'] = str2bool(segmenter_cfg.get('use_crf'))

        if 'use_log_crf' in segmenter_cfg:
            config['segmenter_use_log_crf'] = str2bool(segmenter_cfg.get('use_log_crf'))

        if 'use_sent_boundaries' in segmenter_cfg:
            config['segmenter_use_sent_boundaries'] = str2bool(segmenter_cfg.get('use_sent_boundaries'))

        if 'separated' in segmenter_cfg:
            config['separated_segmentation'] = str2bool(segmenter_cfg.get('separated'))

        if 'if_edu_start_loss' in segmenter_cfg:
            config['segmenter_if_edu_start_loss'] = str2bool(segmenter_cfg.get('if_edu_start_loss'))

        if 'edu_encoding_kind' in model_cfg:
            config['edu_encoding_kind'] = model_cfg.get('edu_encoding_kind')

        if 'du_encoding_kind' in model_cfg:
            config['du_encoding_kind'] = model_cfg.get('du_encoding_kind')

        if 'rel_classification_kind' in model_cfg:
            config['rel_classification_kind'] = model_cfg.get('rel_classification_kind')

        if 'token_bilstm_hidden' in model_cfg:
            config['token_bilstm_hidden'] = int(model_cfg.get('token_bilstm_hidden'))

        if 'use_discriminator' in model_cfg:
            config['use_discriminator'] = str2bool(model_cfg.get('use_discriminator'))

        return config

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------
    def tokenize(self, data: Data) -> Data:
        """Takes word-level tokenized data and converts it to transformer subword inputs."""

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
            word_offsets, tokens['offset_mapping'], data.edu_breaks
        ):
            subword_edu_breaks.append(
                Trainer.recount_spans(doc_word_offsets, doc_subword_offsets, edu_breaks)
            )

        if self.label_maps:
            if self.relinventory_idx >= len(self.label_maps):
                raise IndexError(
                    f'relinventory_idx={self.relinventory_idx} is out of bounds for relation inventories '
                    f'of size {len(self.label_maps)}'
                )
            mapping = self.label_maps[self.relinventory_idx]
            remapped = [[mapping[idx] for idx in doc] for doc in data.relation_label]
        else:
            remapped = data.relation_label

        return Data(
            input_sentences=tokens['input_ids'],
            entity_ids=tokens['entity_ids'],
            entity_position_ids=tokens['entity_position_ids'],
            sent_breaks=None,
            edu_breaks=subword_edu_breaks,
            decoder_input=data.decoder_input,
            relation_label=remapped,
            parsing_breaks=data.parsing_breaks,
            golden_metric=data.golden_metric,
            parents_index=data.parents_index,
            sibling=data.sibling,
            dataset_index=[self.relinventory_idx for _ in range(len(data.input_sentences))],
        )

    @staticmethod
    def divide_chunks(_list: Sequence, n: int) -> Iterable[Sequence]:
        if _list:
            for i in range(0, len(_list), n):
                yield _list[i : min(i + n, len(_list))]
        else:
            yield _list

    def get_batches(self, data: Data, size: int) -> List[Data]:
        """Splits a batch into multiple smaller batches of the given size."""

        if len(data.input_sentences) < size:
            return [data]

        _input_sentences = list(self.divide_chunks(data.input_sentences, size))
        _edu_breaks = list(self.divide_chunks(data.edu_breaks, size))
        _decoder_input = list(self.divide_chunks(data.decoder_input, size))
        _relation_label = list(self.divide_chunks(data.relation_label, size))
        _parsing_breaks = list(self.divide_chunks(data.parsing_breaks, size))
        _golden_metric = list(self.divide_chunks(data.golden_metric, size))
        _dataset_index = list(self.divide_chunks(data.dataset_index, size))

        batches = []
        for (
            input_sentences,
            edu_breaks,
            decoder_input,
            relation_label,
            parsing_breaks,
            golden_metric,
            dataset_index,
        ) in tqdm(
            zip(
                _input_sentences,
                _edu_breaks,
                _decoder_input,
                _relation_label,
                _parsing_breaks,
                _golden_metric,
                _dataset_index,
            ),
            total=len(_input_sentences),
        ):
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
                    sibling=None,
                    dataset_index=dataset_index,
                )
            )

        return batches

    # ------------------------------------------------------------------
    # Offset utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _build_offset_converter(
        tokens: Sequence[str],
        offsets: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Tuple[List[int], List[int]]:
        positions: List[int] = []
        originals: List[int] = []
        cursor = 0

        if offsets is None:
            raise ValueError('Offsets must be provided for pre-tokenized input.')

        for idx, (token, (start, end)) in enumerate(zip(tokens, offsets)):
            token_text = token or ''
            for _ in range(len(token_text)):
                positions.append(cursor)
                originals.append(start)
                start += 1
                cursor += 1
            positions.append(cursor)
            originals.append(end)
            if idx != len(tokens) - 1:
                cursor += 1

        if not positions:
            positions = [0]
            originals = [0]

        return positions, originals

    @staticmethod
    def _map_offset(value: int, positions: List[int], originals: List[int]) -> int:
        if not positions:
            return value

        index = bisect_right(positions, value) - 1
        if index < 0:
            index = 0
        elif index >= len(originals):
            index = len(originals) - 1

        return originals[index]

    def _remap_tree_offsets(
        self,
        unit,
        positions: List[int],
        originals: List[int],
        original_text: str,
    ) -> None:
        left = getattr(unit, 'left', None)
        right = getattr(unit, 'right', None)

        if left is not None:
            self._remap_tree_offsets(left, positions, originals, original_text)
        if right is not None:
            self._remap_tree_offsets(right, positions, originals, original_text)

        if left is None and right is None:
            unit.start = self._map_offset(unit.start, positions, originals)
            unit.end = self._map_offset(unit.end, positions, originals)
        else:
            unit.start = (
                left.start if left is not None else self._map_offset(unit.start, positions, originals)
            )
            unit.end = (
                right.end if right is not None else self._map_offset(unit.end, positions, originals)
            )

        unit.text = original_text[unit.start : unit.end]

    @staticmethod
    def _guess_token_offsets(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
        offsets: List[Tuple[int, int]] = []
        cursor = 0
        for token in tokens:
            if not token:
                offsets.append((cursor, cursor))
                continue

            start = cursor
            while start <= len(text) and text[start:start + len(token)] != token:
                start += 1
                if start >= len(text):
                    start = cursor
                    break
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        return offsets

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------
    def parse_rst(
        self,
        text: str,
        tokens: Optional[Sequence[str]] = None,
        token_offsets: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> dict:
        """Parse text into an RST tree.

        Args:
            text: Original document text.
            tokens: Optional pre-tokenized words to avoid internal tokenization.
            token_offsets: Optional character offsets for the provided tokens.

        Returns:
            A dictionary with token annotations and the predicted RST tree.
        """

        if text is None:
            raise ValueError('`text` must be provided for parsing.')

        if tokens is None:
            razdel_tokens = list(razdel.tokenize(text))
            word_tokens = [token.text for token in razdel_tokens]
            offsets = [(token.start, token.stop) for token in razdel_tokens]
        else:
            word_tokens = list(tokens)
            if token_offsets is None:
                offsets = self._guess_token_offsets(text, word_tokens)
            else:
                offsets = list(token_offsets)

        offset_positions, original_offsets = self._build_offset_converter(word_tokens, offsets)
        output_tokens = [
            Token(text=tok, begin=begin, end=end)
            for tok, (begin, end) in zip(word_tokens, offsets)
        ]

        if len(word_tokens) < 3:
            tree = DUConverter.dummy_tree(word_tokens)
            self._remap_tree_offsets(tree, offset_positions, original_offsets, text)
            return {
                'rst': [tree],
            }

        data = {
            'input_sentences': [word_tokens],
            'edu_breaks': [[]],
            'decoder_input': [[]],
            'relation_label': [[]],
            'parsing_breaks': [[]],
            'golden_metric': [[]],
        }

        input_data = Data(**data)

        predictions = {
            'tokens': [],
            'spans': [],
            'edu_breaks': [],
            'true_spans': [],
            'true_edu_breaks': [],
        }

        batch = self.tokenize(input_data)

        with torch.no_grad():
            (
                _loss_tree,
                _loss_label,
                span_batch,
                _label_tuple_batch,
                predict_edu_breaks,
            ) = self.model.testing_loss(
                batch.input_sentences,
                batch.sent_breaks,
                batch.entity_ids,
                batch.entity_position_ids,
                batch.edu_breaks,
                batch.relation_label,
                batch.parsing_breaks,
                generate_tree=True,
                use_pred_segmentation=True,
                dataset_index=batch.dataset_index,
            )

        predictions['tokens'] += [self.tokenizer.convert_ids_to_tokens(text) for text in batch.input_sentences]
        predictions['spans'] += span_batch
        predictions['edu_breaks'] += predict_edu_breaks
        predictions['true_spans'] += batch.golden_metric
        predictions['true_edu_breaks'] += batch.edu_breaks

        print(f'{predictions = }')
        duc = DUConverter(predictions, tokenization_type='default')
        tree = duc.collect(tokens=data['input_sentences'])[0]

        self._remap_tree_offsets(tree, offset_positions, original_offsets, text)

        return {
            'rst': [tree],
        }
