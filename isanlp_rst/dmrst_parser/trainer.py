import os.path
import pprint
import random
import re

import fire
import numpy as np
import razdel
import spacy
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from isanlp_rst.dmrst_parser.data_manager import DataManager
from isanlp_rst.dmrst_parser.src.config_reader import ConfigReader
from isanlp_rst.dmrst_parser.src.parser.data import Data
from isanlp_rst.dmrst_parser.src.parser.parsing_net import ParsingNet
from isanlp_rst.dmrst_parser.src.parser.training_manager import TrainingManager


class Trainer:
    def __init__(self,
                 data__corpus='GUM',
                 data__cross_validation=False,
                 data__fold=0,
                 data__data_manager_path='data/data_manager_gum.pickle',
                 data__lang='en',
                 data__second_lang_fraction=0.,
                 data__second_lang_fold=0,
                 model__transformer__model_name='xlm-roberta-base',
                 model__transformer__emb_size=768,
                 model__transformer__freeze_first_n=20,
                 model__transformer__normalize=True,
                 model__segmenter__type='tony',
                 model__segmenter__dropout=0.4,
                 model__segmenter__hidden_dim=768,
                 model__segmenter__if_edu_start_loss=False,
                 model__segmenter__lstm_bidirectional=True,
                 model__segmenter__lstm_dropout=0.2,
                 model__segmenter__lstm_num_layers=1,
                 model__segmenter__use_crf=True,
                 model__segmenter__use_log_crf=True,
                 model__segmenter__use_sent_boundaries=False,
                 model__hidden_size=768,
                 model__edu_encoding_kind='trainable',
                 model__du_encoding_kind='avg',
                 model__rel_classification_kind='default',
                 model__use_rel_weights=True,
                 model__token_bilstm_hidden=256,
                 model__use_discriminator=False,
                 model__discriminator_warmup=0,
                 model__discriminator_alpha=1.,
                 model__dwa_bs=12,
                 trainer__seed=42,
                 trainer__batch_size=1,
                 trainer__combine_batches=False,
                 trainer__epochs=100,
                 trainer__eval_size=1000,
                 trainer__use_amp=False,
                 trainer__grad_norm_value=1.0,
                 trainer__grad_clipping_value=10.0,
                 trainer__lr=1e-4,
                 trainer__lm_lr_mutliplier=0.2,
                 trainer__lr_decay_epoch=1,
                 trainer__lr_decay=0.95,
                 trainer__weight_decay=0.01,
                 trainer__patience=5,
                 trainer__save_path='saves/',
                 trainer__project='dmrst-tsa-dwl',
                 trainer__run_name=None,
                 trainer__config=None,
                 trainer__gpu=-1,
                 ):

        self.data_corpus_name = data__corpus
        if not os.path.isfile(data__data_manager_path):
            dp = DataManager(corpus=data__corpus, cross_validation=data__cross_validation)
            dp.from_rs3()
            dp.save(data__data_manager_path)
        self.data_data_manager = DataManager(corpus=data__corpus).from_pickle(data__data_manager_path)
        self.data_cross_validation = data__cross_validation
        self.data_fold = data__fold
        self.data_lang = data__lang
        self.data_second_lang_fraction = data__second_lang_fraction
        self.data_second_lang_fold = data__second_lang_fold

        self.model_transformer_model_name = model__transformer__model_name
        self.model_transformer_emb_size = model__transformer__emb_size
        self.model_transformer_freeze_first_n = model__transformer__freeze_first_n
        self.model_transformer_normalize = model__transformer__normalize
        self.model_transformer_needs_entities = self.model_transformer_model_name.startswith('studio-ousia/mluke')

        self.model_segmenter_type = model__segmenter__type
        self.model_segmenter_dropout = model__segmenter__dropout
        self.model_segmenter_hidden_dim = model__segmenter__hidden_dim
        self.model_segmenter_if_edu_start_loss = model__segmenter__if_edu_start_loss  # Only for 'tony'
        self.model_segmenter_lstm_bidirectional = model__segmenter__lstm_bidirectional
        self.model_segmenter_lstm_dropout = model__segmenter__lstm_dropout
        self.model_segmenter_lstm_num_layers = model__segmenter__lstm_num_layers
        self.model_segmenter_use_crf = model__segmenter__use_crf
        self.model_segmenter_use_log_crf = model__segmenter__use_log_crf
        self.model_segmenter_use_sent_boundaries = model__segmenter__use_sent_boundaries

        self.model_hidden_size = model__hidden_size
        self.model_edu_encoding_kind = model__edu_encoding_kind
        self.model_du_encoding_kind = model__du_encoding_kind
        self.model_rel_classification_kind = model__rel_classification_kind
        self.model_use_rel_weights = model__use_rel_weights
        self.model_token_bilstm_hidden = model__token_bilstm_hidden
        self.model_use_discriminator = model__use_discriminator
        self.model_discriminator_warmup = model__discriminator_warmup
        self.model_discriminator_alpha = model__discriminator_alpha
        self.model_dwa_bs = model__dwa_bs

        self.trainer_seed = trainer__seed
        self.trainer_batch_size = trainer__batch_size
        self.trainer_combine_batches = trainer__combine_batches
        self.trainer_epochs = trainer__epochs
        self.trainer_eval_size = trainer__eval_size
        self.trainer_use_amp = trainer__use_amp
        self.trainer_grad_norm_value = trainer__grad_norm_value
        self.trainer_grad_clipping_value = trainer__grad_clipping_value
        self.trainer_lr = float(trainer__lr)
        self.trainer_lm_lr_mutliplier = trainer__lm_lr_mutliplier
        self.trainer_lr_decay_epoch = trainer__lr_decay_epoch
        self.trainer_lr_decay = trainer__lr_decay
        self.trainer_weight_decay = trainer__weight_decay
        self.trainer_patience = trainer__patience
        self.trainer_save_path = trainer__save_path
        self.trainer_project = trainer__project
        self.trainer_run_name = trainer__run_name
        self.trainer_config = trainer__config
        cuda = 'cpu' if trainer__gpu == -1 else f'cuda:{trainer__gpu}'
        self._cuda_device = torch.device(cuda)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_transformer_model_name, use_fast=True)
        self._transformer = AutoModel.from_pretrained(self.model_transformer_model_name).to(self._cuda_device)

        self._tokenizer.add_tokens(['<P>'])
        self._transformer.resize_token_embeddings(len(self._tokenizer))

        # if self.model_segmenter_use_sent_boundaries and self.data_lang == 'en':
        #     self._spacy = English()
        #     self._spacy.add_pipe("sentencizer")

        if self.model_transformer_needs_entities:
            if self.data_lang == 'en':
                self._spacy = spacy.load("en_core_web_trf")  # python -m spacy download en_core_web_trf

            elif self.data_lang == 'ru':
                self._spacy = spacy.load("ru_core_news_lg")  # python -m spacy download ru_core_news_lg

        self._set_random_seeds()
        self._load_data()
        self._setup_model()

    def _set_random_seeds(self):
        torch.manual_seed(self.trainer_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.trainer_seed)
        np.random.seed(self.trainer_seed)
        random.seed(self.trainer_seed)

    def _load_data(self):
        if self.data_cross_validation:
            train, val, test = self.data_data_manager.get_fold(self.data_fold,
                                                               lang=self.data_lang,
                                                               mixed=self.data_second_lang_fraction)
        else:
            train, val, test = self.data_data_manager.get_data(lang=self.data_lang,
                                                               mixed=self.data_second_lang_fraction,
                                                               mixed_fold=self.data_second_lang_fold)

        print('Loading train set ...')
        self.data_train = self._tokenize(train)
        self._label_weights = self._get_label_weights(train.relation_label)
        print('Loading validation set ...')
        self.data_val = self._tokenize(val)
        print('Loading test set ...')
        self.data_test = self._tokenize(test)

    def _get_label_weights(self, relation_labels):
        def normalize(a):
            a -= a.min(0, keepdim=True)[0]
            return a / a.max(0, keepdim=True)[0]

        if not self.model_use_rel_weights:
            return None

        all_labels = [label for text in relation_labels for label in text]
        unique, counts = np.unique(all_labels, return_counts=True)
        count = torch.zeros(len(self.data_data_manager.relation_table))
        for i, w in zip(unique, counts):
            count[i] = w

        return (1.01 - normalize(count)).to(self._cuda_device)

    @staticmethod
    def recount_spans(word_offsets, subword_offsets, word_span_boundaries):
        """ Given word span boundaries, recount for subwords. """
        subword_span_boundaries = [0]

        for w_end in word_span_boundaries:
            final_char = word_offsets[w_end][1]
            for i in range(1, len(subword_offsets)):
                if subword_offsets[i][0] < subword_offsets[i][1]:
                    if subword_offsets[i][0] >= final_char:
                        # Smth with LUKE segmentation
                        if i - 1 in subword_span_boundaries:
                            subword_span_boundaries.append(i)
                        else:
                            subword_span_boundaries.append(i - 1)
                        break

        if not len(subword_offsets) - 1 in subword_span_boundaries:
            subword_span_boundaries.append(len(subword_offsets) - 1)

        return subword_span_boundaries[1:]

    def _tokenize(self, data):
        """ Takes data with word level tokenization, run current transformer tokenizer and recount EDU boundaries."""

        def get_offset_mappings(input_ids):
            subwords_str = self._tokenizer.convert_ids_to_tokens(input_ids)

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

        if self.model_transformer_needs_entities:
            entity_spans = []
            for doc in tqdm(self._spacy.pipe(texts, disable=["tok2vec", "tagger", "parser",
                                                             "attribute_ruler", "lemmatizer"]),
                            desc='Extracting entities', total=len(texts)):
                entity_spans.append([(ent.start_char, ent.end_char) for ent in doc.ents])

            # There is no return_offsets_mapping for LUKE models
            tokens = self._tokenizer(texts, entity_spans=entity_spans, add_special_tokens=False)
            tokens['offset_mapping'] = [get_offset_mappings(ids) for ids in tokens['input_ids']]

        else:
            tokens = self._tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
            tokens['entity_ids'] = None
            tokens['entity_position_ids'] = None

        # recount edu_breaks for subwords
        subword_edu_breaks = []
        for doc_word_offsets, doc_subword_offsets, edu_breaks in zip(
                word_offsets, tokens['offset_mapping'], data.edu_breaks):
            subword_edu_breaks.append(self.recount_spans(doc_word_offsets, doc_subword_offsets, edu_breaks))

        # collecting sentence_breaks
        if self.model_segmenter_use_sent_boundaries:
            subword_sent_breaks = [self._get_sentence_breaks(text, offset_mappings)
                                   for text, offset_mappings in zip(data.input_sentences, tokens['offset_mapping'])]
        else:
            subword_sent_breaks = None

        return Data(
            input_sentences=tokens['input_ids'],
            entity_ids=tokens['entity_ids'],
            entity_position_ids=tokens['entity_position_ids'],
            sent_breaks=subword_sent_breaks,
            edu_breaks=subword_edu_breaks,
            decoder_input=data.decoder_input,
            relation_label=data.relation_label,
            parsing_breaks=data.parsing_breaks,
            golden_metric=data.golden_metric,
            parents_index=data.parents_index,
            sibling=data.sibling
        )

    def _get_sentence_breaks(self, tokens, token_offsets):
        text = ' '.join(tokens)
        sent_breaks_chr = []

        if self.data_lang == 'ru':
            for sent in razdel.sentenize(text):
                sent_breaks_chr.append(sent.stop - 1)

        elif self.data_lang == 'en':
            doc = self._spacy(text)
            for sent in doc.sents:
                sent_breaks_chr.append(sent.end_char - 1)

        # Given token offsets, match found sentence breaks
        sent_breaks = []
        cur_sentence = 0
        last_offset = None
        for i, token in enumerate(token_offsets):
            if token == last_offset:
                # Apparently, some offsets are duplicated in the transformer's tokenizer output
                if sent_breaks[-1] == 1:
                    sent_breaks[-1] = 0
                    sent_breaks.append(1)
                else:
                    sent_breaks.append(0)
            else:
                if token[1] - 1 >= sent_breaks_chr[cur_sentence]:
                    cur_sentence += 1
                    sent_breaks.append(1)
                else:
                    sent_breaks.append(0)
            last_offset = token

        return sent_breaks

    def _setup_model(self):

        # Freeze some layers of the transformer. (For the original implementation, set n=3)
        for name, param in self._transformer.named_parameters():
            # gets layer number from all those "encoder.layer.number.*" transformer parameters.
            layer_num = re.findall("layer\.(\d+)\.", name)

            if len(layer_num) > 0 and int(layer_num[0]) >= self.model_transformer_freeze_first_n:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.model = ParsingNet(relation_table=self.data_data_manager.relation_table,
                                transformer=self._transformer,
                                emb_dim=self.model_transformer_emb_size,
                                hidden_size=self.model_hidden_size,
                                decoder_input_size=self.model_hidden_size,
                                normalize_embeddings=self.model_transformer_normalize,
                                segmenter_type=self.model_segmenter_type,
                                segmenter_use_sent_boundaries=self.model_segmenter_use_sent_boundaries,
                                segmenter_hidden_dim=self.model_segmenter_hidden_dim,
                                segmenter_dropout=self.model_segmenter_dropout,
                                segmenter_lstm_num_layers=self.model_segmenter_lstm_num_layers,
                                segmenter_lstm_dropout=self.model_segmenter_lstm_dropout,
                                segmenter_lstm_bidirectional=self.model_segmenter_lstm_bidirectional,
                                segmenter_use_crf=self.model_segmenter_use_crf,
                                segmenter_use_log_crf=self.model_segmenter_use_log_crf,
                                segmenter_if_edu_start_loss=self.model_segmenter_if_edu_start_loss,
                                edu_encoding_kind=self.model_edu_encoding_kind,
                                du_encoding_kind=self.model_du_encoding_kind,
                                rel_classification_kind=self.model_rel_classification_kind,
                                token_bilstm_hidden=self.model_token_bilstm_hidden,
                                classifier_input_size=self.model_hidden_size,
                                classifier_hidden_size=self.model_hidden_size,
                                classes_number=len(self.data_data_manager.relation_table),
                                label_weights=self._label_weights,
                                classifier_bias=True,
                                cuda_device=self._cuda_device,
                                use_amp=self.trainer_use_amp).to(self._cuda_device)

        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.trainer_lr, weight_decay=self.trainer_weight_decay)

    def train(self):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Total trainable parameter number is: ", count_parameters(self.model))

        training_manager = TrainingManager(self.model,
                                           train_data=self.data_train,
                                           dev_data=self.data_val,
                                           test_data=self.data_test,
                                           batch_size=self.trainer_batch_size,
                                           combine_batches=self.trainer_combine_batches,
                                           eval_size=self.trainer_eval_size,
                                           epochs=self.trainer_epochs,
                                           use_amp=self.trainer_use_amp,
                                           lr=self.trainer_lr, transformer_lr_multiplier=self.trainer_lm_lr_mutliplier,
                                           lr_decay_epoch=self.trainer_lr_decay_epoch, lr_decay=self.trainer_lr_decay,
                                           weight_decay=self.trainer_weight_decay,
                                           grad_norm=self.trainer_grad_norm_value,
                                           grad_clipping_value=self.trainer_grad_clipping_value,
                                           use_discriminator=self.model_use_discriminator,
                                           discriminator_warmup=self.model_discriminator_warmup,
                                           discriminator_alpha=self.model_discriminator_alpha,
                                           patience=self.trainer_patience, use_micro_f1=True,
                                           use_dwa_loss=True, dwa_bs=self.model_dwa_bs,
                                           save_dir=self.trainer_save_path,
                                           project=self.trainer_project, run_name=self.trainer_run_name,
                                           config=self.trainer_config)

        best_metrics = training_manager.train()

        print('--------------------------------------------------------------------')
        print('Training Completed! Best metrics:')
        pprint.pprint(best_metrics)


def main(config_file, ext_vars=None):
    config_reader = ConfigReader(config_file, ext_vars)
    trainer = config_reader.read(Trainer)
    trainer.train()


if __name__ == '__main__':
    fire.Fire(main)
