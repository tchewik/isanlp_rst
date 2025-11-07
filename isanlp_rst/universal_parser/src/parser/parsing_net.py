import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from . import modules
from . import segmenters
from .data import nucs_and_rels
from .discriminator import Discriminator
from .metrics import get_batch_metrics


class ParsingNet(nn.Module):
    def __init__(self, relation_tables, transformer, emb_dim=768, hidden_size=768,
                 window_size=400, window_padding=55,
                 decoder_input_size=768, normalize_embeddings=False, atten_model="Dotproduct", rnn_layers=1,
                 segmenter_type='tony', segmenter_use_sent_boundaries=False, segmenter_hidden_dim=100,
                 segmenter_dropout=0.2,
                 segmenter_lstm_num_layers=1, segmenter_lstm_dropout=0.2, segmenter_lstm_bidirectional=True,
                 segmenter_use_crf=False, segmenter_use_log_crf=True, segmenter_if_edu_start_loss=False,
                 edu_encoding_kind='trainable', du_encoding_kind='avg', rel_classification_kind='default',
                 encoder_document_enc_gru=True, encoder_add_first_and_last=True, edu_embedding_compression_rate=1 / 3,
                 classifier_input_size=768, classifier_hidden_size=768, classes_numbers=None, classifier_bias=True,
                 token_bilstm_hidden=100, label_weights=None, corpora_weights=None,
                 dataset2classifier=None, relation_vocab=None, dataset_masks=None,
                 dropout_e=0.5, dropout_d=0.5, dropout_c=0.5,
                 use_discriminator=False, max_w=190, max_h=20,
                 cuda_device=None, use_amp=False, separated_segmentation=True):

        super(ParsingNet, self).__init__()
        """
            :param transformer: transformers.PreTrainedModel  - LM encoder
            :param emb_dim: int  - word embedding dimension (from LM)
            :param hidden_size: int  - hidden size for encoder and decoder
            :param decoder_input_size: int  - decoder input size
            :param normalize_embeddings: bool  - normalize BERT embeddings for parser
            :param atten_model: str  - pointer attention mechanism for parser, from {'Dotproduct', 'Biaffine'}
            :param rnn_layers: int  - encoder and decoder layer number
            :param segmenter_type: str  - type of the segmentation model from {'linear', 'pointer', and 'tony'}
            :param segmenter_use_sent_boundaries: bool  - whether to use sentence boundaries
            :param segmenter_hidden_dim: int  - hidden size of segmenter if segmenter_type is 'tony'
            :param segmenter_lstm_num_layers: int  - number of LSTM layers if segmenter_type is 'tony'
            :param segmenter_lstm_dropout: int  - LSTM dropout if segmenter_type is 'tony' and *_lstm_num_layers > 1
            :param segmenter_lstm_bidirectional: bool  - turn on BiLSTM if segmenter_type is 'tony'
            :param segmenter_use_crf: bool  - turn on CRF if segmenter_type is 'tony'
            :param segmenter_use_log_crf: bool  - scale CRF loss if segmenter_type is 'tony' and *_use_crf == True
            :param segmenter_if_edu_start_loss: bool  - use second FF to predict EDU beginnings (set True for 'linear')
            :param edu_encoding_kind: str  - EDU encoding from token encodings, from {'avg', 'trainable'}
                                             'avg': averages token encodings provided by the transformer
                                             'trainable': same, but with trainable weights
                                             'gru': E-GRU
                                             'bigru': E-GRU, but with BiGRU
            :param du_encoding_kind: str  - DU encoding for label classification only,
                                            from {'avg', 'trainable', 'none', 'bert'}
                                            'avg': averages EDU encodings (1 DU = 1 vector)
                                            'trainable': same, but with trainable weights (1 DU = 1 vector)
                                            'none': pass EDU encodings directly (1 DU = all EDU encodings)
                                            'bert': outputs of the transformer (1 DU = all token embeddings)
            :param rel_classification_kind: str  - Label classification kind, from {'default', 'with_bimpm'}
            :param encoder_document_enc_gru: bool  - Whether to use document-level GRU encoding in the encoder
            :param encoder_add_first_and_last: bool  - Whether to add first and last embeddings to the EDU encoding
            :param edu_embedding_compression_rate: bool - 1/3 if EDU = concat([first_emb, gru_enc, last_emb]) 
            :param classifier_input_size: int  - classifier input size
            :param classifier_hidden_size: int  - classifier hidden size
            :param classes_number: int  - (Is passed from data_manager, do not set manually) 
            :param classifier_bias: bool  - employ bias in the label classifier
            :param label_weights: None or torch.FloatTensor  - (Is also passed automatically)
            :param dropout_e: float  - dropout rate for encoder
            :param dropout_d: float  - dropout rate for decoder
            :param dropout_c: float  - dropout rate for label classifier
            :param device: torch.device  - (Optional) cuda device if present        
        """

        self.hidden_size = hidden_size
        self.decoder_input_size = decoder_input_size
        self.classifier_input_size = classifier_input_size
        self.classifier_hidden_size = classifier_hidden_size
        self.normalize_embeddings = normalize_embeddings
        self.relation_tables = relation_tables
        self.relation_vocab = relation_vocab if relation_vocab is not None else relation_tables[0]
        self.dataset_masks = [torch.tensor(m, device=cuda_device, dtype=torch.bool) for m in dataset_masks] if dataset_masks is not None else None
        self.classes_numbers = classes_numbers
        self.classifier_bias = classifier_bias
        self.label_weights = label_weights
        self.rnn_layers = rnn_layers
        self.use_discriminator = use_discriminator
        self._cuda_device = cuda_device
        self.use_amp = use_amp
        self.segmenter_type = segmenter_type
        if dataset2classifier is None:
            dataset2classifier = list(range(len(classes_numbers)))
        self.dataset2classifier = dataset2classifier

        if not corpora_weights:
            corpora_weights = [1. for _ in range(len(classes_numbers))]
        self.corpora_weights = corpora_weights

        if separated_segmentation:
            _segmenters = []
            for i in range(len(self.dataset2classifier)):
                if segmenter_type == 'linear':
                    _segmenters.append(segmenters.LinearSegmenter(emb_dim, cuda_device=self._cuda_device))
                elif segmenter_type == 'tony':
                    _segmenters.append(segmenters.ToNySegmenter(emb_dim,
                                                                use_sentence_boundaries=segmenter_use_sent_boundaries,
                                                                use_lstm=bool(segmenter_lstm_num_layers > 0),
                                                                num_layers=segmenter_lstm_num_layers,
                                                                dropout=segmenter_dropout,
                                                                lstm_dropout=segmenter_lstm_dropout,
                                                                bidirectional=segmenter_lstm_bidirectional,
                                                                hidden_dim=segmenter_hidden_dim,
                                                                use_crf=segmenter_use_crf,
                                                                use_log_crf=segmenter_use_log_crf,
                                                                if_edu_start_loss=segmenter_if_edu_start_loss,
                                                                cuda_device=self._cuda_device))
            self.segmenters = nn.ModuleList(_segmenters)

        else:
            if segmenter_type == 'linear':
                self.segmenters = [segmenters.LinearSegmenter(emb_dim, cuda_device=self._cuda_device)]
            elif segmenter_type == 'tony':
                self.segmenters = [segmenters.ToNySegmenter(emb_dim,
                                                            use_sentence_boundaries=segmenter_use_sent_boundaries,
                                                            use_lstm=bool(segmenter_lstm_num_layers > 0),
                                                            num_layers=segmenter_lstm_num_layers,
                                                            dropout=segmenter_dropout,
                                                            lstm_dropout=segmenter_lstm_dropout,
                                                            bidirectional=segmenter_lstm_bidirectional,
                                                            hidden_dim=segmenter_hidden_dim,
                                                            use_crf=segmenter_use_crf,
                                                            use_log_crf=segmenter_use_log_crf,
                                                            if_edu_start_loss=segmenter_if_edu_start_loss,
                                                            cuda_device=self._cuda_device)]

        self.edu_encoding_kind = edu_encoding_kind
        self.encoder_document_enc_gru = encoder_document_enc_gru
        self.encoder_add_first_and_last = encoder_add_first_and_last
        self.edu_embedding_compression_rate = edu_embedding_compression_rate

        self.encoder = modules.EncoderRNN(transformer, emb_dim, hidden_size, rnn_layers, dropout_e,
                                          window_size=window_size, window_padding=window_padding,
                                          edu_encoding_kind=self.edu_encoding_kind,
                                          normalize_embeddings=normalize_embeddings,
                                          segmenters=self.segmenters,
                                          document_enc_gru=self.encoder_document_enc_gru,
                                          add_first_and_last=self.encoder_add_first_and_last,
                                          edu_embedding_compression_rate=self.edu_embedding_compression_rate,
                                          token_bilstm_hidden=token_bilstm_hidden,
                                          corpora_weights=self.corpora_weights,
                                          cuda_device=self._cuda_device)

        self.du_encoding_kind = du_encoding_kind
        if du_encoding_kind == 'trainable':
            self._du_attention = torch.nn.Linear(emb_dim, 1, device=self._cuda_device)

        decoder_input_size = self.decoder_input_size
        decoder_hidden_size = self.decoder_input_size
        if self.encoder_add_first_and_last:
            decoder_input_size *= 3 * self.edu_embedding_compression_rate
            decoder_input_size = int(decoder_input_size)
            decoder_hidden_size *= 3 * self.edu_embedding_compression_rate
            decoder_hidden_size = int(decoder_hidden_size)

        self.decoder = modules.DecoderRNN(decoder_input_size, decoder_hidden_size, rnn_layers, dropout_d,
                                          cuda_device=self._cuda_device)
        self.pointer = modules.PointerAtten(atten_model, hidden_size).to(self._cuda_device)

        self.rel_classification_kind = rel_classification_kind
        if rel_classification_kind == 'default':
            if self.dataset_masks is None:
                label_classifiers = []
                for i in range(len(classes_numbers)):
                    label_classifiers.append(
                        modules.DefaultLabelClassifier(classifier_input_size, classifier_hidden_size,
                                                       classes_numbers[i],
                                                       bias=True, dropout=dropout_c,
                                                       cuda_device=self._cuda_device)
                    )
                self.label_classifiers = nn.ModuleList(label_classifiers)
            else:
                self.label_classifier = modules.DefaultLabelClassifier(
                    classifier_input_size, classifier_hidden_size,
                    len(self.relation_vocab), bias=True,
                    dropout=dropout_c, cuda_device=self._cuda_device)

        self.max_w = max_w
        self.max_h = max_h

    def turn_on_discriminator(self):
        self.use_discriminator = True
        self.down = nn.Sequential(nn.Conv2d(2, 32, (3, self.max_w // 2), 1, device=self._cuda_device), nn.ReLU())
        self.down.apply(self._init_weights)
        self.max_p = nn.MaxPool2d(kernel_size=(3, 3), stride=3)
        self.discriminator = Discriminator(max_w=self.max_w, max_h=self.max_h, device=self._cuda_device)

    @staticmethod
    def _init_weights(layer):
        classname = layer.__class__.__name__
        if (classname.find("Conv") != -1) or (classname.find("Linear") != -1):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)

    def cnn_feat_ext(self, img):
        out = self.down(img)
        return self.max_p(out)

    def forward(self):
        raise RuntimeError('Parsing Network does not have forward process.')

    def training_loss(self, input_texts, sent_breaks, entity_ids, entity_position_ids, edu_breaks,
                      label_index, parsing_index, decoder_input_index, dataset_index):

        # Obtain encoder outputs and last hidden states
        if self.du_encoding_kind == 'bert':
            encoder_outputs, last_hidden_states, total_edu_loss, _, embeddings = self.encoder(input_texts, entity_ids,
                                                                                              entity_position_ids,
                                                                                              edu_breaks,
                                                                                              sent_breaks=sent_breaks,
                                                                                              dataset_index=dataset_index)
        else:
            encoder_outputs, last_hidden_states, total_edu_loss, _ = self.encoder(input_texts, entity_ids,
                                                                                  entity_position_ids, edu_breaks,
                                                                                  sent_breaks=sent_breaks,
                                                                                  dataset_index=dataset_index)

        label_loss_functions = [nn.NLLLoss(weight=label_weights) for label_weights in self.label_weights]
        span_loss_func = nn.NLLLoss()

        loss_label_batch = 0
        loss_tree_batch = torch.FloatTensor([0.0]).to(self._cuda_device)
        loop_label_batch = 0
        loop_tree_batch = 0

        batch_size = len(label_index)
        for i in range(batch_size):

            cur_label_index = torch.tensor(label_index[i]).to(self._cuda_device)
            cur_parsing_index = parsing_index[i]
            cur_decoder_input_index = decoder_input_index[i]
            cur_dataset_index = dataset_index[i]

            d_loss = 0.  # Default value for the elementary trees

            if len(edu_breaks[i]) == 1:
                continue

            elif len(edu_breaks[i]) == 2:
                # Obtain the encoded representations. The dimension: [2,hidden_size]
                cur_encoder_outputs = encoder_outputs[i][:len(edu_breaks[i])]

                if self.du_encoding_kind == 'bert':
                    left_b, middle_b, right_b = 0, edu_breaks[i][0], edu_breaks[i][1]
                    input_left, input_right = self._encode_du_bert(input_texts[i], edu_breaks[i],
                                                                   left_b, middle_b, right_b, embeddings[i])
                    if self.rel_classification_kind != 'bimpm':
                        input_left = torch.mean(input_left, keepdim=True, dim=0)
                        input_right = torch.mean(input_right, keepdim=True, dim=0)
                else:
                    # Use the last hidden state of a span to predict the relation between these two span.
                    input_left = cur_encoder_outputs[0].unsqueeze(0)
                    input_right = cur_encoder_outputs[1].unsqueeze(0)

                    # (1, 1, encoding_size)
                    if self.du_encoding_kind == 'none' and not self.rel_classification_kind == 'with_bimpm':
                        input_left = input_left.unsqueeze(0)
                        input_right = input_right.unsqueeze(0)

                cls_idx = self.dataset2classifier[cur_dataset_index]
                if self.dataset_masks is not None:
                    mask = self.dataset_masks[cls_idx]
                    relation_weights, log_relation_weights = \
                        self.label_classifier(input_left, input_right, mask=mask)
                else:
                    relation_weights, log_relation_weights = \
                        self.label_classifiers[cls_idx](input_left, input_right)

                loss_label_batch += label_loss_functions[cls_idx](log_relation_weights, cur_label_index
                                                                    ) * self.corpora_weights[cur_dataset_index]
                loop_label_batch += 1

            else:

                cur_encoder_outputs = encoder_outputs[i][:len(edu_breaks[i])]
                cur_last_hidden_states = last_hidden_states[:, i, :].unsqueeze(1)
                cur_decoder_hidden = cur_last_hidden_states.contiguous()

                edu_index = [x for x in range(len(cur_encoder_outputs))]
                stacks = ['__StackRoot__', edu_index]

                for j in range(len(cur_decoder_input_index)):

                    if stacks[-1] != '__StackRoot__':
                        stack_head = stacks[-1]

                        if len(stack_head) < 3:

                            # Will remove this from stacks after computing the relation between these two EDUS
                            if self.du_encoding_kind == 'bert':
                                left_b, middle_b, right_b = 0, cur_parsing_index[j], stack_head[-1]
                                input_left, input_right = self._encode_du_bert(input_texts[i], edu_breaks[i],
                                                                               left_b, middle_b, right_b, embeddings[i])
                            else:
                                input_left = cur_encoder_outputs[cur_parsing_index[j]].unsqueeze(0)
                                input_right = cur_encoder_outputs[stack_head[-1]].unsqueeze(0)

                                # (1, 1, encoding_size)
                                if self.du_encoding_kind == 'none' and not self.rel_classification_kind == 'with_bimpm':
                                    input_left = input_left.unsqueeze(0)
                                    input_right = input_right.unsqueeze(0)

                            assert cur_parsing_index[j] < stack_head[-1], f'{input_texts[i] = }'


                            # keep the last hidden state consistent.
                            cur_decoder_input = torch.mean(
                                cur_encoder_outputs[stack_head], keepdim=True, dim=0).unsqueeze(0)
                            cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input,
                                                                                  last_hidden=cur_decoder_hidden)

                            cls_idx = self.dataset2classifier[cur_dataset_index]
                            if self.dataset_masks is not None:
                                mask = self.dataset_masks[cls_idx]
                                _, log_relation_weights = \
                                    self.label_classifier(input_left, input_right, mask=mask)
                            else:
                                _, log_relation_weights = \
                                    self.label_classifiers[cls_idx](input_left, input_right)

                            loss_label_batch += label_loss_functions[cls_idx](
                                log_relation_weights, cur_label_index[j].unsqueeze(0)
                            ) * self.corpora_weights[cur_dataset_index]

                            del stacks[-1]
                            loop_label_batch += 1

                        else:
                            # Compute Tree Loss
                            # We don't attend to the last EDU of a span to be parsed
                            cur_decoder_input = torch.mean(cur_encoder_outputs[stack_head], keepdim=True,
                                                           dim=0).unsqueeze(0)

                            # Predict the parsing tree break
                            cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input,
                                                                                  last_hidden=cur_decoder_hidden)

                            atten_weights, log_atten_weights = self.pointer(cur_encoder_outputs[stack_head[:-1]],
                                                                            cur_decoder_output.squeeze(0).squeeze(0))
                            cur_ground_index = torch.tensor([int(cur_parsing_index[j]) - int(stack_head[0])]).to(
                                self._cuda_device)

                            loss_tree_batch += span_loss_func(log_atten_weights, cur_ground_index
                                                              ) * self.corpora_weights[cur_dataset_index]

                            if self.du_encoding_kind == 'bert':
                                input_left_du, input_right_du = self._encode_du_bert(input_texts[i], edu_breaks[i],
                                                                                     stack_head[0],
                                                                                     cur_parsing_index[j],
                                                                                     stack_head[-1], embeddings[i])
                            else:
                                input_left_du, input_right_du = self._encode_du(cur_encoder_outputs, stack_head[0],
                                                                                cur_parsing_index[j], stack_head[-1])

                            cls_idx = self.dataset2classifier[cur_dataset_index]
                            if self.dataset_masks is not None:
                                mask = self.dataset_masks[cls_idx]
                                relation_weights, log_relation_weights = \
                                    self.label_classifier(input_left_du, input_right_du, mask=mask)
                            else:
                                relation_weights, log_relation_weights = \
                                    self.label_classifiers[cls_idx](input_left_du, input_right_du)

                            loss_label_batch += label_loss_functions[cls_idx](
                                log_relation_weights, cur_label_index[j].unsqueeze(0)
                            ) * self.corpora_weights[cur_dataset_index]

                            # Stacks stuff
                            stack_left = stack_head[:(cur_parsing_index[j] - stack_head[0] + 1)]
                            stack_right = stack_head[(cur_parsing_index[j] - stack_head[0] + 1):]
                            del stacks[-1]
                            loop_label_batch += 1
                            loop_tree_batch += 1

                            # Remove ONE-EDU part, TWO-EDU span will be removed after classifier in next step
                            if len(stack_right) > 1:
                                stacks.append(stack_right)
                            if len(stack_left) > 1:
                                stacks.append(stack_left)

        loss_label_batch /= loop_label_batch
        loss_tree_batch /= max(loop_tree_batch, 1)

        return loss_tree_batch, loss_label_batch, total_edu_loss

    def testing_loss(self, input_sentence, input_sent_breaks, input_entity_ids, input_entity_position_ids,
                     input_edu_breaks, label_index, parsing_index, generate_tree, use_pred_segmentation, dataset_index):
        '''
            Input:
                input_sentence: [batch_size, length]
                input_EDU_breaks: e.g. [[2,4,6,9],[2,5,8,10,13],[6,8],[6]]
                LabelIndex: e.g. [[0,3,32],[20,11,14,19],[20],[],]
                ParsingIndex: e.g. [[1,2,0],[3,2,0,1],[0],[]]
            Output: log_atten_weights
                Average loss of tree in a batch
                Average loss of relation in a batch
        '''

        # Obtain encoder outputs and last hidden states
        if self.du_encoding_kind == 'bert':
            encoder_outputs, last_hidden_states, _, predict_edu_breaks, embeddings = self.encoder(
                input_sentence, input_entity_ids, input_entity_position_ids,
                input_edu_breaks, sent_breaks=input_sent_breaks, is_test=use_pred_segmentation,
                dataset_index=dataset_index)
        else:
            encoder_outputs, last_hidden_states, _, predict_edu_breaks = self.encoder(
                input_sentence, input_entity_ids, input_entity_position_ids,
                input_edu_breaks, sent_breaks=input_sent_breaks, is_test=use_pred_segmentation,
                dataset_index=dataset_index)

        if use_pred_segmentation:
            edu_breaks = predict_edu_breaks
        else:
            edu_breaks = input_edu_breaks

        label_index = [[0, ] * (len(i) - 1) for i in edu_breaks]
        parsing_index = [[0, ] * (len(i) - 1) for i in edu_breaks]

        label_loss_function = nn.NLLLoss()
        span_loss_function = nn.NLLLoss()

        loss_label_batch = torch.FloatTensor([0.0]).to(self._cuda_device)
        loss_tree_batch = torch.FloatTensor([0.0]).to(self._cuda_device)
        loop_label_batch = 0
        loop_tree_batch = 0

        label_batch = []
        tree_batch = []

        if generate_tree:
            span_batch = []

        for i in range(len(edu_breaks)):

            cur_label = []
            cur_tree = []

            cur_label_index = torch.tensor(label_index[i]).to(self._cuda_device)
            cur_ParsingIndex = parsing_index[i]
            cur_dataset_index = dataset_index[i]

            if len(edu_breaks[i]) == 1:

                # For a sentence containing only ONE EDU, it has no corresponding relation label and parsing tree break.
                tree_batch.append([])
                label_batch.append([])

                if generate_tree:
                    span_batch.append(['NONE'])

            elif len(edu_breaks[i]) == 2:

                # Obtain the encoded representations, the dimension: [2, hidden_size]
                cur_encoder_outputs = encoder_outputs[i][:len(edu_breaks[i])]

                #  Directly run the classifier to obtain predicted label
                if self.du_encoding_kind == 'bert':
                    left_b, middle_b, right_b = 0, edu_breaks[i][0], edu_breaks[i][1]
                    input_left, input_right = self._encode_du_bert(input_sentence[i], edu_breaks[i],
                                                                   left_b, middle_b, right_b, embeddings[i])
                else:
                    # Use the last hidden state of a span to predict the relation between these two span.
                    input_left = cur_encoder_outputs[0].unsqueeze(0)
                    input_right = cur_encoder_outputs[1].unsqueeze(0)

                    # (1, 1, encoding_size)
                    if self.du_encoding_kind == 'none' and self.rel_classification_kind != 'with_bimpm':
                        input_left = input_left.unsqueeze(0)
                        input_right = input_right.unsqueeze(0)

                cls_idx = self.dataset2classifier[cur_dataset_index]
                if self.dataset_masks is not None:
                    mask = self.dataset_masks[cls_idx]
                    relation_weights, log_relation_weights = self.label_classifier(
                        input_left, input_right, mask=mask)
                else:
                    relation_weights, log_relation_weights = self.label_classifiers[cls_idx](
                        input_left, input_right)

                _, topindex = relation_weights.topk(1)
                label_idx = int(topindex[0][0])
                tree_batch.append([0])
                label_batch.append([label_idx])

                loop_label_batch += 1

                if generate_tree:
                    # Generate a span structure: e.g. (1:Nucleus=span:8,9:Satellite=Attribution:12)
                    if self.dataset_masks is not None:
                        nuclearity_left, nuclearity_right, relation_left, relation_right = \
                            nucs_and_rels(label_idx, self.relation_vocab)
                    else:
                        nuclearity_left, nuclearity_right, relation_left, relation_right = \
                        nucs_and_rels(label_idx, self.relation_tables[cls_idx])
                    span = '('
                    span += '1:' + str(nuclearity_left) + '=' + str(relation_left)
                    span += ';entropy=' + '{:.5f}'.format(0.0)
                    span += ':1,2:' + str(nuclearity_right) + '=' + str(relation_right) + ':2)'
                    span_batch.append([span])

            else:
                # Obtain the encoded representations, the dimension: [num_EDU, hidden_size]
                cur_encoder_outputs = encoder_outputs[i][:len(edu_breaks[i])]

                edu_index = [x for x in range(len(cur_encoder_outputs))]
                stacks = ['__StackRoot__', edu_index]

                # Obtain last hidden state
                cur_last_hidden_states = last_hidden_states[:, i, :].unsqueeze(1)
                cur_decoder_hidden = cur_last_hidden_states.contiguous()

                loop_index = 0

                if generate_tree:
                    span = ''

                tmp_decode_step = -1

                while stacks[-1] != '__StackRoot__':
                    stack_head = stacks[-1]

                    if len(stack_head) < 3:

                        tmp_decode_step += 1
                        # Predict relation label
                        if self.du_encoding_kind == 'bert':
                            left_b, middle_b, right_b = 0, stack_head[0], stack_head[-1]
                            input_left, input_right = self._encode_du_bert(input_sentence[i], edu_breaks[i],
                                                                           left_b, middle_b, right_b, embeddings[i])
                        else:
                            input_left = cur_encoder_outputs[stack_head[0]].unsqueeze(0)
                            input_right = cur_encoder_outputs[stack_head[-1]].unsqueeze(0)

                            # (1, 1, encoding_size)
                            if self.du_encoding_kind == 'none' and self.rel_classification_kind != 'with_bimpm':
                                input_left = input_left.unsqueeze(0)
                                input_right = input_right.unsqueeze(0)

                        cls_idx = self.dataset2classifier[cur_dataset_index]
                        if self.dataset_masks is not None:
                            mask = self.dataset_masks[cls_idx]
                            relation_weights, log_relation_weights = self.label_classifier(
                                input_left, input_right, mask=mask)
                        else:
                            relation_weights, log_relation_weights = self.label_classifiers[cls_idx](
                                input_left, input_right)
                        _, topindex = relation_weights.topk(1)
                        label_idx = int(topindex[0][0])
                        cur_label.append(label_idx)

                        # For 2 EDU case, we directly point the first EDU as the current parsing tree break
                        cur_tree.append(stack_head[0])

                        # keep the last hidden state consistent.
                        cur_decoder_input = torch.mean(cur_encoder_outputs[stack_head], keepdim=True, dim=0).unsqueeze(
                            0)
                        cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input,
                                                                              last_hidden=cur_decoder_hidden)

                        loop_label_batch += 1
                        loop_index += 1
                        del stacks[-1]

                        if generate_tree:
                            # To generate a tree structure

                            if self.dataset_masks is not None:
                                nuclearity_left, nuclearity_right, relation_left, relation_right = \
                                        nucs_and_rels(label_idx, self.relation_vocab)
                            else:
                                (nuclearity_left, nuclearity_right, relation_left, relation_right) = nucs_and_rels(
                                    label_idx, self.relation_tables[cls_idx])

                            cur_span = '(' + str(stack_head[0] + 1) + ':' + str(nuclearity_left) + '=' + str(
                                relation_left)
                            cur_span += ';entropy=' + '{:.5f}'.format(0.0)
                            cur_span += ':' + str(stack_head[0] + 1) + ',' + str(stack_head[-1] + 1) + ':' + str(
                                nuclearity_right) + '=' + \
                                       str(relation_right) + ':' + str(stack_head[-1] + 1) + ')'

                            span += ' ' + cur_span

                    else:  # Length of stack_head >= 3

                        tmp_decode_step += 1

                        # Alternative way is to take the last one as the input.
                        # You need to prepare data accordingly for training.
                        cur_decoder_input = torch.mean(cur_encoder_outputs[stack_head], keepdim=True, dim=0
                                                       ).unsqueeze(0)

                        # Predict the parsing tree break
                        cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input,
                                                                              last_hidden=cur_decoder_hidden)
                        atten_weights, log_atten_weights = self.pointer(cur_encoder_outputs[stack_head[:-1]],
                                                                        cur_decoder_output.squeeze(0).squeeze(0))

                        split_values, topindex_tree = atten_weights.topk(1)
                        tree_predict = int(topindex_tree[0][0]) + stack_head[0]
                        split_entropy = self._calculate_normalized_entropy(log_atten_weights)

                        cur_tree.append(tree_predict)

                        if self.du_encoding_kind == 'bert':
                            input_left_du, input_right_du = self._encode_du_bert(input_sentence[i], edu_breaks[i],
                                                                                 stack_head[0], tree_predict,
                                                                                 stack_head[-1], embeddings[i])
                        else:
                            input_left_du, input_right_du = self._encode_du(cur_encoder_outputs, stack_head[0],
                                                                            tree_predict, stack_head[-1])

                        cls_idx = self.dataset2classifier[cur_dataset_index]
                        if self.dataset_masks is not None:
                            mask = self.dataset_masks[cls_idx]
                            relation_weights, log_relation_weights = self.label_classifier(
                                input_left_du, input_right_du, mask=mask)
                        else:
                            relation_weights, log_relation_weights = self.label_classifiers[cls_idx](
                                input_left_du, input_right_du)

                        _, topindex_label = relation_weights.topk(1)
                        label_idx = int(topindex_label[0][0])
                        cur_label.append(label_idx)

                        # Stacks stuff
                        stack_left = stack_head[:(tree_predict - stack_head[0] + 1)]
                        stack_right = stack_head[(tree_predict - stack_head[0] + 1):]

                        del stacks[-1]
                        loop_label_batch += 1
                        loop_tree_batch += 1
                        loop_index += 1

                        # Remove ONE-EDU part
                        if len(stack_right) > 1:
                            stacks.append(stack_right)
                        if len(stack_left) > 1:
                            stacks.append(stack_left)

                        if generate_tree:
                            # Generate a span structure: e.g. (1:Nucleus=span:8,9:Satellite=Attribution:12)
                            cls_idx = self.dataset2classifier[cur_dataset_index]
                            if self.dataset_masks is not None:
                                nuclearity_left, nuclearity_right, relation_left, relation_right = \
                                    nucs_and_rels(label_idx, self.relation_vocab)
                            else:
                                nuclearity_left, nuclearity_right, relation_left, relation_right = \
                                    nucs_and_rels(label_idx, self.relation_tables[cls_idx])

                            cur_span = '(' + str(stack_head[0] + 1) + ':' + str(nuclearity_left) + '=' + str(
                                relation_left)
                            cur_span += ';entropy=' + '{:.5f}'.format(split_entropy)
                            cur_span += ':' + str(tree_predict + 1) + ',' + str(tree_predict + 2) + ':' + str(
                                nuclearity_right) + '=' + \
                                       str(relation_right) + ':' + str(stack_head[-1] + 1) + ')'

                            span += ' ' + cur_span

                tree_batch.append(cur_tree)
                label_batch.append(cur_label)
                if generate_tree:
                    span_batch.append([span.strip()])

        loss_label_batch /= min(1, loop_label_batch)

        if loop_tree_batch == 0:
            loop_tree_batch = 1

        loss_tree_batch /= loop_tree_batch

        loss_label_batch = loss_label_batch.detach().cpu().numpy()
        loss_tree_batch = loss_tree_batch.detach().cpu().numpy()

        merged_label_gold = []
        for tmp_i in label_index:
            merged_label_gold.extend(tmp_i)

        merged_label_pred = []
        for tmp_i in label_batch:
            merged_label_pred.extend(tmp_i)

        return loss_tree_batch, loss_label_batch, (span_batch if generate_tree else None), (
            merged_label_gold, merged_label_pred), edu_breaks

    def _encode_du(self, cur_encoder_outputs, left_boundary, du_break, right_boundary):
        """
        :param cur_encoder_outputs: torch.FloatTensor  - EDU embeddings of shape (n_edus, embedding_dim)
        :param left_boundary: int  - Start boundary of the left DU
        :param du_break: int  - Start boundary of the right DU
        :param right_boundary: int  - End boundary of the right DU
        :return: one DU embedding of size (1, embedding_dim).
        """

        def slice_tensor(tensor, start, end):
            # Somehow slicing works only in place, so this is a shortcut
            return tensor[start:end + 1, :]

        if self.du_encoding_kind == 'last':
            # Just take last EDU as representation
            input_left = cur_encoder_outputs[du_break].unsqueeze(0)
            input_right = cur_encoder_outputs[right_boundary].unsqueeze(0)
        else:
            if self.du_encoding_kind == 'none':
                input_left = slice_tensor(cur_encoder_outputs, left_boundary, du_break).unsqueeze(0)
                input_right = slice_tensor(cur_encoder_outputs, du_break + 1, right_boundary).unsqueeze(0)
            if self.du_encoding_kind == 'avg':
                # Merge EDU representations into discourse unit reps (DMRST default)
                input_left = torch.mean(slice_tensor(cur_encoder_outputs, left_boundary, du_break),
                                        keepdim=True, dim=0)
                input_right = torch.mean(slice_tensor(cur_encoder_outputs, du_break + 1, right_boundary),
                                         keepdim=True, dim=0)

            elif self.du_encoding_kind == 'trainable':
                # Weighted sum of EDU representations with trainable weights
                attn_weights_left = nn.functional.softmax(self._du_attention(
                    slice_tensor(cur_encoder_outputs, left_boundary, du_break)), dim=0)
                attn_weights_right = nn.functional.softmax(self._du_attention(
                    slice_tensor(cur_encoder_outputs, du_break + 1, right_boundary)), dim=0)

                attn_weights_left = (attn_weights_left + 1e-4).clamp(max=1.)  # To avoid zeros
                attn_weights_right = (attn_weights_right + 1e-4).clamp(max=1.)

                input_left = (slice_tensor(cur_encoder_outputs, left_boundary, du_break) * attn_weights_left
                              ).sum(dim=0).unsqueeze(0)
                input_right = (slice_tensor(cur_encoder_outputs, du_break + 1, right_boundary) * attn_weights_right
                               ).sum(dim=0).unsqueeze(0)

        return input_left, input_right

    def _encode_du_bert(self, token_ids, edu_breaks, left_boundary, du_break, right_boundary, embeddings):
        """
        :param token_ids: list  - token ids of shape (n_tokens,)
        :param edu_breaks: list  - positions of all edu breaks in tokens
        :param left_boundary: int  - start EDU position of the left DU
        :param du_break: int  - start EDU position of the right DU
        :param right_boundary: int  - end boundary of the right DU
        :return: token embeddings of size (2, num_tokens, embedding_dim).
        """

        def convert_du_to_tokens(left_b, middle_b, right_b):
            left = 0 if left_b == 0 else edu_breaks[left_b - 1] + 1  # Right neighbor of the previous right boundary
            middle = edu_breaks[middle_b]
            right = edu_breaks[right_b]
            return left, middle, right

        left, middle, right = convert_du_to_tokens(left_boundary, du_break, right_boundary)

        input_left = embeddings[left:middle + 1, :].unsqueeze(0)
        input_right = embeddings[middle + 1:, :].unsqueeze(0)

        if self.rel_classification_kind != 'bimpm':
            input_left = torch.mean(input_left, dim=1)
            input_right = torch.mean(input_right, dim=1)

        return input_left, input_right

    def _calculate_normalized_entropy(self, log_atten_weights: torch.Tensor) -> float:
        """
        Calculates the normalized entropy H(p) / log(N) from the
        log-probabilities (output of the pointer network).

        Args:
            log_atten_weights (torch.Tensor): The log-probabilities
                                              (log-softmax) from self.pointer.
                                              Expected shape: [1, N] or [N].

        Returns:
            float: The normalized entropy, a value between 0.0 (total
                   certainty) and 1.0 (total uncertainty).
        """
        with torch.no_grad():
            N = log_atten_weights.shape[-1]  # Number of splitting options
            if N <= 1:
                return 0.0

            probs = torch.exp(log_atten_weights.detach())
            raw_entropy = -torch.sum(probs * log_atten_weights.detach(), dim=-1)
            max_entropy = torch.log(torch.tensor(N, device=probs.device, dtype=probs.dtype))
            normalized_entropy = raw_entropy / max_entropy
            return normalized_entropy.cpu().item()

    def eval_loss(self, batch, use_pred_segmentation=True, use_org_parseval=True):

        (batch_input_sentences, batch_sent_breaks, batch_entity_ids, batch_entity_position_ids,
         batch_edu_breaks, batch_decoder_inputs, batch_relation_labels,
         batch_parsing_breaks, batch_golden_metrics, dataset_index) = batch

        loss_tree_batch, loss_label_batch, span_batch, _, predict_edu_breaks = self.testing_loss(
            batch_input_sentences, batch_sent_breaks, batch_entity_ids, batch_entity_position_ids,
            batch_edu_breaks, batch_relation_labels, batch_parsing_breaks,
            generate_tree=True, use_pred_segmentation=use_pred_segmentation, dataset_index=dataset_index)

        metrics = get_batch_metrics(span_batch, batch_golden_metrics,
                                    predict_edu_breaks, batch_edu_breaks, use_org_parseval)
        return (loss_tree_batch, loss_label_batch), metrics

