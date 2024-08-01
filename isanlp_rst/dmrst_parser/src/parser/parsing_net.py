import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from . import modules
from . import segmenters
from .bimpm import BiMPM
from .data import nucs_and_rels
from .discriminator import Discriminator
from .metrics import get_batch_metrics


class ParsingNet(nn.Module):
    def __init__(self, relation_table, transformer, emb_dim=768, hidden_size=768,
                 window_size=400, window_padding=55,
                 decoder_input_size=768, normalize_embeddings=False, atten_model="Dotproduct", rnn_layers=1,
                 segmenter_type='tony', segmenter_use_sent_boundaries=False, segmenter_hidden_dim=100,
                 segmenter_dropout=0.2,
                 segmenter_lstm_num_layers=1, segmenter_lstm_dropout=0.2, segmenter_lstm_bidirectional=True,
                 segmenter_use_crf=False, segmenter_use_log_crf=True, segmenter_if_edu_start_loss=False,
                 edu_encoding_kind='trainable', du_encoding_kind='avg', rel_classification_kind='default',
                 encoder_document_enc_gru=True, encoder_add_first_and_last=True, edu_embedding_compression_rate=1 / 3,
                 classifier_input_size=768, classifier_hidden_size=768, classes_number=None, classifier_bias=True,
                 token_bilstm_hidden=100, label_weights=None,
                 dropout_e=0.5, dropout_d=0.5, dropout_c=0.5,
                 use_discriminator=False, max_w=190, max_h=20,
                 cuda_device=None, use_amp=False):

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
        self.relation_table = relation_table
        self.classes_number = classes_number
        self.classifier_bias = classifier_bias
        self.label_weights = label_weights
        self.rnn_layers = rnn_layers
        self.use_discriminator = use_discriminator
        self._cuda_device = cuda_device
        self.use_amp = use_amp
        self.segmenter_type = segmenter_type

        if segmenter_type == 'linear':
            self.segmenter = segmenters.LinearSegmenter(emb_dim, cuda_device=self._cuda_device)
        elif segmenter_type == 'pointer':
            self.segmenter = segmenters.PointerSegmenter(emb_dim, cuda_device=self._cuda_device)
        elif segmenter_type == 'tony':
            self.segmenter = segmenters.ToNySegmenter(emb_dim,
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
                                                      cuda_device=self._cuda_device)

        self.edu_encoding_kind = edu_encoding_kind
        self.encoder_document_enc_gru = encoder_document_enc_gru
        self.encoder_add_first_and_last = encoder_add_first_and_last
        self.edu_embedding_compression_rate = edu_embedding_compression_rate

        self.encoder = modules.EncoderRNN(transformer, emb_dim, hidden_size, rnn_layers, dropout_e,
                                          window_size=window_size, window_padding=window_padding,
                                          edu_encoding_kind=self.edu_encoding_kind,
                                          normalize_embeddings=normalize_embeddings,
                                          segmenter=self.segmenter,
                                          document_enc_gru=self.encoder_document_enc_gru,
                                          add_first_and_last=self.encoder_add_first_and_last,
                                          edu_embedding_compression_rate=self.edu_embedding_compression_rate,
                                          token_bilstm_hidden=token_bilstm_hidden,
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
            self.label_classifier = modules.DefaultLabelClassifier(classifier_input_size, classifier_hidden_size,
                                                                   classes_number,
                                                                   bias=True, dropout=dropout_c,
                                                                   cuda_device=self._cuda_device)
        elif rel_classification_kind == 'with_bimpm':
            default_label_encoder = modules.DefaultLabelClassifier(classifier_input_size, classifier_hidden_size,
                                                                   classes_number, bias=True, dropout=dropout_c,
                                                                   cuda_device=self._cuda_device)
            bimpm_label_encoder = BiMPM(word_dim=hidden_size, hidden_size=token_bilstm_hidden,
                                        class_number=classes_number,
                                        cuda_device=self._cuda_device, use_amp=use_amp)
            self.label_classifier = modules.DefaultPlusBiMPMClassifier(default_encoder=default_label_encoder,
                                                                       bimpm_encoder=bimpm_label_encoder)

        self.max_w = max_w
        self.max_h = max_h
        if self.use_discriminator:
            self.down, self.max_p, self.discriminator = None, None, None
            self.turn_on_discriminator()

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
                      label_index, parsing_index, decoder_input_index):

        # Obtain encoder outputs and last hidden states
        if self.du_encoding_kind == 'bert':
            encoder_outputs, last_hidden_states, total_edu_loss, _, embeddings = self.encoder(input_texts, entity_ids,
                                                                                              entity_position_ids,
                                                                                              edu_breaks,
                                                                                              sent_breaks=sent_breaks)
        else:
            encoder_outputs, last_hidden_states, total_edu_loss, _ = self.encoder(input_texts, entity_ids,
                                                                                  entity_position_ids, edu_breaks,
                                                                                  sent_breaks=sent_breaks)

        label_loss_func = nn.NLLLoss(weight=self.label_weights)
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

                if self.rel_classification_kind == 'with_bimpm':
                    _, log_relation_weights = self.label_classifier(
                        left_edus=input_left.unsqueeze(0),
                        right_edus=input_right.unsqueeze(0),
                        left_du=input_left, right_du=input_right,
                    )
                else:
                    _, log_relation_weights = self.label_classifier(input_left, input_right)

                loss_label_batch += label_loss_func(log_relation_weights, cur_label_index)
                loop_label_batch += 1

            else:
                if self.use_discriminator:
                    true_points = []  # [(point, relation, is_left_edu, is_right_edu), ...]
                    pred_points = []  # [(start_idx, log_softmax_point, relation, is_left_edu, is_right_edu), ...]

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

                            assert cur_parsing_index[j] < stack_head[-1]

                            # keep the last hidden state consistent.
                            cur_decoder_input = torch.mean(
                                cur_encoder_outputs[stack_head], keepdim=True, dim=0).unsqueeze(0)
                            cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input,
                                                                                  last_hidden=cur_decoder_hidden)

                            if self.rel_classification_kind == 'with_bimpm':
                                _, log_relation_weights = self.label_classifier(
                                    left_edus=input_left.unsqueeze(0),
                                    right_edus=input_right.unsqueeze(0),
                                    left_du=input_left, right_du=input_right,
                                )
                            else:
                                _, log_relation_weights = self.label_classifier(input_left, input_right)
                            loss_label_batch += label_loss_func(log_relation_weights, cur_label_index[j].unsqueeze(0))

                            del stacks[-1]
                            loop_label_batch += 1

                            if self.use_discriminator:
                                true_points.append((cur_parsing_index[j], cur_label_index[j],
                                                    True, 0, True))
                                pred_points.append((cur_parsing_index[j], log_relation_weights,
                                                    True, 0, True))

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

                            loss_tree_batch += span_loss_func(log_atten_weights, cur_ground_index)

                            if self.du_encoding_kind == 'bert':
                                input_left_du, input_right_du = self._encode_du_bert(input_texts[i], edu_breaks[i],
                                                                                     stack_head[0],
                                                                                     cur_parsing_index[j],
                                                                                     stack_head[-1], embeddings[i])
                            else:
                                input_left_du, input_right_du = self._encode_du(cur_encoder_outputs, stack_head[0],
                                                                                cur_parsing_index[j], stack_head[-1])

                            if self.rel_classification_kind == 'with_bimpm':
                                dek = self.du_encoding_kind
                                self.du_encoding_kind = 'none'
                                input_left_edus, input_right_edus = self._encode_du(cur_encoder_outputs, stack_head[0],
                                                                                    cur_parsing_index[j],
                                                                                    stack_head[-1])
                                self.du_encoding_kind = dek

                                relation_weights, log_relation_weights = self.label_classifier(
                                    left_edus=input_left_edus,
                                    right_edus=input_right_edus,
                                    left_du=input_left_du, right_du=input_right_du,
                                )
                            else:
                                relation_weights, log_relation_weights = self.label_classifier(input_left_du,
                                                                                               input_right_du)

                            loss_label_batch += label_loss_func(log_relation_weights, cur_label_index[j].unsqueeze(0))

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

                            if self.use_discriminator:
                                is_left_edu = cur_parsing_index[j] == stack_head[0]
                                is_right_edu = cur_parsing_index[j] == stack_head[-1]

                                true_points.append((cur_parsing_index[j], cur_label_index[j],
                                                    is_left_edu, stack_head[0], is_right_edu))

                                _, topindex_label = relation_weights.topk(1)
                                label_pred_idx = int(topindex_label[0][0])

                                pred_points.append((stack_head[0], log_atten_weights, label_pred_idx,
                                                    is_left_edu, is_right_edu))

                if self.use_discriminator:
                    self._example_number = np.random.choice(1000)

                    true_img = self._construct_true_img(true_points)
                    d_true = self.discriminator(true_img)

                    pred_img = self._construct_pred_img(pred_points)
                    d_pred = self.discriminator(pred_img)

                    # Generator loss
                    g_loss = 0.5 * torch.mean((d_pred - 1) ** 2)

                    # Discriminator loss
                    d_loss = 0.5 * (torch.mean((d_true - 1.) ** 2) + torch.mean(d_pred ** 2))
                    d_loss += g_loss

        loss_label_batch /= loop_label_batch
        loss_tree_batch /= max(loop_tree_batch, 1)

        if self.use_discriminator:
            return loss_tree_batch, loss_label_batch, total_edu_loss, d_loss
        else:
            return loss_tree_batch, loss_label_batch, total_edu_loss

    def testing_loss(self, input_sentence, input_sent_breaks, input_entity_ids, input_entity_position_ids,
                     input_edu_breaks, label_index, parsing_index, generate_tree, use_pred_segmentation):
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
                input_edu_breaks, sent_breaks=input_sent_breaks, is_test=use_pred_segmentation)
        else:
            encoder_outputs, last_hidden_states, _, predict_edu_breaks = self.encoder(
                input_sentence, input_entity_ids, input_entity_position_ids,
                input_edu_breaks, sent_breaks=input_sent_breaks, is_test=use_pred_segmentation)

        if use_pred_segmentation:
            edu_breaks = predict_edu_breaks
            if label_index is None and parsing_index is None:
                label_index = [[0, ] * (len(i) - 1) for i in edu_breaks]
                parsing_index = [[0, ] * (len(i) - 1) for i in edu_breaks]
        else:
            edu_breaks = input_edu_breaks

        label_loss_function = nn.NLLLoss(self.label_weights)
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
                    # print('bert encoded! 324')
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

                if self.rel_classification_kind == 'with_bimpm':
                    relation_weights, log_relation_weights = self.label_classifier(
                        left_edus=input_left.unsqueeze(0),
                        right_edus=input_right.unsqueeze(0),
                        left_du=input_left, right_du=input_right
                    )
                else:
                    relation_weights, log_relation_weights = self.label_classifier(input_left, input_right)

                _, topindex = relation_weights.topk(1)
                label_idx = int(topindex[0][0])
                tree_batch.append([0])
                label_batch.append([label_idx])

                if use_pred_segmentation is False:
                    loss_label_batch += label_loss_function(log_relation_weights, cur_label_index)

                loop_label_batch += 1

                if generate_tree:
                    # Generate a span structure: e.g. (1:Nucleus=span:8,9:Satellite=Attribution:12)
                    nuclearity_left, nuclearity_right, relation_left, relation_right = \
                        nucs_and_rels(label_idx, self.relation_table)
                    span = '(1:' + str(nuclearity_left) + '=' + str(relation_left) + \
                           ':1,2:' + str(nuclearity_right) + '=' + str(relation_right) + ':2)'
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

                        if self.rel_classification_kind == 'with_bimpm':
                            relation_weights, log_relation_weights = self.label_classifier(
                                left_edus=input_left.unsqueeze(0),
                                right_edus=input_right.unsqueeze(0),
                                left_du=input_left, right_du=input_right
                            )
                        else:
                            relation_weights, log_relation_weights = self.label_classifier(input_left, input_right)
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

                        if use_pred_segmentation is False:
                            # Align ground true label
                            if loop_index > (len(cur_ParsingIndex) - 1):
                                cur_label_true = cur_label_index[-1]
                            else:
                                cur_label_true = cur_label_index[loop_index]

                            loss_label_batch += label_loss_function(log_relation_weights, cur_label_true.unsqueeze(0))

                        loop_label_batch += 1
                        loop_index += 1
                        del stacks[-1]

                        if generate_tree:
                            # To generate a tree structure
                            (nuclearity_left, nuclearity_right, relation_left, relation_right) = nucs_and_rels(
                                label_idx, self.relation_table)

                            cur_span = '(' + str(stack_head[0] + 1) + ':' + str(nuclearity_left) + '=' + str(
                                relation_left) + \
                                       ':' + str(stack_head[0] + 1) + ',' + str(stack_head[-1] + 1) + ':' + str(
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

                        _, topindex_tree = atten_weights.topk(1)
                        tree_predict = int(topindex_tree[0][0]) + stack_head[0]

                        cur_tree.append(tree_predict)

                        if self.du_encoding_kind == 'bert':
                            input_left_du, input_right_du = self._encode_du_bert(input_sentence[i], edu_breaks[i],
                                                                                 stack_head[0], tree_predict,
                                                                                 stack_head[-1], embeddings[i])
                        else:
                            input_left_du, input_right_du = self._encode_du(cur_encoder_outputs, stack_head[0],
                                                                            tree_predict, stack_head[-1])

                        if self.rel_classification_kind == 'with_bimpm':
                            dek = self.du_encoding_kind
                            self.du_encoding_kind = 'none'
                            input_left_edus, input_right_edus = self._encode_du(cur_encoder_outputs, stack_head[0],
                                                                                tree_predict, stack_head[-1])
                            self.du_encoding_kind = dek

                            relation_weights, log_relation_weights = self.label_classifier(
                                left_edus=input_left_edus,
                                right_edus=input_right_edus,
                                left_du=input_left_du, right_du=input_right_du,
                            )
                        else:
                            relation_weights, log_relation_weights = self.label_classifier(
                                input_left_du, input_right_du)

                        _, topindex_label = relation_weights.topk(1)
                        label_idx = int(topindex_label[0][0])
                        cur_label.append(label_idx)

                        if use_pred_segmentation is False:

                            # Align ground true label and tree
                            if loop_index > (len(cur_ParsingIndex) - 1):
                                cur_label_true = cur_label_index[-1]
                                cur_tree_true = cur_ParsingIndex[-1]
                            else:
                                cur_label_true = cur_label_index[loop_index]
                                cur_tree_true = cur_ParsingIndex[loop_index]

                            temp_ground = max(0, (int(cur_tree_true) - int(stack_head[0])))
                            if temp_ground >= (len(stack_head) - 1):
                                temp_ground = stack_head[-2] - stack_head[0]
                            # Compute Tree Loss
                            cur_ground_index = torch.tensor([temp_ground])
                            cur_ground_index = cur_ground_index.to(self._cuda_device)

                            loss_tree_batch += span_loss_function(log_atten_weights, cur_ground_index)
                            loss_label_batch += label_loss_function(log_relation_weights, cur_label_true.unsqueeze(0))

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
                            nuclearity_left, nuclearity_right, relation_left, relation_right = \
                                nucs_and_rels(label_idx, self.relation_table)

                            cur_span = '(' + str(stack_head[0] + 1) + ':' + str(nuclearity_left) + '=' + str(
                                relation_left) + \
                                       ':' + str(tree_predict + 1) + ',' + str(tree_predict + 2) + ':' + str(
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

    def eval_loss(self, batch, use_pred_segmentation=True, use_org_parseval=True):

        (batch_input_sentences, batch_sent_breaks, batch_entity_ids, batch_entity_position_ids,
         batch_edu_breaks, batch_decoder_inputs, batch_relation_labels,
         batch_parsing_breaks, batch_golden_metrics) = batch

        loss_tree_batch, loss_label_batch, span_batch, label_tuple_batch, predict_edu_breaks = self.testing_loss(
            batch_input_sentences, batch_sent_breaks, batch_entity_ids, batch_entity_position_ids,
            batch_edu_breaks, batch_relation_labels, batch_parsing_breaks,
            generate_tree=True, use_pred_segmentation=use_pred_segmentation)

        metrics = get_batch_metrics(span_batch, batch_golden_metrics,
                                    predict_edu_breaks, batch_edu_breaks, use_org_parseval)
        return (loss_tree_batch, loss_label_batch), metrics

    def _construct_true_img(self, true_points):
        """
        :param true_points: (list)  - [(breaking_point, rel_idx, is_left_edu, is_right_edu), ...]
        :param n: (int)  - Number of edus
        :return: (torch.tensor)  - "image" of the gold tree, two channels
        """

        # Initialize matrices with -2
        x_st = torch.zeros([400, 400], device=self._cuda_device) - 2
        x_nr = torch.zeros([400, 400], device=self._cuda_device) - 2

        # Depth stack tracks current row index for each subtree depth
        depth = None

        for i, (j, rel_idx, is_left_edu, left_start_idx, is_right_edu) in enumerate(true_points):
            if not depth:
                depth = 1

            row = depth

            # Set split point
            x_st[row - 1, j + 1] = 0
            x_nr[row - 1, j + 1] = rel_idx

            # Fill leaf nodes
            if is_left_edu:
                x_st[row, j] = -1
                x_nr[row, j] = -1
            if is_right_edu:
                x_st[row, j + 2] = -1
                x_st[row, j + 2] = -1

            if not is_left_edu or not is_right_edu:
                depth += 1
            else:
                # Pop depth if subtree complete
                depth = 1

        # matrix_to_image_2chan(x_st.detach().cpu().numpy(),
        #                       x_nr.detach().cpu().numpy(),
        #                       filename=f'images/true_{self._example_number}.png')

        # (1, 2, max_h, max_w)
        img = torch.cat((x_st[:self.max_h, :self.max_w].unsqueeze(0),
                         x_nr[:self.max_h, :self.max_w].unsqueeze(0)), dim=0).unsqueeze(0)
        img = self.discriminator.cnn_feat_ext(img)
        return img

    def _construct_pred_img(self, pred_points, kind='stack'):
        """
        :param pred_points: (list)  - [(start_idx, log_softmax_point_prediction, relation, is_left_edu, is_right_edu), ...]
        :param n: (int)  - Number of splits
        :return: (torch.tensor)  - "image" of the gold tree, two channels
        """

        # Initialize matrices with zeros
        x_st = torch.zeros([400, 400], device=self._cuda_device) - 2
        x_nr = torch.zeros([400, 400], device=self._cuda_device) - 2

        if kind == 'stack':
            # Depth stack tracks current row index for each subtree depth
            depth_stack = []

            for i, (start_idx, j_logs, rel_idx, is_left_edu, is_right_edu) in enumerate(pred_points):
                if not depth_stack:
                    depth_stack = [1]

                row = depth_stack[-1]

                # Find the possible split point
                _, topindex_tree = j_logs.topk(1)
                j = int(topindex_tree[0][0]) + start_idx

                # Set split point
                x_st[row - 1, start_idx:start_idx + j_logs.size(1)] += j_logs[0]
                x_nr[row - 1, j + 1] = rel_idx

                # Fill leaf nodes
                if is_left_edu:
                    x_st[row, j] = -1
                    x_nr[row, j] = -1
                if is_right_edu:
                    x_st[row, j + 2] = -1
                    x_st[row, j + 2] = -1

                if not is_left_edu and not is_right_edu:
                    # depth_stack.append(depth_stack[-1] + 1)
                    depth_stack[-1] += 1
                    pass
                else:
                    # Pop depth if subtree complete
                    depth_stack.pop()

        elif kind == 'graph':
            # For sorting by tree level
            preds = []
            for start_idx, j_logs, rel_idx, is_left_edu, is_right_edu in pred_points:
                preds.append((start_idx, start_idx + j_logs[0].shape[-1]))

            tree = construct_tree(preds)
            root = [n for n, d in tree.in_degree() if d == 0][0]

            # Calculate the shortest paths (their length are rows in our matrix)
            rows_dict = dict()  # dictionary {i: row}
            for n in tree.nodes():
                if nx.has_path(tree, root, n):
                    rows_dict[n] = nx.shortest_path_length(tree, root, n)

            for i, (start_idx, j_logs, rel_idx, is_left_edu, is_right_edu) in enumerate(pred_points):
                row = rows_dict.get(i)
                if not row:
                    continue
                else:
                    row += 1

                # Find the possible split point
                _, topindex_tree = j_logs.topk(1)
                j = int(topindex_tree[0][0]) + start_idx

                # Set split point
                x_st[row - 1, start_idx:start_idx + j_logs.size(1)] += j_logs[0]
                x_nr[row - 1, j + 1] = rel_idx

                # Fill leaf nodes
                if is_left_edu:
                    x_st[row, j] = -1
                    x_nr[row, j] = -1
                if is_right_edu:
                    x_st[row, j + 2] = -1
                    x_st[row, j + 2] = -1

        # matrix_to_image_2chan(x_st.detach().cpu().numpy(),
        #                       x_nr.detach().cpu().numpy(),
        #                       filename=f'images/pred_{self._example_number}.png')

        # (1, 2, num_splits, num_edus)
        img = torch.cat((x_st[:self.max_h, :self.max_w].unsqueeze(0),
                         x_nr[:self.max_h, :self.max_w].unsqueeze(0)), dim=0).unsqueeze(0)
        img = self.cnn_feat_ext(img[:, :, :self.max_h, :self.max_w].detach())

        return img


def construct_tree(spans):
    G = nx.DiGraph()

    # Add spans as nodes
    for i, span in enumerate(spans):
        G.add_node(i)

    # Find child nodes
    for i, span1 in enumerate(spans):
        for j, span2 in enumerate(spans):
            if i != j:
                if span1[1] + 1 == span2[0]:
                    concat = (span1[0], span2[1])
                elif span2[1] + 1 == span1[0]:
                    concat = (span2[0], span1[1])
                else:
                    continue

                if concat in spans:
                    parent = spans.index(concat)
                    if parent != i:
                        G.add_edge(parent, i)

    return merge_trees(G)


def merge_trees(G):
    subtrees = list(nx.weakly_connected_components(G))

    while len(subtrees) > 1:
        dists = []
        for i in range(len(subtrees) - 1):
            G1 = G.subgraph(subtrees[i]).copy()
            G2 = G.subgraph(subtrees[i + 1]).copy()
            root1 = [n for n, d in G1.in_degree() if d == 0][0]
            root2 = [n for n, d in G2.in_degree() if d == 0][0]
            dists.append((root1, root2, abs(root1 - root2)))

        # Merge closest subtrees
        min_dist = min(dists, key=lambda x: x[2])
        G1.add_edge(min_dist[0], min_dist[1])
        G = nx.compose(G1, G2)
        subtrees = list(nx.weakly_connected_components(G))

    return G


def matrix_to_image(matrix, filename, min_val=-2, max_val=0):
    """Convert a 2D matrix to a grayscale image

    Args:
        matrix: 2D numpy array
        filename: Output image file
        min_val: Minimum value to map to 0. Defaults to matrix min.
        max_val: Maximum value to map to 255. Defaults to matrix max.

    Returns:
        PIL Image object
    """

    if len(matrix.shape) != 2:
        raise ValueError('Input matrix must be 2D')

    if min_val is None:
        min_val = matrix.min()
    if max_val is None:
        max_val = matrix.max()

    matrix = (matrix - min_val) / (max_val - min_val)
    matrix = (255 * matrix).astype(np.uint8)

    with open(filename, 'wb') as f:
        img = Image.fromarray(matrix)
        img.save(f)
        return img


def matrix_to_image_2chan(matrix1, matrix2, filename, min_val=-2, max_val=0):
    """Convert two 2D matrices to a 2-channel image

    Args:
        matrix1: First 2D numpy array (channel 1)
        matrix2: Second 2D numpy array (channel 2)
        filename: Output image file
        min_val: Minimum value to map to 0. Defaults to matrix min.
        max_val: Maximum value to map to 255. Defaults to matrix max.

    Returns:
        PIL Image object
    """

    shape1 = matrix1.shape
    shape2 = matrix2.shape

    if len(shape1) != 2 or len(shape2) != 2:
        raise ValueError('Input matrices must be 2D')

    if shape1 != shape2:
        raise ValueError('Input matrices must have matching dimensions')

    if min_val is None:
        min_val = min(matrix1.min(), matrix2.min())
    if max_val is None:
        max_val = max(matrix1.max(), matrix2.max())

    matrix1 = (matrix1 - min_val) / (max_val - min_val)
    matrix2 = (matrix2 - min_val) / (max_val - min_val)
    matrix1 = (255 * matrix1).astype(np.uint8)
    matrix2 = (255 * matrix2).astype(np.uint8)

    image = np.dstack((matrix1, matrix2, np.zeros(shape1)))
    img_u8 = (255 * image).astype(np.uint8)

    with open(filename, 'wb') as f:
        img = Image.fromarray(img_u8, mode='RGB')
        img.save(f)
        return img
