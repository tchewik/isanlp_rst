import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    """
    Input:
        [batch, length]
    Output:
        encoder_output: [batch, length, hidden_size]
        encoder_hidden: [rnn_layers, batch, hidden_size]
    """

    def __init__(self, transformer, word_dim, hidden_size, rnn_layers, dropout, normalize_embeddings,
                 segmenter, edu_encoding_kind, document_enc_gru, add_first_and_last, edu_embedding_compression_rate,
                 window_size, window_padding,
                 edu_dropout=0.3, token_bilstm_hidden=100, cuda_device=None):
        """
            :param transformer: transformers.PreTrainedModel  - LM encoder
            :param word_dim: int  - word embedding dimension (from LM)
            :param hidden_size: int  - hidden size for encoder and decoder
            :param rnn_layers: int  - encoder and decoder layer number
            :param dropout: float  - dropout rate to be applied to the embeddings
            :param segmenter: nn.Module  - segmentation module
            :param edu_encoding_kind: str  - strategy of EDU encoding, {'avg', 'trainable', 'gru', 'bigru'}
            :param document_enc_gru: bool  - whether to pass EDU encodings through document-level GRU
            :param add_first_and_last: bool  - whether to add first and last embeddings to EDU encoding
            :param edu_embedding_compression_rate: float  - rate of the final linear layer for EDU encoding
                                                            (1 for no compressing, 1/3 in the original implementation)
            :param cuda_device: torch.device  - (Optional) cuda device if present
        """

        super(EncoderRNN, self).__init__()

        self._cuda_device = cuda_device
        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.word_dim = word_dim
        self.window_size = window_size
        self.window_padding = window_padding

        self.dropout = nn.Dropout(dropout)
        self.edu_dropout = nn.Dropout(edu_dropout)
        self.transformer = transformer

        self.normalize_embeddings = normalize_embeddings
        if self.normalize_embeddings:
            self.layer_norm = nn.LayerNorm(word_dim, elementwise_affine=True, device=self._cuda_device)

        self.add_first_and_last = add_first_and_last
        self.edu_embedding_compression_rate = edu_embedding_compression_rate
        if self.add_first_and_last and self.edu_embedding_compression_rate < 1.:
            reduce_dim_input_size = self.hidden_size
            if add_first_and_last:
                reduce_dim_input_size += 2 * self.word_dim
            self.reduce_dim_layer = nn.Linear(reduce_dim_input_size,
                                              int(self.hidden_size * 3 * self.edu_embedding_compression_rate),
                                              bias=False, device=self._cuda_device)

        self.segmenter = segmenter
        self.edu_encoding_kind = edu_encoding_kind
        if self.edu_encoding_kind == 'trainable':
            self._edu_attention = torch.nn.Linear(word_dim, 1, device=self._cuda_device)
            self._init_weights(self._edu_attention)
            self._edu_attention_dropout = nn.Dropout(0.2)

        elif self.edu_encoding_kind == 'gru':
            self._edu_gru = nn.GRU(word_dim, word_dim, batch_first=True, device=self._cuda_device)
            self._init_weights(self._edu_gru)

        elif self.edu_encoding_kind == 'bigru':
            self._edu_gru = nn.GRU(word_dim, word_dim // 2, batch_first=True,
                                   bidirectional=True, device=self._cuda_device)
            self._init_weights(self._edu_gru)

        elif self.edu_encoding_kind == 'bilstm':
            self._edu_lstm = nn.LSTM(word_dim, word_dim // 2, batch_first=True,
                                     bidirectional=True, device=self._cuda_device)

        self.document_enc_gru = document_enc_gru
        if self.document_enc_gru:
            self.doc_gru_enc = nn.GRU(word_dim, hidden_size // 2, num_layers=2, batch_first=True, dropout=0.2,
                                      bidirectional=True, device=self._cuda_device)

        self._token_bilstm_hidden = token_bilstm_hidden
        self._embedding_bilstm = nn.LSTM(word_dim, token_bilstm_hidden, num_layers=1,
                                         bidirectional=True, device=self._cuda_device)

    @staticmethod
    def _init_weights(layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform(layer.weight)

        elif type(layer) == nn.GRU:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        elif type(layer) == nn.LSTM:
            for name, param in layer.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, input_tokenized_texts, entity_ids, entity_position_ids,
                edu_breaks, sent_breaks=None, is_test=False):
        all_outputs = []
        all_hidden = []

        # for segmenter initialization
        total_edu_loss = torch.FloatTensor([0.0]).to(self._cuda_device)
        predict_edu_breaks_list = []
        tem_outputs = []

        for i in range(len(input_tokenized_texts)):
            token_ids = torch.LongTensor(input_tokenized_texts[i]).to(self._cuda_device)
            entity_ids_i = torch.LongTensor(entity_ids[i]).to(self._cuda_device) if entity_ids else None
            entity_position_ids_i = torch.LongTensor(entity_position_ids[i]).to(self._cuda_device
                                                                                ) if entity_position_ids else None
            # Shape: (n_subwords, 768)
            embeddings = self._fixed_sliding_window(token_ids, entity_ids_i, entity_position_ids_i)
            if self.normalize_embeddings:
                embeddings = self.layer_norm(embeddings)

            cur_sent_break = sent_breaks[i] if sent_breaks else None
            if is_test:
                cur_edu_break = self.segmenter.test_segment_loss(embeddings.squeeze(), cur_sent_break)
                predict_edu_breaks_list.append(cur_edu_break)
            else:
                cur_edu_break = edu_breaks[i]  # Only gold segmentation for parser during training
                total_edu_loss += self.segmenter.train_segment_loss(embeddings.squeeze(), cur_edu_break, cur_sent_break)

            outputs, hidden = self.encode_edus(self.dropout(embeddings.squeeze(dim=0)), cur_edu_break)
            tem_outputs.append(outputs)
            all_hidden.append(hidden)

        if edu_breaks is not None or not is_test:
            max_edu_break_num = max([len(tmp_l) for tmp_l in edu_breaks])

        if is_test:
            max_edu_break_num = max([len(tmp_l) for tmp_l in predict_edu_breaks_list])

        for output in tem_outputs:
            batch_size, cur_break_num, edu_dim = output.shape
            all_outputs.append(
                torch.cat(
                    [output, torch.zeros(1, max_edu_break_num - cur_break_num, edu_dim).to(self._cuda_device)],
                    dim=1))

        res_merged_output = torch.cat(all_outputs, dim=0)
        res_merged_hidden = torch.cat(all_hidden, dim=1)

        return res_merged_output, res_merged_hidden, total_edu_loss, predict_edu_breaks_list  # , embeddings

    def encode_edus(self, embeddings, cur_edu_break):
        tmp_edus_list = []
        tmp_break_list = [0, ] + [tmp_j + 1 for tmp_j in cur_edu_break]

        for tmp_i in range(len(tmp_break_list) - 1):
            assert tmp_break_list[tmp_i] < tmp_break_list[tmp_i + 1]
            edu_embeddings = embeddings[tmp_break_list[tmp_i]:tmp_break_list[tmp_i + 1], :]  # Shape: (n_subwords, 768)
            edu_embedding = self._encode_edu(edu_embeddings)  # Shape: (1, word_emb_shape)
            tmp_edus_list.append(edu_embedding)

        outputs = torch.cat(tmp_edus_list, dim=0).unsqueeze(dim=0)

        if self.document_enc_gru:
            outputs, hidden = self.doc_gru_enc(outputs)
            hidden = hidden.view(2, 2, 1, int(self.hidden_size / 2))[-1]
            hidden = hidden.transpose(0, 1).view(1, 1, -1).contiguous()

        if self.add_first_and_last:
            first_words = []
            last_words = []
            for tmp_i in range(len(tmp_break_list) - 1):
                first_words.append(embeddings[tmp_break_list[tmp_i]].unsqueeze(dim=0))
                last_words.append(embeddings[tmp_break_list[tmp_i + 1] - 1].unsqueeze(dim=0))

            outputs = torch.cat((outputs, torch.cat(first_words, dim=0).unsqueeze(dim=0),
                                 torch.cat(last_words, dim=0).unsqueeze(dim=0)), dim=2)

            if self.add_first_and_last and self.edu_embedding_compression_rate < 1.:
                outputs = self.reduce_dim_layer(outputs)

        return outputs, hidden

    def _encode_edu(self, edu_embeddings):
        """
        :param edu_embeddings: torch.FloatTensor  - Subwords embeddings of shape (n_subwords, embedding_dim)
        :return: one EDU embedding of size (1, embedding_dim).
        """
        if self.edu_encoding_kind == 'avg':
            return torch.mean(edu_embeddings, dim=0, keepdim=True)

        if self.edu_encoding_kind == 'trainable':
            # (n_subwords, 1)
            attn_weights = self._edu_attention_dropout(F.softmax(self._edu_attention(edu_embeddings), dim=0))

            return (edu_embeddings * attn_weights).sum(dim=0).unsqueeze(0)

            # weighted_sum = attn_weights.unsqueeze(-1) * edu_embeddings
            # summed = torch.sum(weighted_sum, 1)
            # counts = torch.sum(weights, 1).unsqueeze(-1)
            # return summed / counts

        if self.edu_encoding_kind in ('gru', 'bigru'):
            edu_gru_enc, _ = self._edu_gru(edu_embeddings.unsqueeze(0))

            if self.edu_encoding_kind == 'gru':
                return edu_gru_enc[:, -1]

            if self.edu_encoding_kind == 'bigru':
                hidden_size = edu_gru_enc.size(-1) // 2
                forward_output = edu_gru_enc[:, -1, :hidden_size]
                backward_output = edu_gru_enc[:, 0, hidden_size:]
                return torch.cat((forward_output, backward_output), dim=1)

        elif self.edu_encoding_kind == 'bilstm':
            lstm_enc, _ = self._edu_lstm(self.edu_dropout(edu_embeddings.unsqueeze(0)))
            hidden_size = lstm_enc.size(-1) // 2
            forward_output = lstm_enc[:, -1, :hidden_size]
            backward_output = lstm_enc[:, 0, hidden_size:]
            return torch.cat((forward_output, backward_output), dim=1)

    def _fixed_sliding_window(self, token_ids, entity_ids, entity_position_ids, use_bilstm=False):
        """ Sliding window for encoding long sequences. """

        use_entities = entity_ids is not None and entity_position_ids is not None
        if use_entities:
            entity_ids = entity_ids.unsqueeze(0)
            entity_position_ids = entity_position_ids.unsqueeze(0)

        token_ids = token_ids.unsqueeze(0)
        sequence_length = len(token_ids[0])

        if sequence_length < 512:
            # (1, sequence_length, emb_size)
            if use_entities:
                return self.transformer(token_ids, entity_ids=entity_ids, entity_position_ids=entity_position_ids)[0]
            else:
                return self.transformer(token_ids)[0]

        slide_steps = int(np.ceil(sequence_length / self.window_size))
        window_embed_list = []
        for tmp_step in range(slide_steps):
            if tmp_step == 0:
                end = self.window_size + 2 * self.window_padding
                cur_token_ids = token_ids[:, :end]

                if use_entities:
                    # print(f'{entity_position_ids = }') [[[ 27,  ...
                    cur_entities = [i for i, entity_positions in enumerate(entity_position_ids[0])
                                    if entity_positions[0] < end]
                    one_win_res = self.transformer(cur_token_ids,
                                                   entity_ids=entity_ids[:, cur_entities],
                                                   entity_position_ids=entity_position_ids[:, cur_entities]
                                                   )[0][:, :self.window_size, :]
                else:
                    one_win_res = self.transformer(cur_token_ids)[0][:, :self.window_size, :]

            elif tmp_step == slide_steps - 1:
                start = sequence_length - ((sequence_length - (self.window_size * tmp_step)) + 2 * self.window_padding)
                if False:
                    end = start + token_ids[:, start:].shape[1]
                    cur_entities = [i for i, entity_positions in enumerate(entity_position_ids[0])
                                    if start <= entity_positions[0] and max(entity_positions) < end]
                    current_position_ids = entity_position_ids[:, cur_entities].clone()
                    current_position_ids = torch.where(current_position_ids == -1, -1, current_position_ids - start)
                    # print(f'212 ::: {current_position_ids = }, {token_ids[:, start:].shape}')
                    one_win_res = self.transformer(token_ids[:, start:],
                                                   entity_ids=entity_ids[:, cur_entities],
                                                   entity_position_ids=current_position_ids)[0][:, 2 * self.window_padding:, :]
                else:
                    one_win_res = self.transformer(token_ids[:, start:])[0][:, 2 * self.window_padding:, :]
            else:
                start = self.window_size * tmp_step - self.window_padding
                end = self.window_size * (tmp_step + 1) + self.window_padding

                if use_entities:
                    cur_entities = [i for i, entity_positions in enumerate(entity_position_ids[0])
                                    if start <= entity_positions[0] <= max(entity_positions) < end]
                    current_position_ids = entity_position_ids[:, cur_entities].clone()
                    current_position_ids = torch.where(current_position_ids == -1, -1, current_position_ids - start)

                    one_win_res = self.transformer(token_ids[:, start:end],
                                                   entity_ids=entity_ids[:, cur_entities],
                                                   entity_position_ids=current_position_ids
                                                   )[0][:, padding:self.window_size + self.window_padding, :]
                else:
                    one_win_res = self.transformer(token_ids[:, start:end])[0][:,
                                  self.window_padding:self.window_size + self.window_padding, :]

            if use_bilstm:
                one_win_res, _ = self._embedding_bilstm(one_win_res)

            window_embed_list.append(one_win_res)

        embeddings = torch.cat(window_embed_list, dim=1)
        assert embeddings.size(1) == sequence_length

        return embeddings

    def encode_du_pair(self, token_ids, breaking_point, use_bilstm=False):
        """ Encodes the sequence of tokens, returns two matrices: for left and right texts. """
        token_ids = torch.LongTensor(token_ids).to(self._cuda_device)

        # Shape: (1, n_subwords, emb_dim|bilstm_hidden_size)
        embeddings = self._fixed_sliding_window(token_ids, use_bilstm=use_bilstm)
        return embeddings[:, :breaking_point + 1, :], embeddings[:, breaking_point + 1:, :]


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_layers, dropout, cuda_device):
        super(DecoderRNN, self).__init__()

        '''
        Input:
            input: [1,length,input_size]
            initial_hidden_state: [rnn_layer,1,hidden_size]
        Output:
            output: [1,length,input_size]
            hidden_states: [rnn_layer,1,hidden_size]
        '''
        # Define GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers=rnn_layers, batch_first=True,
                          dropout=(0 if rnn_layers == 1 else dropout), device=cuda_device)

    def forward(self, input_hidden_states, last_hidden):
        # Forward through unidirectional GRU
        outputs, hidden = self.gru(input_hidden_states, last_hidden)

        return outputs, hidden


class PointerAtten(nn.Module):
    def __init__(self, atten_model, hidden_size):
        super(PointerAtten, self).__init__()

        '''       
        Input:
            Encoder_outputs: [length,encoder_hidden_size]
            Current_decoder_output: [decoder_hidden_size] 
            Attention_model: 'Biaffine' or 'Dotproduct' 

        Output:
            attention_weights: [1,length]
            log_attention_weights: [1,length]
        '''

        self.atten_model = atten_model
        self.weight1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, cur_decoder_output):

        if self.atten_model == 'Biaffine':

            EW1_temp = self.weight1(encoder_outputs)
            EW1 = torch.matmul(EW1_temp, cur_decoder_output).unsqueeze(1)
            EW2 = self.weight2(encoder_outputs)
            bi_affine = EW1 + EW2
            bi_affine = bi_affine.permute(1, 0)

            # Obtain attention weights and logits (to compute loss)
            atten_weights = F.softmax(bi_affine, 0)
            log_atten_weights = F.log_softmax(bi_affine + 1e-6, 0)

        elif self.atten_model == 'Dotproduct':

            dot_prod = torch.matmul(encoder_outputs, cur_decoder_output).unsqueeze(0)
            # Obtain attention weights and logits (to compute loss)
            atten_weights = F.softmax(dot_prod, 1)
            log_atten_weights = F.log_softmax(dot_prod + 1e-6, 1)

        # Return attention weights and log attention weights
        return atten_weights, log_atten_weights


class DefaultLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, classes_number,
                 bias=True, dropout=0.5, cuda_device=None):
        """
        :param input_size: int  - input size
        :param hidden_size: int  - hidden size of linear DU encoders
        :param classes_number: int  - number of classes
        :param bias: bool  - whether to include bias in bilinear score
        :param dropout: float  - dropout for DU representations
        :param cuda_device: torch.device  - cuda device

        :return relation_weights, log_relation_weights: [1, num_classes], [1, num_classes]
        """

        super(DefaultLabelClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.labelspace_left = nn.Linear(input_size, hidden_size, bias=False)
        self.labelspace_right = nn.Linear(input_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.weight_left = nn.Linear(hidden_size, classes_number, bias=False)
        self.weight_right = nn.Linear(hidden_size, classes_number, bias=False)

        self.weight_bilateral = nn.Bilinear(hidden_size, hidden_size, classes_number, bias=bias,
                                            device=cuda_device)

        self._cuda_device = cuda_device

    def forward(self, input_left, input_right, **kwargs):
        labelspace_left = self.dropout(F.elu(self.labelspace_left(input_left)))
        labelspace_right = self.dropout(F.elu(self.labelspace_right(input_right)))

        output = (self.weight_bilateral(labelspace_left, labelspace_right) + self.weight_left(
            labelspace_left) + self.weight_right(labelspace_right))

        # Obtain relation weights and log relation weights (for loss)
        relation_weights = F.softmax(output, 1)
        log_relation_weights = F.log_softmax(output + 1e-6, 1)

        return relation_weights, log_relation_weights


class DefaultPlusBiMPMClassifier(nn.Module):
    def __init__(self, default_encoder, bimpm_encoder):
        super(DefaultPlusBiMPMClassifier, self).__init__()

        self._default_encoder = default_encoder
        self._bimpm_encoder = bimpm_encoder

        # Update input size of the default encoder
        self._default_encoder.input_size += self._bimpm_encoder.hidden_size * 2 + 1
        self._default_encoder.labelspace_left = nn.Linear(self._default_encoder.input_size,
                                                          self._default_encoder.hidden_size, bias=False)
        self._default_encoder.labelspace_right = nn.Linear(self._default_encoder.input_size,
                                                           self._default_encoder.hidden_size, bias=False)

        self._cuda_device = self._default_encoder._cuda_device

    def forward(self, left_edus, right_edus, left_du, right_du):
        """ Default classifier takes as input averaged DU representations,
            BiMPM computes over sequences of EDUs. """

        # 1. Acquire the BiMPM hidden representations for left and right DU #####

        # (2, batch, hidden_size * 2), (2, batch, hidden_size * 2), (batch, 2)
        bimpm_left, bimpm_right, lengths = self._bimpm_encoder.encode(left_edus, right_edus)

        # (batch_size, self._bimpm_encoder.hidden_size * 2 + 1)
        x_left = torch.cat([bimpm_left.permute(1, 0, 2).contiguous().view(-1, self._bimpm_encoder.hidden_size * 2),
                            lengths[:, :1]], dim=1)
        x_right = torch.cat([bimpm_left.permute(1, 0, 2).contiguous().view(-1, self._bimpm_encoder.hidden_size * 2),
                             lengths[:, :1]], dim=1)

        # 2. Concat bimpm representations & parser's du representations for left and right unit #####

        # (batch_size, self._bimpm_encoder.hidden_size * 2 + 1 + self._default_encoder.input_size)
        x_left = torch.cat([x_left, left_du], dim=1)
        x_right = torch.cat([x_right, right_du], dim=1)

        return self._default_encoder(x_left, x_right)
