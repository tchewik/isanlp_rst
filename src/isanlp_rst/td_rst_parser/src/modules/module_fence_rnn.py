# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.modules import MLP, BertEmbedding, Biaffine, BiLSTM, CharLSTM, BertEmbeddingfinetuning
from src.modules.dropout import IndependentDropout, SharedDropout
from src.utils import Config


class EncoderFenceRnn(nn.Module):
    """

    Args:
        n_words (int):
            Size of the word vocabulary.
        n_feats (int):
            Size of the feat vocabulary.
        n_labels (int):
            Number of labels.
        feat (str):
            Specifies which type of additional feature to use: 'char' | 'bert' | 'tag'.
            'char': Character-level representations extracted by CharLSTM.
            'bert': BERT representations, other pretrained langugae models like `XLNet` are also feasible.
            'tag': POS tag embeddings.
            Default: 'char'.
        n_embed (int):
            Size of word embeddings. Default: 100.
        n_feat_embed (int):
            Size of feature representations. Default: 100.
        n_char_embed (int):
            Size of character embeddings serving as inputs of CharLSTM, required if feat='char'. Default: 50.
        bert (str):
            Specify which kind of language model to use, e.g., 'bert-base-cased' and 'xlnet-base-cased'.
            This is required if feat='bert'. The full list can be found in `transformers`.
            Default: `None`.
        n_bert_layers (int):
            Specify how many last layers to use. Required if feat='bert'.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            Dropout ratio of BERT layers. Required if feat='bert'. Default: .0.
        embed_dropout (float):
            Dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            Dimension of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            Number of LSTM layers. Default: 3.
        lstm_dropout (float):
            Dropout ratio of LSTM. Default: .33.
        n_mlp_span (int):
            Span MLP size. Default: 500.
        n_mlp_label  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            Dropout ratio of MLP layers. Default: .33.
        feat_pad_index (int):
            The index of the padding token in the feat vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    """

    def __init__(self,
                 n_words,
                 n_feats,
                 n_labels,
                 feat='char',
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_span=500,
                 n_mlp_label=100,
                 mlp_dropout=.33,
                 feat_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        if feat == 'char':
            self.feat_embed = CharLSTM(n_chars=n_feats,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=feat_pad_index)
        elif feat == 'bert':
            if kwargs['bert_requires_grad'] == 'False':
                bert_requires_grad = False
            elif kwargs['bert_requires_grad'] == 'True':
                bert_requires_grad = True
            if bert_requires_grad:
                self.feat_embed = BertEmbeddingfinetuning(model=bert,
                                                          n_layers=n_bert_layers,
                                                          n_out=n_feat_embed,
                                                          pad_index=feat_pad_index,
                                                          dropout=mix_dropout)
            else:
                self.feat_embed = BertEmbedding(model=bert,
                                                n_layers=n_bert_layers,
                                                n_out=n_feat_embed,
                                                pad_index=feat_pad_index,
                                                dropout=mix_dropout)
            self.n_feat_embed = self.feat_embed.n_out
        elif feat == 'tag':
            self.feat_embed = nn.Embedding(num_embeddings=n_feats,
                                           embedding_dim=n_feat_embed)
        else:
            raise RuntimeError("The feat type should be in ['char', 'bert', 'tag'].")
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_embed + n_feat_embed,
                           hidden_size=n_lstm_hidden,
                           num_layers=n_lstm_layers,
                           dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)
        self.decoder_layers = n_lstm_layers
        # the MLP layers
        # self.mlp_span_l = MLP(n_in=n_lstm_hidden*2,
        #                       n_out=n_mlp_span,
        #                       dropout=mlp_dropout)
        # self.mlp_span_r = MLP(n_in=n_lstm_hidden*2,
        #                       n_out=n_mlp_span,
        #                       dropout=mlp_dropout)
        self.mlp_span_splitting = MLP(n_in=n_lstm_hidden * 2,
                                      n_out=n_mlp_span,
                                      dropout=mlp_dropout)
        self.mlp_label_l = MLP(n_in=n_lstm_hidden * 2,
                               n_out=n_mlp_label,
                               dropout=mlp_dropout)
        self.mlp_label_r = MLP(n_in=n_lstm_hidden * 2,
                               n_out=n_mlp_label,
                               dropout=mlp_dropout)

        # the Biaffine layers
        # self.span_attn = Biaffine(n_in=n_mlp_span,
        #                           bias_x=True,
        #                           bias_y=False)
        self.label_attn = Biaffine(n_in=n_mlp_label,
                                   n_out=n_labels,
                                   bias_x=True,
                                   bias_y=True)
        # self.crf = CRFConstituency()
        # self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.hx_dense = nn.Linear(2 * n_lstm_hidden, 2 * n_lstm_hidden)

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    def forward(self, words, feats):
        """
        Args:
            words (~torch.LongTensor) [batch_size, seq_len]:
                The word indices.
            feats (~torch.LongTensor):
                The feat indices.
                If feat is 'char' or 'bert', the size of feats should be [batch_size, seq_len, fix_len]
                If 'tag', then the size is [batch_size, seq_len].

        Returns:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The scores of all possible spans.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                The scores of all possible labels on each span.
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1).to('cpu'), True, False)
        x, hidden = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        x_f, x_b = x.chunk(2, -1)
        fencepost = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        decoder_init_state = self._transform_decoder_init_state(hidden)
        # x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        # apply MLPs to the BiLSTM output states
        # span_l = self.mlp_span_l(x)
        # span_r = self.mlp_span_r(x)
        span_split = self.mlp_span_splitting(fencepost)
        label_l = self.mlp_label_l(fencepost)
        label_r = self.mlp_label_r(fencepost)

        # [batch_size, seq_len, seq_len]
        # s_span = self.span_attn(span_l, span_r)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        return fencepost, span_split, decoder_init_state, s_label

    def _transform_decoder_init_state(self, hn):
        # we would generate
        assert isinstance(hn, tuple)
        if isinstance(hn, tuple):
            hn, cn = hn
            _, batch, hidden_size = cn.size()
            # take the last layers
            # [batch, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            cn = torch.cat([cn[-2], cn[-1]], dim=1).unsqueeze(0)
            # take hx_dense to [1, batch, hidden_size]
            cn = self.hx_dense(cn)
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                cn = torch.cat([cn, cn.new_zeros(self.decoder_layers - 1, batch, 2 * hidden_size)], dim=0)
            # hn is tanh(cn)
            hn = torch.tanh(cn)
            hn = (hn, cn)
        else:
            # take the last layers
            # [2, batch, hidden_size]
            hn = hn[-2:]
            # hn [2, batch, hidden_size]
            _, batch, hidden_size = hn.size()
            # first convert hn t0 [batch, 2, hidden_size]
            hn = hn.transpose(0, 1).contiguous()
            # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            hn = hn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [1, batch, hidden_size]
            hn = torch.tanh(self.hx_dense(hn))
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                hn = torch.cat([hn, hn.new_zeros(self.decoder_layers - 1, batch, 2 * hidden_size)], dim=0)
        return hn


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_layers=6, dropout=0.2, decoder_type='lstm'):
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
        if decoder_type == 'gru':
            self.decoder_network = nn.GRU(input_size, hidden_size, rnn_layers, batch_first=True,
                                          dropout=(0 if rnn_layers == 1 else dropout))
        elif decoder_type == 'lstm':
            self.decoder_network = nn.LSTM(input_size, hidden_size, rnn_layers, batch_first=True,
                                           dropout=(0 if rnn_layers == 1 else dropout))

    def forward(self, input_hidden_states, last_hidden):
        # Forward through unidirectional GRU/LSTM
        outputs, hidden = self.decoder_network(input_hidden_states, last_hidden)
        # Return output and final hidden state
        return outputs, hidden


class EncoderFenceDiscourseRnn(nn.Module):
    """
    The implementation of CRF Constituency Parser.
    This parser is also called FANCY (abbr. of Fast and Accurate Neural Crf constituencY) Parser.

    References:
        - Yu Zhang, Houquan Zhou and Zhenghua Li. 2020.
          `Fast and Accurate Neural CRF Constituency Parsing`_.

    Args:
        n_words (int):
            Size of the word vocabulary.
        n_feats (int):
            Size of the feat vocabulary.
        n_labels (int):
            Number of labels.
        feat (str):
            Specifies which type of additional feature to use: 'char' | 'bert' | 'tag'.
            'char': Character-level representations extracted by CharLSTM.
            'bert': BERT representations, other pretrained langugae models like `XLNet` are also feasible.
            'tag': POS tag embeddings.
            Default: 'char'.
        n_embed (int):
            Size of word embeddings. Default: 100.
        n_feat_embed (int):
            Size of feature representations. Default: 100.
        n_char_embed (int):
            Size of character embeddings serving as inputs of CharLSTM, required if feat='char'. Default: 50.
        bert (str):
            Specify which kind of language model to use, e.g., 'bert-base-cased' and 'xlnet-base-cased'.
            This is required if feat='bert'. The full list can be found in `transformers`.
            Default: `None`.
        n_bert_layers (int):
            Specify how many last layers to use. Required if feat='bert'.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            Dropout ratio of BERT layers. Required if feat='bert'. Default: .0.
        embed_dropout (float):
            Dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            Dimension of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            Number of LSTM layers. Default: 3.
        lstm_dropout (float):
            Dropout ratio of LSTM. Default: .33.
        n_mlp_span (int):
            Span MLP size. Default: 500.
        n_mlp_label  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            Dropout ratio of MLP layers. Default: .33.
        feat_pad_index (int):
            The index of the padding token in the feat vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Fast and Accurate Neural CRF Constituency Parsing:
        https://www.ijcai.org/Proceedings/2020/560/
    """

    def __init__(self,
                 n_words,
                 n_feats,
                 n_labels,
                 feat='char',
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_span=500,
                 n_mlp_label=100,
                 mlp_dropout=.33,
                 feat_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        if feat == 'char':
            self.feat_embed = CharLSTM(n_chars=n_feats,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=feat_pad_index)
        elif feat == 'bert':
            self.feat_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=feat_pad_index,
                                            dropout=mix_dropout)
            self.n_feat_embed = self.feat_embed.n_out
        elif feat == 'tag':
            self.feat_embed = nn.Embedding(num_embeddings=n_feats,
                                           embedding_dim=n_feat_embed)
        else:
            raise RuntimeError("The feat type should be in ['char', 'bert', 'tag'].")
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_embed + n_feat_embed,
                           hidden_size=n_lstm_hidden,
                           num_layers=n_lstm_layers,
                           dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)
        self.decoder_layers = n_lstm_layers
        # the MLP layers
        # self.mlp_span_l = MLP(n_in=n_lstm_hidden*2,
        #                       n_out=n_mlp_span,
        #                       dropout=mlp_dropout)
        # self.mlp_span_r = MLP(n_in=n_lstm_hidden*2,
        #                       n_out=n_mlp_span,
        #                       dropout=mlp_dropout)
        self.mlp_span_splitting = MLP(n_in=n_lstm_hidden * 2,
                                      n_out=n_mlp_span,
                                      dropout=mlp_dropout)
        # self.mlp_label_l = MLP(n_in=n_lstm_hidden*2,
        #                        n_out=n_mlp_label,
        #                        dropout=mlp_dropout)
        # self.mlp_label_r = MLP(n_in=n_lstm_hidden*2,
        #                        n_out=n_mlp_label,
        #                        dropout=mlp_dropout)
        #
        #
        # # the Biaffine layers
        # # self.span_attn = Biaffine(n_in=n_mlp_span,
        # #                           bias_x=True,
        # #                           bias_y=False)
        # self.label_attn = Biaffine(n_in=n_mlp_label,
        #                            n_out=n_labels,
        #                            bias_x=True,
        #                            bias_y=True)
        # self.crf = CRFConstituency()
        # self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.hx_dense = nn.Linear(2 * n_lstm_hidden, 2 * n_lstm_hidden)

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    def forward(self, words, feats):
        """
        Args:
            words (~torch.LongTensor) [batch_size, seq_len]:
                The word indices.
            feats (~torch.LongTensor):
                The feat indices.
                If feat is 'char' or 'bert', the size of feats should be [batch_size, seq_len, fix_len]
                If 'tag', then the size is [batch_size, seq_len].

        Returns:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The scores of all possible spans.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                The scores of all possible labels on each span.
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1).cpu(), True, False)
        x, hidden = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        x_f, x_b = x.chunk(2, -1)
        fencepost = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        decoder_init_state = self._transform_decoder_init_state(hidden)
        # x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        # apply MLPs to the BiLSTM output states
        # span_l = self.mlp_span_l(x)
        # span_r = self.mlp_span_r(x)
        span_split = self.mlp_span_splitting(fencepost)
        # label_l = self.mlp_label_l(fencepost)
        # label_r = self.mlp_label_r(fencepost)
        #
        # # [batch_size, seq_len, seq_len]
        # # s_span = self.span_attn(span_l, span_r)
        # # [batch_size, seq_len, seq_len, n_labels]
        # s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        # return fencepost, span_split, decoder_init_state, s_label
        return fencepost, span_split, decoder_init_state

    def _transform_decoder_init_state(self, hn):
        # we would generate
        assert isinstance(hn, tuple)
        if isinstance(hn, tuple):
            hn, cn = hn
            _, batch, hidden_size = cn.size()
            # take the last layers
            # [batch, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            cn = torch.cat([cn[-2], cn[-1]], dim=1).unsqueeze(0)
            # take hx_dense to [1, batch, hidden_size]
            cn = self.hx_dense(cn)
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                cn = torch.cat([cn, cn.new_zeros(self.decoder_layers - 1, batch, 2 * hidden_size)], dim=0)
            # hn is tanh(cn)
            hn = torch.tanh(cn)
            hn = (hn, cn)
        else:
            # take the last layers
            # [2, batch, hidden_size]
            hn = hn[-2:]
            # hn [2, batch, hidden_size]
            _, batch, hidden_size = hn.size()
            # first convert hn t0 [batch, 2, hidden_size]
            hn = hn.transpose(0, 1).contiguous()
            # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            hn = hn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [1, batch, hidden_size]
            hn = torch.tanh(self.hx_dense(hn))
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                hn = torch.cat([hn, hn.new_zeros(self.decoder_layers - 1, batch, 2 * hidden_size)], dim=0)
        return hn


class EncoderFenceDiscourseEduRepRnn(nn.Module):
    """

    Args:
        n_words (int):
            Size of the word vocabulary.
        n_feats (int):
            Size of the feat vocabulary.
        n_labels (int):
            Number of labels.
        feat (str):
            Specifies which type of additional feature to use: 'char' | 'bert' | 'tag'.
            'char': Character-level representations extracted by CharLSTM.
            'bert': BERT representations, other pretrained langugae models like `XLNet` are also feasible.
            'tag': POS tag embeddings.
            Default: 'char'.
        n_embed (int):
            Size of word embeddings. Default: 100.
        n_feat_embed (int):
            Size of feature representations. Default: 100.
        n_char_embed (int):
            Size of character embeddings serving as inputs of CharLSTM, required if feat='char'. Default: 50.
        bert (str):
            Specify which kind of language model to use, e.g., 'bert-base-cased' and 'xlnet-base-cased'.
            This is required if feat='bert'. The full list can be found in `transformers`.
            Default: `None`.
        n_bert_layers (int):
            Specify how many last layers to use. Required if feat='bert'.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            Dropout ratio of BERT layers. Required if feat='bert'. Default: .0.
        embed_dropout (float):
            Dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            Dimension of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            Number of LSTM layers. Default: 3.
        lstm_dropout (float):
            Dropout ratio of LSTM. Default: .33.
        n_mlp_span (int):
            Span MLP size. Default: 500.
        n_mlp_label  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            Dropout ratio of MLP layers. Default: .33.
        feat_pad_index (int):
            The index of the padding token in the feat vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Fast and Accurate Neural CRF Constituency Parsing:
        https://www.ijcai.org/Proceedings/2020/560/
    """

    def __init__(self,
                 n_words,
                 n_feats,
                 n_labels,
                 feat='char',
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_span=500,
                 n_mlp_label=100,
                 mlp_dropout=.33,
                 feat_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        if feat == 'char':
            self.feat_embed = CharLSTM(n_chars=n_feats,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=feat_pad_index)
        elif feat == 'bert':
            self.feat_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=feat_pad_index,
                                            dropout=mix_dropout)
            self.n_feat_embed = self.feat_embed.n_out
        elif feat == 'tag':
            self.feat_embed = nn.Embedding(num_embeddings=n_feats,
                                           embedding_dim=n_feat_embed)
        else:
            raise RuntimeError("The feat type should be in ['char', 'bert', 'tag'].")
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.token_lstm = BiLSTM(input_size=n_embed + n_feat_embed,
                                 hidden_size=n_lstm_hidden,
                                 num_layers=n_lstm_layers,
                                 dropout=lstm_dropout)
        self.token_lstm_dropout = SharedDropout(p=lstm_dropout)

        self.edu_lstm = BiLSTM(input_size=n_lstm_hidden * 2,
                               hidden_size=n_lstm_hidden,
                               num_layers=n_lstm_layers,
                               dropout=lstm_dropout)
        self.edu_lstm_dropout = SharedDropout(p=lstm_dropout)
        self.decoder_layers = n_lstm_layers

        self.mlp_span_splitting = MLP(n_in=n_lstm_hidden * 2,
                                      n_out=n_mlp_span,
                                      dropout=mlp_dropout)

        self.pad_index = pad_index
        self.unk_index = unk_index
        self.hx_dense = nn.Linear(2 * n_lstm_hidden, 2 * n_lstm_hidden)

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)

        return self

    def forward(self, words, feats, edu_break):
        """
        Args:
            words (~torch.LongTensor) [batch_size, seq_len]:
                The word indices.
            feats (~torch.LongTensor):
                The feat indices.
                If feat is 'char' or 'bert', the size of feats should be [batch_size, seq_len, fix_len]
                If 'tag', then the size is [batch_size, seq_len].
            edu_break (~torch.LongTensor) [batch_size, max_edu_num]
                The last indices of EDU

        Returns:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The scores of all possible spans.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                The scores of all possible labels on each span.
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1).to('cpu'), True, False)
        x, hidden = self.token_lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.token_lstm_dropout(x)

        x_f, x_b = x.chunk(2, -1)
        fencepost = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)

        # find edu_boundary_representation
        padded_zero = edu_break.new_zeros(batch_size)
        edu_include_boundary_zero = torch.cat([padded_zero.unsqueeze(-1), edu_break], dim=1)
        edu_len_include_boundary_zero = edu_include_boundary_zero.ne(self.pad_index).sum(1) + 1
        _, edu_len = edu_include_boundary_zero.shape

        edu_boundary_emb = fencepost[torch.arange(batch_size).unsqueeze(1), edu_include_boundary_zero]
        edu_boundary_rep = pack_padded_sequence(edu_boundary_emb, edu_len_include_boundary_zero.to('cpu'), True, False)
        edu_boundary_rep, hidden_edu = self.edu_lstm(edu_boundary_rep)
        edu_boundary_rep, _ = pad_packed_sequence(edu_boundary_rep, True, total_length=edu_len)
        edu_boundary_rep = self.edu_lstm_dropout(edu_boundary_rep)

        # decoder_init_state = self._transform_decoder_init_state(hidden)
        # span_split = self.mlp_span_splitting(fencepost)
        decoder_init_state = self._transform_decoder_init_state(hidden_edu)
        span_split = self.mlp_span_splitting(edu_boundary_rep)

        # return fencepost, span_split, decoder_init_state
        return edu_boundary_rep, span_split, decoder_init_state

    def _transform_decoder_init_state(self, hn):
        # we would generate
        assert isinstance(hn, tuple)
        if isinstance(hn, tuple):
            hn, cn = hn
            _, batch, hidden_size = cn.size()
            # take the last layers
            # [batch, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            cn = torch.cat([cn[-2], cn[-1]], dim=1).unsqueeze(0)
            # take hx_dense to [1, batch, hidden_size]
            cn = self.hx_dense(cn)
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                cn = torch.cat([cn, cn.new_zeros(self.decoder_layers - 1, batch, 2 * hidden_size)], dim=0)
            # hn is tanh(cn)
            hn = torch.tanh(cn)
            hn = (hn, cn)
        else:
            # take the last layers
            # [2, batch, hidden_size]
            hn = hn[-2:]
            # hn [2, batch, hidden_size]
            _, batch, hidden_size = hn.size()
            # first convert hn t0 [batch, 2, hidden_size]
            hn = hn.transpose(0, 1).contiguous()
            # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            hn = hn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [1, batch, hidden_size]
            hn = torch.tanh(self.hx_dense(hn))
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                hn = torch.cat([hn, hn.new_zeros(self.decoder_layers - 1, batch, 2 * hidden_size)], dim=0)
        return hn
