import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BiMPM(nn.Module):
    def __init__(self, word_dim, hidden_size, class_number, num_perspective=10, dropout=0.4,
                 max_len=1000,
                 with_full_match=False, with_maxpool_match=True,
                 with_attentive_match=True, with_max_attentive_match=True,
                 cuda_device=None, use_amp=False):
        super(BiMPM, self).__init__()

        self.d = word_dim
        self.l = num_perspective
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.with_full_match = with_full_match
        self.with_maxpool_match = with_maxpool_match
        self.with_attentive_match = with_attentive_match
        self.with_max_attentive_match = with_max_attentive_match
        self._cuda_device = cuda_device
        self.use_amp = use_amp

        # ----- Context Representation Layer -----
        self.context_LSTM = nn.LSTM(
            input_size=self.d,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            device=self._cuda_device
        )

        # ----- Matching Layer -----
        for i in range(1, 9):
            setattr(self, f'mp_w{i}', nn.Parameter(torch.rand(self.l, hidden_size)))

        # ----- Aggregation Layer -----
        self.aggregation_LSTM = nn.LSTM(
            input_size=2 + self.l * 2 * sum([int(v) for v in [self.with_full_match, self.with_maxpool_match,
                                                              self.with_attentive_match, self.with_max_attentive_match]]),
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            device=self._cuda_device
        )

        # ----- Prediction Layer -----
        linear_input_size = hidden_size * 4 + 2 + self.d * 2  # perspectives, lengths, averaged embeddings
        self.pred_fc1 = nn.Linear(linear_input_size, hidden_size * 2, device=self._cuda_device)
        self.pred_fc2 = nn.Linear(linear_input_size // 2, class_number, device=self._cuda_device)

        self.reset_parameters()

    def reset_parameters(self):

        # ----- Context Representation Layer -----
        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0)
        nn.init.constant_(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0)
        nn.init.constant_(self.context_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_hh_l0_reverse, val=0)

        # ----- Matching Layer -----
        for i in range(1, 9):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal_(w)

        # ----- Aggregation Layer -----
        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)

        # ----- Prediction Layer ----
        nn.init.uniform_(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc1.bias, val=0)

        nn.init.uniform_(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc2.bias, val=0)

    def mp_matching_func(self, v1, v2, w):
        """
        :param v1: (batch, seq_len, hidden_size)
        :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
        :param w: (l, hidden_size)
        :return: (batch, l)
        """
        seq_len = v1.size(1)

        # (1, 1, hidden_size, l)
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
        # (batch, seq_len, hidden_size, l)
        v1 = w * torch.stack([v1] * self.l, dim=3)
        if len(v2.size()) == 3:
            v2 = w * torch.stack([v2] * self.l, dim=3)
        else:
            v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * self.l, dim=3)

        m = F.cosine_similarity(v1, v2, dim=2)
        return m

    def mp_matching_func_pairwise(self, v1, v2, w):
        """
        :param v1: (batch, seq_len1, hidden_size)
        :param v2: (batch, seq_len2, hidden_size)
        :param w: (l, hidden_size)
        :return: (batch, l, seq_len1, seq_len2)
        """

        # (1, l, 1, hidden_size)
        w = w.unsqueeze(0).unsqueeze(2)
        # (batch, l, seq_len, hidden_size)
        v1, v2 = w * torch.stack([v1] * self.l, dim=1), w * torch.stack([v2] * self.l, dim=1)
        # (batch, l, seq_len, hidden_size->1)
        v1_norm = v1.norm(p=2, dim=3, keepdim=True)
        v2_norm = v2.norm(p=2, dim=3, keepdim=True)

        # (batch, l, seq_len1, seq_len2)
        if self.use_amp:
            n = torch.matmul(v1.half(), v2.half().transpose(2, 3))
            d = v1_norm.half() * v2_norm.half().transpose(2, 3)
        else:
            n = torch.matmul(v1, v2.transpose(2, 3))
            d = v1_norm * v2_norm.transpose(2, 3)

        # (batch, seq_len1, seq_len2, l)
        return self.div_with_small_value(n, d).permute(0, 2, 3, 1)

    def attention(self, v1, v2):
        """
        :param v1: (batch, seq_len1, hidden_size)
        :param v2: (batch, seq_len2, hidden_size)
        :return: (batch, seq_len1, seq_len2)
        """

        # (batch, seq_len1, 1)
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        # (batch, 1, seq_len2)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

        # (batch, seq_len1, seq_len2)
        if self.use_amp:
            a = torch.bmm(v1.half(), v2.half().permute(0, 2, 1))
            d = v1_norm.half() * v2_norm.half()
        else:
            a = torch.bmm(v1, v2.permute(0, 2, 1))
            d = v1_norm * v2_norm

        return self.div_with_small_value(a, d)

    def div_with_small_value(self, n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def _reduce_length(self, emb):
        """ Inputs can be of infinite length, hence BiMPM matching can cause OOM.
            This is a way to limit the input averaging the middle part of an unbearable long sequence.

            :param: emb (torch.FloatTensor)  - all embeddings of [1, seq_length, emb_size]
            Output (torch.FloatTensor)  - embeddings of [1, self.max_len+1, emb_size]
            """
        if self.max_len:
            if emb.size(1) > self.max_len + 1:
                return torch.cat([emb[:, :self.max_len//2, :],
                                  emb[:, self.max_len//2:-self.max_len//2, :].mean(dim=1, keepdim=True),
                                  emb[:, -self.max_len//2:, :]], dim=1)
        return emb

    def encode(self, left, right, len1=None, len2=None):
        left = self._reduce_length(left)
        right = self._reduce_length(right)

        # ----- Context Representation Layer -----
        # (batch, seq_len, hidden_size * 2)
        left, _ = self.context_LSTM(left)
        right, _ = self.context_LSTM(right)

        left = self.dropout(left)
        right = self.dropout(right)

        # If passing token embeddings, for EDU embeddings you should pass token lengths in the function
        if not len1:
            len1 = left.size(1)
        if not len2:
            len2 = right.size(1)

        # (batch, 2)
        lengths = torch.tensor([len1 / (len1 + len2), len2 / (len1 + len2)],
                               dtype=torch.float, device=self._cuda_device).unsqueeze(0)

        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(left, self.hidden_size, dim=-1)
        con_h_fw, con_h_bw = torch.split(right, self.hidden_size, dim=-1)

        # array to keep the matching vectors for the two DUs
        matching_vector_1: List[torch.Tensor] = []
        matching_vector_2: List[torch.Tensor] = []

        # 0. unweighted cosine
        # First calculate the cosine similarities between each forward
        # (or backward) contextual embedding and every forward (or backward)
        # contextual embedding of the other sentence.

        # (batch, seq_len1, seq_len2)
        cosine_sim = F.cosine_similarity(con_p_fw.unsqueeze(-2), con_h_fw.unsqueeze(-3), dim=3)

        # (batch, seq_len*, 1)
        cosine_max_1, _ = cosine_sim.max(dim=2, keepdim=True)
        cosine_mean_1 = cosine_sim.mean(dim=2, keepdim=True)
        cosine_max_2, _ = cosine_sim.permute(0, 2, 1).max(dim=2, keepdim=True)
        cosine_mean_2 = cosine_sim.permute(0, 2, 1).mean(dim=2, keepdim=True)

        matching_vector_1.extend([cosine_max_1, cosine_mean_1])
        matching_vector_2.extend([cosine_max_2, cosine_mean_2])

        # 1. Full-Matching
        # Each time step of forward (or backward) contextual embedding of one DU
        # is compared with the last time step of the forward (or backward)
        # contextual embedding of the other DU
        # (batch, seq_len, hidden_size), (batch, hidden_size)
        # -> (batch, seq_len, l)
        if self.with_full_match:
            mv_p_full_fw = self.mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
            mv_p_full_bw = self.mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
            mv_h_full_fw = self.mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
            mv_h_full_bw = self.mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

            matching_vector_1.extend([mv_p_full_fw, mv_p_full_bw])
            matching_vector_2.extend([mv_h_full_fw, mv_h_full_bw])

        # 2. Maxpooling-Matching
        # Each time step of forward (or backward) contextual embedding of one DU
        # is compared with every time step of the forward (or backward)
        # contextual embedding of the other DU, and only the max value of each
        # dimension is retained.
        if self.with_maxpool_match:
            # (batch, seq_len1, seq_len2, l)
            mv_max_fw = self.mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3)
            mv_max_bw = self.mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4)

            # (batch, seq_len, l)
            mv_p_max_fw, _ = mv_max_fw.max(dim=2)
            mv_p_max_bw, _ = mv_max_bw.max(dim=2)
            mv_h_max_fw, _ = mv_max_fw.max(dim=1)
            mv_h_max_bw, _ = mv_max_bw.max(dim=1)

            matching_vector_1.extend([mv_p_max_fw.float(), mv_p_max_bw.float()])
            matching_vector_2.extend([mv_h_max_fw.float(), mv_h_max_bw.float()])

        # 3. Attentive-Matching
        # Each forward (or backward) similarity is taken as the weight
        # of the forward (or backward) contextual embedding, and calculate an
        # attentive vector for the sentence by weighted summing all its
        # contextual embeddings.
        # Finally, match each forward (or backward) contextual embedding
        # with its corresponding attentive vector.

        # (batch, seq_len1, seq_len2)
        att_fw = self.attention(con_p_fw, con_h_fw)
        att_bw = self.attention(con_p_bw, con_h_bw)

        if self.with_attentive_match:
            # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
            # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
            # -> (batch, seq_len1, seq_len2, hidden_size)
            att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
            att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
            # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
            # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
            # -> (batch, seq_len1, seq_len2, hidden_size)
            att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
            att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)

            # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
            att_mean_h_fw = self.div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
            att_mean_h_bw = self.div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

            # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
            att_mean_p_fw = self.div_with_small_value(att_p_fw.sum(dim=1),
                                                      att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
            att_mean_p_bw = self.div_with_small_value(att_p_bw.sum(dim=1),
                                                      att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

            # (batch, seq_len, l)
            mv_p_att_mean_fw = self.mp_matching_func(con_p_fw, att_mean_h_fw, self.mp_w5)
            mv_p_att_mean_bw = self.mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
            mv_h_att_mean_fw = self.mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
            mv_h_att_mean_bw = self.mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)

            matching_vector_1.extend([mv_p_att_mean_fw, mv_p_att_mean_bw])
            matching_vector_2.extend([mv_h_att_mean_fw, mv_h_att_mean_bw])

        # 4. Max-Attentive-Matching
        # Pick the contextual embeddings with the highest cosine similarity as the attentive
        # vector, and match each forward (or backward) contextual embedding with its
        # corresponding attentive vector.

        if self.with_max_attentive_match:
            # (batch, seq_len1, hidden_size)
            att_max_h_fw, _ = att_h_fw.max(dim=2)
            att_max_h_bw, _ = att_h_bw.max(dim=2)
            # (batch, seq_len2, hidden_size)
            att_max_p_fw, _ = att_p_fw.max(dim=1)
            att_max_p_bw, _ = att_p_bw.max(dim=1)

            # (batch, seq_len, l)
            mv_p_att_max_fw = self.mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
            mv_p_att_max_bw = self.mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
            mv_h_att_max_fw = self.mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
            mv_h_att_max_bw = self.mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)

            matching_vector_1.extend([mv_p_att_max_fw, mv_p_att_max_bw])
            matching_vector_2.extend([mv_h_att_max_fw, mv_h_att_max_bw])

        # (batch, seq_len, l * 8 + 2)
        mv_p = torch.cat(matching_vector_1, dim=2)
        mv_h = torch.cat(matching_vector_2, dim=2)

        # print(f'{mv_p.shape = } {mv_h.shape = }')

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # ----- Aggregation Layer -----
        # (batch, seq_len, l * 8 + 2) -> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)

        return agg_p_last, agg_h_last, lengths

    def forward(self, left, right, len1=None, len2=None, **kwargs):
        agg_p_last, agg_h_last, lengths = self.encode(left, right, len1, len2)

        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)], dim=1)
        x = self.dropout(x)

        # (batch, hidden_size * 4 + 2)
        x = torch.cat([x, lengths], dim=1)

        # ----- Prediction Layer -----
        # (batch, hidden_size * 2)
        x = torch.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)

        # Obtain relation weights and log relation weights (for loss)
        relation_weights = F.softmax(x, 1)
        log_relation_weights = F.log_softmax(x + 1e-6, 1)

        return relation_weights, log_relation_weights
