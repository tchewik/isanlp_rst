import torch
import torch.nn as nn
import torch.nn.functional as F

from . import modules


class PointerSegmenter(nn.Module):

    def __init__(self, hidden_size, atten_model=None, decoder_input_size=None, rnn_layers=None, dropout_d=None,
                 if_edu_start_loss=True, cuda_device=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.pointer = modules.PointerAtten(atten_model, hidden_size)
        self.encoder = nn.GRU(hidden_size, int(hidden_size / 2), num_layers=1, batch_first=True, dropout=0.2,
                              bidirectional=True)
        self.decoder = modules.DecoderRNN(decoder_input_size, hidden_size, rnn_layers, dropout_d)
        self.loss_fn = nn.NLLLoss()
        self.if_edu_start_loss = if_edu_start_loss
        self._cuda_device = cuda_device

    def forward(self):
        raise RuntimeError('Segmenter does not have forward process.')

    def train_segment_loss(self, word_embeddings, edu_breaks):
        outputs, last_hidden = self.encoder(word_embeddings.unsqueeze(0))
        outputs = outputs.squeeze()
        cur_decoder_hidden = outputs[-1, :].unsqueeze(0).unsqueeze(0)
        edu_breaks = [0] + edu_breaks
        total_loss = torch.FloatTensor([0.0]).to(self._cuda_device)
        for step, start_index in enumerate(edu_breaks[:-1]):
            cur_decoder_output, cur_decoder_hidden = self.decoder(outputs[start_index].unsqueeze(0).unsqueeze(0),
                                                                  last_hidden=cur_decoder_hidden)

            _, log_atten_weights = self.pointer(outputs[start_index:], cur_decoder_output.squeeze(0).squeeze(0))
            cur_ground_index = torch.tensor([edu_breaks[step + 1] - start_index]).to(self._cuda_device)
            total_loss = total_loss + self.loss_fn(log_atten_weights, cur_ground_index)

        return total_loss

    def test_segment_loss(self, word_embeddings):
        outputs, last_hidden = self.encoder(word_embeddings.unsqueeze(0))
        outputs = outputs.squeeze()
        cur_decoder_hidden = outputs[-1, :].unsqueeze(0).unsqueeze(0)
        start_index = 0
        predict_segment = []
        sentence_length = outputs.shape[0]
        while start_index < sentence_length:
            cur_decoder_output, cur_decoder_hidden = self.decoder(outputs[start_index].unsqueeze(0).unsqueeze(0),
                                                                  last_hidden=cur_decoder_hidden)
            atten_weights, log_atten_weights = self.pointer(outputs[start_index:],
                                                            cur_decoder_output.squeeze(0).squeeze(0))
            _, top_index_seg = atten_weights.topk(1)

            seg_index = int(top_index_seg[0][0]) + start_index
            predict_segment.append(seg_index)
            start_index = seg_index + 1

        if predict_segment[-1] != sentence_length - 1:
            predict_segment.append(sentence_length - 1)

        return predict_segment


class LinearSegmenter(nn.Module):
    def __init__(self, hidden_size, use_sentence_boundaries=False, if_edu_start_loss=True, cuda_device=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_sentence_boundaries = use_sentence_boundaries
        self.if_edu_start_loss = if_edu_start_loss
        self._cuda_device = cuda_device

        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size, 2, device=self._cuda_device)
        self.linear_start = nn.Linear(hidden_size, 2, device=self._cuda_device)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 10.0]).to(self._cuda_device))
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.linear.weight, -0.005, 0.005)
        nn.init.constant_(self.linear.bias, val=0)
        nn.init.uniform_(self.linear_start.weight, -0.005, 0.005)
        nn.init.constant_(self.linear_start.bias, val=0)

    def forward(self):
        raise RuntimeError('Segmenter does not have forward process.')

    def train_segment_loss(self, word_embeddings, edu_breaks, sent_breaks=None):
        edu_break_target = torch.zeros(word_embeddings.size(0), dtype=torch.long, device=self._cuda_device)
        edu_start_target = torch.zeros(word_embeddings.size(0), dtype=torch.long, device=self._cuda_device)

        for i in edu_breaks:
            edu_break_target[i] = 1

        edu_start_target[0] = 1
        for i in edu_breaks[:-1]:
            edu_start_target[i + 1] = 1

        outputs = self.linear(self.dropout(word_embeddings)).to(self._cuda_device)
        start_outputs = self.linear_start(self.dropout(word_embeddings)).to(self._cuda_device)

        if self.if_edu_start_loss:
            first_loss = self.loss_fn(outputs, edu_break_target)
            second_loss = self.loss_fn(start_outputs, edu_start_target)
            total_loss = first_loss + second_loss
        else:
            total_loss = self.loss_fn(outputs, edu_break_target)
        return total_loss

    def test_segment_loss(self, word_embeddings, sent_breaks=None):
        outputs = self.linear(self.dropout(word_embeddings))
        if self.use_sentence_boundaries:
            for i in sent_breaks:
                outputs[i][0] = 0.
                outputs[i][1] = 1.

        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
        predict_segment = [i for i, k in enumerate(pred) if k == 1]

        if word_embeddings.size(0) - 1 not in predict_segment:
            predict_segment.append(word_embeddings.size(0) - 1)

        return predict_segment


class CRF(nn.Module):
    """Conditional random field.

    modified from https://github.com/kmkurn/pytorch-crf/blob/master/torchcrf/__init__.py

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.

    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags, batch_first=True):
        super().__init__()

        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.Tensor(num_tags))
        self.end_transitions = nn.Parameter(torch.Tensor(num_tags))
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.start_transitions, -1., 1.)
        torch.nn.init.uniform_(self.end_transitions, -1., 1.)
        torch.nn.init.uniform_(self.transitions, -1., 1.)

    def __repr__(self):
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions, tags, mask=None):
        """Compute the conditional negative log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The negative log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        return -llh.mean()

    def decode(self, emissions, mask=None):
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = list()

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = list()

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(0)

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on

            best_tags = list()
            for hist in range(seq_length - seq_ends[idx]):
                best_tags.append(torch.zeros_like(best_last_tag))

            best_tags.append(best_last_tag)

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag)

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags = torch.stack(best_tags, dim=0)
            best_tags_list.append(best_tags)

        best_tags_list = torch.stack(best_tags_list, dim=0)

        return best_tags_list


class ToNySegmenter(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 use_sentence_boundaries: bool = False,
                 dropout: float = 0.2,
                 use_lstm: bool = True,
                 hidden_dim: int = 100,
                 num_layers: int = 1,
                 lstm_dropout: float = 0.4,
                 bidirectional: bool = True,
                 use_crf: bool = False,
                 use_log_crf: bool = True,
                 scale_crf: bool = False,
                 if_edu_start_loss: bool = False,
                 cuda_device: torch.device = None):

        super().__init__()

        self.use_sentence_boundaries = use_sentence_boundaries
        self.dropout = nn.Dropout(p=dropout)
        self.use_lstm = use_lstm
        self.num_layers = num_layers
        self.lstm_dropout = lstm_dropout
        self.bidirectional = bidirectional
        self.use_crf = use_crf
        self.use_log_crf = use_log_crf
        self.scale_crf = scale_crf
        self.if_edu_start_loss = if_edu_start_loss
        self._cuda_device = cuda_device

        if self.use_lstm:
            self.lstm = nn.LSTM(embedding_dim,
                                hidden_dim,
                                num_layers=self.num_layers,
                                dropout=self.lstm_dropout,
                                bidirectional=self.bidirectional,
                                device=self._cuda_device)
            self._init_weights(self.lstm)

        linear_input_size = hidden_dim if self.use_lstm else embedding_dim
        if self.use_lstm and self.bidirectional:
            linear_input_size *= 2

        self.hidden2tag = nn.Linear(linear_input_size, 2, device=self._cuda_device)
        if self.if_edu_start_loss:
            self.linear_start = nn.Linear(linear_input_size, 2, device=self._cuda_device)

        if self.use_crf:
            self.crf = CRF(2, batch_first=True)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 10.0]).to(self._cuda_device))

    @staticmethod
    def _init_weights(layer):
        if type(layer) == nn.LSTM:
            for name, param in layer.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, encodings, sent_breaks=None):
        logits = self.hidden2tag(encodings)

        if self.use_sentence_boundaries and sent_breaks is not None:
            logits[sent_breaks][0] = -300.  # inf breaks half-precision computation

        return F.log_softmax(logits, dim=1)

    def encode(self, embeddings):
        if self.use_lstm:
            lstm_out, _ = self.lstm(self.dropout(embeddings))
            return lstm_out.view(len(embeddings), -1)
        else:
            return self.dropout(embeddings)

    def train_segment_loss(self, word_embeddings, edu_breaks, sent_breaks):
        encodings = self.encode(word_embeddings)
        prediction = self(encodings, sent_breaks)

        targets = torch.zeros(word_embeddings.size(0), dtype=torch.long, device=self._cuda_device)
        for i in edu_breaks:
            targets[i] = 1

        if self.use_crf:
            loss = self._crf(prediction, targets)
            if self.use_log_crf:
                loss = torch.clamp(torch.log(loss), max=100)
        else:
            loss = self.loss_fn(prediction, targets)

        if self.if_edu_start_loss:
            start_outputs = self.linear_start(encodings)

            edu_start_targets = torch.zeros(word_embeddings.size(0), dtype=torch.long, device=self._cuda_device)
            edu_start_targets[0] = 1
            for i in edu_breaks[:-1]:
                edu_start_targets[i + 1] = 1

            loss += self.loss_fn(start_outputs, edu_start_targets)

        return loss

    @staticmethod
    def _scaled_sigmoid(x, value):
        return value / (1 + torch.exp(-1 / value * x))

    def _crf(self, prediction, targets):
        seq_length, num_tags = prediction.shape
        emissions = prediction.unsqueeze(0)
        tags = targets.unsqueeze(0)
        mask = torch.ones((1, seq_length), dtype=torch.uint8, device=self._cuda_device)
        return self.crf(emissions, tags, mask)

    def test_segment_loss(self, word_embeddings, sent_breaks=None):
        encodings = self.encode(word_embeddings)
        prediction = self(encodings, sent_breaks)

        if self.use_crf:
            pred = self.crf.decode(prediction.unsqueeze(0))[0]
        else:
            pred = torch.argmax(prediction, dim=1).detach().cpu().tolist()

        predict_segment = [i for i, k in enumerate(pred) if k == 1]

        if word_embeddings.size(0) - 1 not in predict_segment:
            predict_segment.append(word_embeddings.size(0) - 1)

        return predict_segment
