"""
BiMPM (Bilateral Multi-Perspective Matching) model implementation.
"""

from typing import Dict, List

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.bimpm_matching import BiMpmMatching
from allennlp.nn import InitializerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


class BiMpm(Model):
    """
    This ``Model`` augments with additional features the BiMPM model described in `Bilateral Multi-Perspective
    Matching for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017.
    implemented in https://github.com/galsang/BIMPM-pytorch>`_.
    Additional features are added before the feedforward classifier.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 matcher_word: BiMpmMatching,
                 encoder1: Seq2SeqEncoder,
                 matcher_forward1: BiMpmMatching,
                 matcher_backward1: BiMpmMatching,
                 encoder2: Seq2SeqEncoder,
                 matcher_forward2: BiMpmMatching,
                 matcher_backward2: BiMpmMatching,
                 aggregator: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 encode_together: bool = False,
                 encode_lstm: bool = True,
                 dropout: float = 0.1,
                 class_weights: list = [],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs) -> None:
        super().__init__(vocab, **kwargs)

        self.text_field_embedder = text_field_embedder

        self.matcher_word = matcher_word

        self.encoder1 = encoder1
        self.matcher_forward1 = matcher_forward1
        self.matcher_backward1 = matcher_backward1

        self.encoder2 = encoder2
        self.matcher_forward2 = matcher_forward2
        self.matcher_backward2 = matcher_backward2

        self.aggregator = aggregator

        self.encode_together = encode_together
        self.encode_lstm = encode_lstm

        matching_dim = self.matcher_word.get_output_dim()

        if self.encode_lstm:
            matching_dim += self.matcher_forward1.get_output_dim(
            ) + self.matcher_backward1.get_output_dim(
            ) + self.matcher_forward2.get_output_dim(
            ) + self.matcher_backward2.get_output_dim(
            )

        check_dimensions_match(matching_dim, self.aggregator.get_input_dim(),
                               "sum of dim of all matching layers", "aggregator input dim")

        self.classifier_feedforward = classifier_feedforward

        self.dropout = torch.nn.Dropout(dropout)

        if class_weights:
            self.class_weights = class_weights
        else:
            self.class_weights = [1.] * self.classifier_feedforward.get_output_dim()

        self.metrics = {"accuracy": CategoricalAccuracy(),
                        "f1": F1Measure(1)}

        self.loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(self.class_weights))

        initializer(self)

    def forward(self,
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                same_sentence: List[Dict[str, torch.IntTensor]],
                same_paragraph: List[Dict[str, torch.IntTensor]],
                label: torch.LongTensor = None,  # pylint:disable=unused-argument
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            The premise from a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            The hypothesis from a ``TextField``
        label : torch.LongTensor, optional (default = None)
            The label for the pair of the premise and the hypothesis
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Additional information about the pair
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        def encode_pair(x1, x2, mask1=None, mask2=None):
            _joined_pair: Dict[str, torch.LongTensor] = {}

            for key in premise.keys():
                bsz = premise[key].size(0)
                x1_len, x2_len = premise[key].size(1), hypothesis[key].size(1)
                sep = torch.empty([bsz, 1], dtype=torch.long, device=premise[key].device)
                sep.data.fill_(0)  # 2 is the id for </s>

                x = torch.cat([premise[key], hypothesis[key]], dim=1)
                _joined_pair[key] = x

            x_output = self.dropout(self.text_field_embedder(_joined_pair))
            return x_output[:, :x1_len], x_output[:, -x2_len:], mask1, mask2

        mask_premise = util.get_text_field_mask(premise)
        mask_hypothesis = util.get_text_field_mask(hypothesis)

        if self.encode_together:
            embedded_premise, embedded_hypothesis, _, _ = encode_pair(premise, hypothesis)
        else:
            embedded_premise = self.dropout(self.text_field_embedder(premise))
            embedded_hypothesis = self.dropout(self.text_field_embedder(hypothesis))

        # embedding and encoding of the premise
        encoded_premise1 = self.dropout(self.encoder1(embedded_premise, mask_premise))
        encoded_premise2 = self.dropout(self.encoder2(encoded_premise1, mask_premise))

        # embedding and encoding of the hypothesis
        encoded_hypothesis1 = self.dropout(self.encoder1(embedded_hypothesis, mask_hypothesis))
        encoded_hypothesis2 = self.dropout(self.encoder2(encoded_hypothesis1, mask_hypothesis))

        matching_vector_premise: List[torch.Tensor] = []
        matching_vector_hypothesis: List[torch.Tensor] = []

        def add_matching_result(matcher, encoded_premise, encoded_hypothesis):
            # utility function to get matching result and add to the result list
            matching_result = matcher(encoded_premise, mask_premise, encoded_hypothesis, mask_hypothesis)
            matching_vector_premise.extend(matching_result[0])
            matching_vector_hypothesis.extend(matching_result[1])

        # calculate matching vectors from word embedding, first layer encoding, and second layer encoding
        add_matching_result(self.matcher_word, embedded_premise, embedded_hypothesis)
        half_hidden_size_1 = self.encoder1.get_output_dim() // 2
        add_matching_result(self.matcher_forward1,
                            encoded_premise1[:, :, :half_hidden_size_1],
                            encoded_hypothesis1[:, :, :half_hidden_size_1])
        add_matching_result(self.matcher_backward1,
                            encoded_premise1[:, :, half_hidden_size_1:],
                            encoded_hypothesis1[:, :, half_hidden_size_1:])

        half_hidden_size_2 = self.encoder2.get_output_dim() // 2
        add_matching_result(self.matcher_forward2,
                            encoded_premise2[:, :, :half_hidden_size_2],
                            encoded_hypothesis2[:, :, :half_hidden_size_2])
        add_matching_result(self.matcher_backward2,
                            encoded_premise2[:, :, half_hidden_size_2:],
                            encoded_hypothesis2[:, :, half_hidden_size_2:])

        # concat the matching vectors
        matching_vector_cat_premise = self.dropout(torch.cat(matching_vector_premise, dim=2))
        matching_vector_cat_hypothesis = self.dropout(torch.cat(matching_vector_hypothesis, dim=2))

        # aggregate the matching vectors
        aggregated_premise = self.dropout(self.aggregator(matching_vector_cat_premise, mask_premise))
        aggregated_hypothesis = self.dropout(self.aggregator(matching_vector_cat_hypothesis, mask_hypothesis))

        # encode additional information
        batch_size, _ = aggregated_premise.size()
        encoded_same_sentence = same_sentence.float().view(batch_size, -1)
        encoded_same_paragraph = same_paragraph.float().view(batch_size, -1)

        # the final forward layer
        logits = self.classifier_feedforward(
            torch.cat([aggregated_premise,
                       aggregated_hypothesis,
                       encoded_same_sentence,
                       encoded_same_paragraph], dim=-1))

        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {'logits': logits, "probs": probs}

        if label is not None:
            loss = self.loss(logits, label)
            output_dict["loss"] = loss

            for metric in self.metrics.values():
                metric(logits, label)

        return output_dict

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary("labels").get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "f1": self.metrics["f1"].get_metric(reset)['f1'],
            "accuracy": self.metrics["accuracy"].get_metric(reset)
        }

    default_predictor = 'custom_bimpm_predictor'
