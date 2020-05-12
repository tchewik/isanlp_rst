
# encode simultaneously

"""
BiMPM (Bilateral Multi-Perspective Matching) model implementation.
"""

from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
import numpy

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from allennlp.modules.bimpm_matching import BiMpmMatching

from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_final_encoder_states
import torch.nn.functional as F


@Model.register("contextual_bimpm_cnn")
class ContextualBiMpm(Model):
    """
    This ``Model`` augments with additional features the BiMPM model described in `Bilateral Multi-Perspective 
    Matching for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017.
    implemented in https://github.com/galsang/BIMPM-pytorch>`_.
    Additional features are added before the feedforward classifier.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder_context: Seq2SeqEncoder,
                 matcher_word: BiMpmMatching,
                 encoder1: Seq2SeqEncoder,
                 matcher_forward1: BiMpmMatching,
                 matcher_backward1: BiMpmMatching,
                 encoder2: Seq2SeqEncoder,
                 matcher_forward2: BiMpmMatching,
                 matcher_backward2: BiMpmMatching,
                 aggregator: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(ContextualBiMpm, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder

        self.encoder_context = encoder_context
        
        self.matcher_word = matcher_word

        self.encoder1 = encoder1
        self.matcher_forward1 = matcher_forward1
        self.matcher_backward1 = matcher_backward1

        self.encoder2 = encoder2
        self.matcher_forward2 = matcher_forward2
        self.matcher_backward2 = matcher_backward2

        self.aggregator = aggregator

        matching_dim = self.matcher_word.get_output_dim() + \
                       self.matcher_forward1.get_output_dim() + self.matcher_backward1.get_output_dim() + \
                       self.matcher_forward2.get_output_dim() + self.matcher_backward2.get_output_dim()
                        
        #matching_dim *= 3  # contextual matches on both sides
        check_dimensions_match(matching_dim, self.aggregator.get_input_dim(),
                               "sum of dim of all matching layers", "aggregator input dim")
        
        self.classifier_feedforward = classifier_feedforward

        self.dropout = torch.nn.Dropout(dropout)

        self.metrics = {"accuracy": CategoricalAccuracy(),
                        "f1": F1Measure(1)}

        self.loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1]))

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                left_context: Dict[str, torch.LongTensor],
                right_context: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, torch.FloatTensor]],
                label: torch.LongTensor=None,# pylint:disable=unused-argument
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

        mask_premise = util.get_text_field_mask(premise)
        mask_hypothesis = util.get_text_field_mask(hypothesis)
        mask_left_context = util.get_text_field_mask(left_context)
        mask_right_context = util.get_text_field_mask(right_context)
        
        _joint_input: Dict = {}
        _joint_str = ""
        _joint_tensor: torch.LongTensor()
        print('>>', left_context)
        for _input in (left_context, premise, hypothesis, right_context):
            _joint_str += _input.key()
        
        _joint_embedding = self.dropout(self.text_field_embedder(_joint_input))
        _left_context, _premise, _hypothesis, _right_context = (_joint_embedding[:len(left_context)],
            _joint_embedding[len(left_context):len(left_context)+len(premise)],
            _joint_embedding[len(left_context)+len(premise):len(left_context)+len(premise)+len(hypothesis)],
            _joint_embedding[len(left_context)+len(premise)+len(hypothesis):len(left_context)+len(premise)+len(hypothesis)+len(right_context)],
                                                               )

        # embedding and encoding of the premise
        #embedded_premise = self.dropout(self.text_field_embedder(premise))
        encoded_premise1 = self.dropout(self.encoder1(_premise, mask_premise))
        encoded_premise2 = self.dropout(self.encoder2(encoded_premise1, mask_premise))

        # embedding and encoding of the hypothesis
        #embedded_hypothesis = self.dropout(self.text_field_embedder(hypothesis))
        encoded_hypothesis1 = self.dropout(self.encoder1(_hypothesis, mask_hypothesis))
        encoded_hypothesis2 = self.dropout(self.encoder2(encoded_hypothesis1, mask_hypothesis))
        
        # embedding and encoding of the contexts
        #embedded_left_context = self.dropout(self.text_field_embedder(left_context))
        encoded_left_context = self.dropout(
            self.encoder_context(_left_context, mask_left_context))

        #embedded_right_context = self.dropout(self.text_field_embedder(right_context))
        encoded_right_context = self.dropout(
            self.encoder_context(_right_context, mask_right_context))
        
        matching_vector_premise: List[torch.Tensor] = []
        matching_vector_hypothesis: List[torch.Tensor] = []
        #matching_vector_left_context: List[torch.Tensor] = []
        #matching_vector_right_context: List[torch.Tensor] = []

        def add_matching_result(matcher, encoded_premise, encoded_hypothesis):
            # utility function to get matching result and add to the result list
            matching_result = matcher(encoded_premise, mask_premise, encoded_hypothesis, mask_hypothesis)
            matching_vector_premise.extend(matching_result[0])
            matching_vector_hypothesis.extend(matching_result[1])
            # add matching results to left and right contexts
#             matching_result = matcher(encoded_left_context, mask_left_context, encoded_premise, mask_premise)
#             matching_vector_premise.extend(matching_result[1])
#             matching_result = matcher(encoded_left_context, mask_left_context, encoded_hypothesis, mask_hypothesis)
#             matching_vector_hypothesis.extend(matching_result[1])
#             matching_result = matcher(encoded_premise, mask_premise, encoded_right_context, mask_right_context)
#             matching_vector_premise.extend(matching_result[0])
#             matching_result = matcher(encoded_hypothesis, mask_hypothesis, encoded_right_context, mask_right_context)
#             matching_vector_hypothesis.extend(matching_result[0])
#             #matching_vector_left_context.extend(matching_result[0])
#             #matching_result = matcher(encoded_hypothesis, mask_hypothesis, encoded_right_context, mask_right_context)
#             #matching_vector_right_context.extend(matching_result[1])

        # calculate matching vectors from word embedding, first layer encoding, and second layer encoding
        add_matching_result(self.matcher_word, 
                            embedded_premise, embedded_hypothesis)
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
        #matching_vector_cat_left_context = self.dropout(torch.cat(matching_vector_left_context, dim=2))
        #matching_vector_cat_right_context = self.dropout(torch.cat(matching_vector_right_context, dim=2))

        # aggregate the matching vectors
        aggregated_premise = self.dropout(self.aggregator(matching_vector_cat_premise, mask_premise))
        aggregated_hypothesis = self.dropout(self.aggregator(matching_vector_cat_hypothesis, mask_hypothesis))
        #aggregated_left_context = self.dropout(self.aggregator(matching_vector_cat_left_context, mask_left_context))
        #aggregated_right_context = self.dropout(self.aggregator(matching_vector_cat_right_context, mask_right_context))

        # encode additional information
        batch_size, _ = aggregated_premise.size()
        encoded_meta = self.dropout(metadata.float().view(batch_size, -1))
        encoded_left_context = encoded_left_context.view(batch_size, -1)
        encoded_right_context = encoded_right_context.view(batch_size, -1)
        
        # the final forward layer
        logits = self.classifier_feedforward(torch.cat([aggregated_premise, aggregated_hypothesis, 
                                                        encoded_meta, encoded_left_context, encoded_right_context], dim=-1))
#                                                         aggregated_left_context, aggregated_right_context], dim=-1))
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {'logits': logits, "probs": probs}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts indices to string labels, and adds a ``"label"`` key to the result.
        """
        predictions = output_dict["probs"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            # f1 get_metric returns (precision, recall, f1)
            "f1": self.metrics["f1"].get_metric(reset=reset)[2],
            #"precision": self.metrics["f1"].get_metric(reset=reset)[0],
            #"recall": self.metrics["f1"].get_metric(reset=reset)[1],
            "accuracy": self.metrics["accuracy"].get_metric(reset=reset)
        }
        #return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
