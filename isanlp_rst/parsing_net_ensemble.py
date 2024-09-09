import torch
import torch.nn as nn
from isanlp_rst.dmrst_parser.src.parser.data import nucs_and_rels


class ParsingNetEnsemble(nn.Module):
    def __init__(self, models, weights=None, aggregation_method="average"):
        """
        Initialize the ensemble with a list of trained ParsingNet models.

        :param models: List of ParsingNet models.
        :param weights: Optional weights for models if using weighted average aggregation.
        :param aggregation_method: Method to aggregate predictions. Options are "average", "weighted_average", "vote".
        """
        super(ParsingNetEnsemble, self).__init__()
        self.models = models  # List of ParsingNet models
        self.aggregation_method = aggregation_method

        if aggregation_method == "weighted_average":
            self._weights = weights if weights else [1.0] * len(models)

    def segment_input(self, input_data, segmenter_model_idx=0):
        """
        Use one of the models to perform segmentation and obtain EDU breaks.

        :param input_data: The input data to be segmented.
        :param segmenter_model_idx: Index of the model to use for segmentation.
        :return: Segmented EDU breaks and embeddings from the encoder.
        """
        model = self.models[segmenter_model_idx]

        # Assuming the input_data has the necessary fields for the EncoderRNN
        tokenized_texts = input_data.input_sentences
        entity_ids = input_data.entity_ids
        entity_position_ids = input_data.entity_position_ids
        edu_breaks = input_data.edu_breaks
        sent_breaks = input_data.sent_breaks

        # Perform segmentation using the encoder of the chosen model
        with torch.no_grad():
            encoder_output, hidden_state, edu_loss, predicted_edu_breaks = model.encoder(
                tokenized_texts, entity_ids, entity_position_ids, edu_breaks, sent_breaks, is_test=True
            )

        return predicted_edu_breaks, encoder_output

    def aggregate_predictions(self, model_outputs):
        """
        Aggregate the predictions of the models at each step.

        :param model_outputs: List of model outputs (logits/probabilities) for the current step.
        :return: Aggregated predictions (logits/probabilities).
        """
        if self.aggregation_method == "average":
            return torch.mean(torch.stack(model_outputs), dim=0)
        elif self.aggregation_method == "vote":
            votes = torch.stack([torch.argmax(output, dim=-1) for output in model_outputs], dim=0)
            return torch.mode(votes, dim=0)[0]
        elif self.aggregation_method == "weighted_average":
            weighted_sum = sum(weight * output for weight, output in zip(self._weights, model_outputs))
            return weighted_sum / sum(self._weights)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

    def infer_step(self, input_data, prev_decision, classifier_model_idx=0):
        """
        Perform a single inference step for the ensemble. The classification for the splitting point
        is handled by the model specified by `classifier_model_idx`.

        :param input_data: Input data containing sentence tokens, entity ids, EDU breaks, etc.
        :param prev_decision: The decision (split point) made in the previous step of the parsing process.
        :param classifier_model_idx: Index of the model used for classification at the split point.
        :return: Tuple containing the predicted split point (logits) and relation label (logits) for the current step.
        """
        step_outputs_splits = []
        # Aggregate predictions from all models for the split point
        for model in self.models:
            split_output, _ = model.infer_step(input_data, prev_decision)
            step_outputs_splits.append(split_output)

        # Aggregate splits using the ensemble strategy (average, weighted average, or vote)
        aggregated_splits = self.aggregate_predictions(step_outputs_splits)

        # Get the classification logits from the specified model
        _, log_relation_weights = self.models[classifier_model_idx].infer_step(input_data, prev_decision)

        return aggregated_splits, log_relation_weights

    def testing_loss(self, input_data, generate_tree, classifier_model_idx=0,
                     use_pred_segmentation=True):
        """
        Calculate the testing loss by making predictions using the ensemble at each step.

        :param input_sentence: Input sentences for testing.
        :param input_entity_ids: Entity IDs for tokens.
        :param input_entity_position_ids: Position of entities.
        :param input_edu_breaks: EDU breaks to use.
        :param label_index: Ground truth labels for each relation.
        :param parsing_index: Ground truth parsing splits.
        :param generate_tree: Whether to generate an RST tree structure.
        :param classifier_model_idx: A list specifying which model to use for classification at each step.
        :param use_pred_segmentation: Whether to use predicted EDU breaks or ground truth.
        :return: Loss values for tree structure and relation classification.
        """

        input_sentence = input_data.input_sentences
        input_entity_ids = input_data.entity_ids
        input_entity_position_ids = input_data.entity_position_ids
        input_edu_breaks = input_data.edu_breaks
        label_index = input_data.relation_label
        parsing_index = input_data.parsing_breaks

        # Step 1: Get encoder outputs and hidden states
        encoder_outputs_list = []
        for model in self.models:
            if model.du_encoding_kind == 'bert':
                encoder_outputs, last_hidden_states, _, predict_edu_breaks, embeddings = model.encoder(
                    input_sentence, input_entity_ids, input_entity_position_ids, input_edu_breaks,
                    is_test=use_pred_segmentation)
            else:
                encoder_outputs, last_hidden_states, _, predict_edu_breaks = model.encoder(
                    input_sentence, input_entity_ids, input_entity_position_ids, input_edu_breaks,
                    is_test=use_pred_segmentation)
            encoder_outputs_list.append(encoder_outputs)

        if use_pred_segmentation:
            edu_breaks = predict_edu_breaks
            if label_index is None and parsing_index is None:
                label_index = [[0, ] * (len(i) - 1) for i in edu_breaks]
                parsing_index = [[0, ] * (len(i) - 1) for i in edu_breaks]
        else:
            edu_breaks = input_edu_breaks

        label_loss_function = nn.NLLLoss()
        span_loss_function = nn.NLLLoss()

        loss_label_batch = torch.FloatTensor([0.0]).to(self.models[0]._cuda_device)
        loss_tree_batch = torch.FloatTensor([0.0]).to(self.models[0]._cuda_device)
        loop_label_batch = 0
        loop_tree_batch = 0

        label_batch = []
        tree_batch = []

        if generate_tree:
            span_batch = []

        # Step 2: Iterate through each EDU break to make predictions
        for i in range(len(edu_breaks)):
            cur_label = []
            cur_tree = []

            cur_label_index = torch.tensor(label_index[i]).to(self.models[0]._cuda_device)
            cur_parsing_index = parsing_index[i]

            stack = [list(range(len(edu_breaks[i])))]

            while stack:
                stack_head = stack.pop()

                if len(stack_head) == 1:
                    continue  # No split needed for a single EDU

                # Step 3: Perform inference using the ensemble for this step
                prev_decision = None if not cur_tree else cur_tree[-1]

                # Get ensembled split and classification predictions
                aggregated_splits, log_relation_weights = self.infer_step(input_data, prev_decision,
                                                                          classifier_model_idx)

                predicted_split_idx = torch.argmax(aggregated_splits, dim=-1).item()
                predicted_label_idx = torch.argmax(log_relation_weights, dim=-1).item()

                cur_label.append(predicted_label_idx)
                cur_tree.append(predicted_split_idx)

                # Spanning logic for RST generation
                if generate_tree:
                    nuclearity_left, nuclearity_right, relation_left, relation_right = \
                        nucs_and_rels(predicted_label_idx,
                                      self.models[0].relation_table)  # Use the relation table of the first model

                    cur_span = f"({stack_head[0] + 1}:{nuclearity_left}={relation_left}:{stack_head[0] + 1}," \
                               f"{stack_head[-1] + 1}:{nuclearity_right}={relation_right}:{stack_head[-1] + 1}) "

                if len(stack_head[:predicted_split_idx + 1]) > 1:
                    stack.append(stack_head[:predicted_split_idx + 1])
                if len(stack_head[predicted_split_idx + 1:]) > 1:
                    stack.append(stack_head[predicted_split_idx + 1:])

            label_batch.append(cur_label)
            tree_batch.append(cur_tree)
            merged_label_gold.extend(cur_label_index.cpu().tolist())
            merged_label_pred.extend(cur_label)

            if generate_tree:
                span_batch.append(cur_span.strip())

        loss_label_batch /= loop_label_batch
        loss_tree_batch /= max(loop_tree_batch, 1)

        return loss_tree_batch, loss_label_batch, (span_batch if generate_tree else None), (
            merged_label_gold, merged_label_pred), edu_breaks

    def forward(self, data, generate_tree=True, classifier_model_idx=0):
        return self.testing_loss(data, generate_tree, classifier_model_idx, use_pred_segmentation=True)

