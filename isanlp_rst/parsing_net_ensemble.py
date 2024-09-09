import torch
import torch.nn as nn
from isanlp_rst.dmrst_parser.du_converter import DUConverter


class ParsingNetEnsemble(nn.Module):
    def __init__(self, models, weights=None, aggregation_method="average"):
        """
        Initialize the ensemble with a list of trained ParsingNet models.

        :param models: List of ParsingNet models
        """
        super(ParsingNetEnsemble, self).__init__()
        self.models = models  # List of ParsingNet models
        self.aggregation_method = aggregation_method

        if aggregation_method == "weighted_average":
            self._weights = weights

            if not self._weights:
                self._weights = [1.0] * len(models)

    def segment_input(self, input_data, segmenter_model_idx=0):
        """
        Use one of the models to perform segmentation and obtain EDU breaks.

        :param input_data: The input data to be segmented.
        :param segmenter_model_idx: Index of the model to use for segmentation.
        :return: Segmented EDU breaks and embeddings from the encoder.
        """
        model = self.models[segmenter_model_idx]

        # Assuming the input_data has the necessary fields for the EncoderRNN
        tokenized_texts = input_data.input_sentences  # Adjust if necessary to match the input format
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

        :param model_outputs: List of model outputs (logits/probabilities) for the current step
        :return: Aggregated predictions (logits/probabilities)
        """
        if self.aggregation_method == "average":
            # Average the logits/probabilities from all models
            return torch.mean(torch.stack(model_outputs), dim=0)
        elif self.aggregation_method == "vote":
            # Majority voting on discrete predictions (assuming hard decisions)
            votes = torch.stack([torch.argmax(output, dim=-1) for output in model_outputs], dim=0)
            return torch.mode(votes, dim=0)[0]
        elif self.aggregation_method == "weighted_average":
            weighted_sum = sum(weight * output for weight, output in zip(self._weights, model_outputs))
            return weighted_sum / sum(self._weights)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

    def forward(self, input_data, segmenter_model_idx=0, generate_tree=False):
        """
        Perform inference with the ensemble of ParsingNet models. Segmentation is done by a specific model,
        while parsing and labeling are done using the ensemble.

        :param input_data: The input data to be parsed.
        :param segmenter_model_idx: The index of the model used for segmentation.
        :param generate_tree: Whether to generate the RST tree structure.
        :return: A dictionary containing the tokens, spans, EDU breaks, and predicted labels.
        """
        # Step 1: Perform segmentation using one specific model's segmenter
        edu_breaks, encoder_output = self.segment_input(input_data, segmenter_model_idx)
        input_data.edu_breaks = edu_breaks  # Add EDU breaks to the input data for parsing

        predictions = {
            'tokens': [],
            'spans': [],
            'edu_breaks': edu_breaks,
            'labels': [],
        }

        prev_decision = None

        # Step 2: Iterate over the parsing process
        for step in range(len(input_data.edu_breaks)):
            step_outputs_splits = []
            step_outputs_labels = []

            # For each model in the ensemble, get predictions for the current step
            for model in self.models:
                split_output, label_output = model.infer_step(input_data, prev_decision)
                step_outputs_splits.append(split_output)
                step_outputs_labels.append(label_output)

            print(f'{step_outputs_splits = }')
            print(f'{step_outputs_labels = }')

            # Aggregate the predictions from all models at the current step
            aggregated_splits = self.aggregate_predictions(step_outputs_splits)
            aggregated_labels = self.aggregate_predictions(step_outputs_labels)

            print(f'{aggregated_splits = }')
            print(f'{aggregated_labels = }')

            # Make the final decision for this step
            prev_decision = torch.argmax(aggregated_splits, dim=-1)

            # Store the aggregated decisions
            predictions['spans'].append(aggregated_splits)
            predictions['labels'].append(aggregated_labels)

        if generate_tree:
            # Additional step to generate the rhetorical structure tree
            tree = self.generate_rst_tree(predictions)
            return {
                'tokens': self.models[0].tokenizer.convert_ids_to_tokens(input_data['text'][0]),
                # Get tokens from first model
                'rst': tree
            }

        return predictions

    def generate_rst_tree(self, predictions):
        # Convert predictions to a tree structure
        duc = DUConverter(predictions, tokenization_type='default')
        tree = duc.collect()[0]

        return tree
