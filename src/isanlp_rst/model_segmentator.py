import os

import numpy as np
from tensorflow.python.keras.models import load_model

from isanlp.annotation_rst import DiscourseUnit
from features_processor_segmentation import FeaturesProcessorSegmentation
import tensorflow as tf


global graph
graph = tf.get_default_graph()


class ModelSegmentator:
    def __init__(self, model_dir_path):
        self._features_processor = FeaturesProcessorSegmentation(model_dir_path)
        self._model = load_model(os.path.join(model_dir_path, 'segmentation', 'neural_model.h5'))

    def __call__(self, *args, **kwargs):
        """
        :param args: 'text', 'tokens', 'sentences', 'postag', 'lemma' values of an isanlp annotation
        :return: list of DiscourseUnit
        """

        features = self._features_processor(*args)
        return self._build_discourse_units(args[0], args[1], self._predict(features))

    def _predict(self, features):
        """
        :param list features: features to feed directly into the model
        :return: numbers of tokens predicted as EDU right boundaries
        """
        with graph.as_default():
            predictions = self._model.predict(features)
        return [addr[0] for addr in np.argwhere(np.argmax(predictions, axis=1))]

    def _build_discourse_units(self, text, tokens, numbers):
        """
        :param text: original text
        :param list tokens: isanlp.annotation.Token
        :param numbers: positions of tokens predicted as EDU right boundaries
        :return: list of DiscourseUnit
        """
        edus = []

        new_edu = DiscourseUnit(0,
                                start=0,
                                end=tokens[numbers[0]].end,
                                text=text[:tokens[numbers[0]].end],
                                relation='elementary')
        edus.append(new_edu)

        for i in range(1, len(numbers)):
            new_edu = DiscourseUnit(i,
                                    start=tokens[numbers[i-1]].end,
                                    end=tokens[numbers[i]].end,
                                    text=text[tokens[numbers[i-1]].end:tokens[numbers[i]].end],
                                    relation='elementary')
            edus.append(new_edu)

        return edus
