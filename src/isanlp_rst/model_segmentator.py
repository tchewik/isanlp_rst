import os

import numpy as np
import tensorflow as tf
from features_processor_segmentation import FeaturesProcessorSegmentation
from isanlp.annotation_rst import DiscourseUnit
from tensorflow.keras.models import load_model


global graph
graph = tf.compat.v1.get_default_graph()


class ModelSegmentator:
    def __init__(self, model_dir_path):
        self._features_processor = FeaturesProcessorSegmentation(model_dir_path)
        with graph.as_default():
            self._model = load_model(os.path.join(model_dir_path, 'segmentator', 'neural_model.h5'))

    def __call__(self, *args, **kwargs):
        """
        :param args: 'text', 'tokens', 'sentences', 'lemma', 'postag', 'syntax_dep_tree' values of an isanlp annotation
        :return: list of DiscourseUnit
        """
        features = self._features_processor(*args)
        return self._build_discourse_units(args[0], args[1], self._predict(features))

    def _predict(self, features):
        """
        :param list features: features to feed directly into the model
        :return: numbers of tokens predicted as EDU right boundaries
        """
        print('_predict(...) started...')

        with graph.as_default():
            predictions = self._model.predict(features)  
        
        print('_predict(...) finished...')
        return [addr[0] for addr in np.argwhere(np.argmax(predictions, axis=1))]

    def _build_discourse_units(self, text, tokens, numbers):
        """
        :param text: original text
        :param list tokens: isanlp.annotation.Token
        :param numbers: positions of tokens predicted as EDU left boundaries (beginners)
        :return: list of DiscourseUnit
        """
        edus = []

        new_edu = DiscourseUnit(0,
                                start=0,
                                end=tokens[numbers[1]].start,
                                text=text[:tokens[numbers[1]].start],
                                relation='elementary')
        edus.append(new_edu)

        for i in range(1, len(numbers)):
            new_edu = DiscourseUnit(i,
                                    start=tokens[numbers[i - 1]].end,
                                    end=tokens[numbers[i]].end,
                                    text=text[tokens[numbers[i - 1]].end:tokens[numbers[i]].end],
                                    relation='elementary')
            edus.append(new_edu)

        return edus
