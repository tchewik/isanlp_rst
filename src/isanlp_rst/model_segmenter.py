import os

import numpy as np
import tensorflow as tf
from features_processor_segmentation import FeaturesProcessorSegmentation
from isanlp.annotation_rst import DiscourseUnit
from tensorflow.keras.models import load_model


global graph
graph = tf.compat.v1.get_default_graph()

global sess
sess = tf.compat.v1.Session()


class ModelSegmenter:
    def __init__(self, model_dir_path):
        self._features_processor = FeaturesProcessorSegmentation(model_dir_path)
        self._confidence_threshold = 0.5
        
        with sess.as_default():
            with graph.as_default():
                self._model = load_model(os.path.join(model_dir_path, 'segmentator', 'neural_model.h5'))

    def __call__(self, *args, **kwargs):
        """
        :param args: 'text', 'tokens', 'sentences', 'lemma', 'postag', 'syntax_dep_tree' values of an isanlp annotation
        :return: list of DiscourseUnit
        """
        features, sentence_boundaries = self._features_processor(*args)
        return self._build_discourse_units(args[0], args[1], self._predict(features, sentence_boundaries))

    def _predict(self, features, sentence_boundaries):
        """
        :param list features: features to feed directly into the model
        :param np.array sentence_boundaries: 1D binary array of sentence boundary markers for each token
        :param float confidence_threshold: threshold to apply to models softmax predictions
        :return: numbers of tokens predicted as EDU left boundaries
        """

        with sess.as_default():
            with graph.as_default():
                predictions = self._model.predict(features)
                
        predictions = predictions[:,1] > self._confidence_threshold
        augmented_predictions = predictions.astype(int) | sentence_boundaries.astype(int)
        return np.argwhere(augmented_predictions)[:, 0]

    def _build_discourse_units(self, text, tokens, numbers):
        """
        :param text: original text
        :param list tokens: isanlp.annotation.Token
        :param numbers: positions of tokens predicted as EDU left boundaries (beginners)
        :return: list of DiscourseUnit
        """
        
        edus = []
    
        if numbers.shape[0]:
            for i in range(0, len(numbers)-1):
                new_edu = DiscourseUnit(i,
                                        start=tokens[numbers[i]].begin,
                                        end=tokens[numbers[i+1]].begin - 1,
                                        text=text[tokens[numbers[i]].begin:tokens[numbers[i+1]].begin],
                                        relation='elementary')
                edus.append(new_edu)

            if numbers.shape[0] == 1:
                i = -1
            
            new_edu = DiscourseUnit(i+1,
                            start=tokens[numbers[-1]].begin,
                            end=len(text),
                            text=text[tokens[numbers[-1]].begin:],
                            relation='elementary')
            edus.append(new_edu)

        return edus
