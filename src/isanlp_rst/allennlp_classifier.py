import os

from allennlp.predictors import Predictor
from symbol_map import SYMBOL_MAP


class AllenNLPClassifier:
    """
    Wrapper for allennlp classification model along with preprocessors, saved in the same directory:
        [required]
        - model.tar.gz            : trained model

    Predicts labels and probabilities on the data with fields:
        - Left span tokens
        - Right span tokens
    """

    def __init__(self, model_dir_path):
        self.model_dir_path = model_dir_path
        self._max_len = 100

        self._symbol_map = SYMBOL_MAP
        self._left_dummy_placement = '-'
        self._right_dummy_placement = '###'

        self._model = Predictor.from_path(os.path.join(self.model_dir_path, 'model.tar.gz'), 'textual-entailment')

    def predict_proba(self, snippet_x, snippet_y):
        _snippet_x = self._prepare_sequence(snippet_x, is_left_snippet=True)
        _snippet_y = self._prepare_sequence(snippet_y, is_left_snippet=True)

        if len(_snippet_x.split()) == 0 or len(_snippet_y.split()) == 0 or len(
                _snippet_x.split()) > self._max_len or len(
            _snippet_y.split()) > self._max_len:
            return [1., 0.]

        return self._model.predict(_snippet_x, _snippet_y)['probs']

    def predict_proba_batch(self, snippet_x, snippet_y):
        predictions = self._model.predict_batch_json([
            {'premise': self._prepare_sequence(snippet_x[i], is_left_snippet=True),
             'hypothesis': self._prepare_sequence(snippet_y[i], is_left_snippet=False)} if 0 < len(
                snippet_x[i]) and 0 < len(snippet_y[i]) else
            {'premise': self._left_dummy_placement, 'hypothesis': self._right_dummy_placement}
            for i in range(len(snippet_x))])

        return [prediction['probs'] for prediction in predictions]

    def predict(self, snippet_x, snippet_y):
        _snippet_x = self._prepare_sequence(snippet_x, is_left_snippet=True)
        _snippet_y = self._prepare_sequence(snippet_y, is_left_snippet=True)

        if len(_snippet_x.split()) == 0 or len(_snippet_y.split()) == 0 or len(
                _snippet_x.split()) > self._max_len or len(
            _snippet_y.split()) > self._max_len:
            return 'other_NN'

        return self._model.predict(_snippet_x, _snippet_y)['label']

    def predict_batch(self, snippet_x, snippet_y):
        predictions = self._model.predict_batch_json([
            {'premise': self._prepare_sequence(snippet_x[i], is_left_snippet=True),
             'hypothesis': self._prepare_sequence(snippet_y[i], is_left_snippet=False)} if 0 < len(
                snippet_x[i]) <= self._max_len and 0 < len(
                snippet_y[i]) <= self._max_len else
            {'premise': self._left_dummy_placement, 'hypothesis': self._right_dummy_placement}
            for i in range(len(snippet_x))])

        return [prediction['label'] for prediction in predictions]

    def _prepare_sequence(self, sequence, is_left_snippet):

        result = []
        iterator = sequence.split() if type(sequence) == str else sequence

        if len(iterator) > self._max_len:
            return is_left_snippet * self._left_dummy_placement + (not is_left_snippet) * self._right_dummy_placement

        for token in iterator:

            for key, value in self._symbol_map.items():
                token = token.replace(key, value)

            for keyword in ['www', 'http']:
                if keyword in token:
                    token = '_html_'

            result.append(token)

        return ' '.join(result)
