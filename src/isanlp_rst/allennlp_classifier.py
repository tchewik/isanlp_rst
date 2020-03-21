import os

from allennlp.predictors import Predictor


class AllenNLPClassifier:
    """
    Wrapper for allennlp classification model along with preprocessors, saved in the same directory:
        [required]
        - model.tar.gz            : trained model
    """

    def __init__(self, model_dir_path):
        self.model_dir_path = model_dir_path
        self._max_len = 300

        self._model = Predictor.from_path(os.path.join(self.model_dir_path, 'model.tar.gz'))

    def predict_proba(self, snippet_x, snippet_y):
        if len(snippet_x.split()) > self._max_len or len(snippet_x.split()) == 0 or len(
                snippet_y.split()) > self._max_len or len(snippet_y.split()) == 0:
            return [1., 0.]

        return self._model.predict(snippet_x, snippet_y)['probs']

    def predict_proba_batch(self, snippet_x, snippet_y):
        predictions = self._model.predict_batch_json([
            {'premise': snippet_x[i],
             'hypothesis': snippet_y[i]} if 0 < len(snippet_x[i].split()) <= self._max_len and 0 < len(
                snippet_y[i].split()) <= self._max_len else
            {'premise': '1', 'hypothesis': '-'}
            for i in range(len(snippet_x))])

        return [prediction['probs'] for prediction in predictions]

    def predict(self, snippet_x, snippet_y):
        return self._model.predict(snippet_x, snippet_y)['label']
