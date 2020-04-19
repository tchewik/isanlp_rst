import os

from models.customization_package.model.custom_bimpm_predictor import CustomBiMPMPredictor
from models.customization_package.dataset_readers.custom_reader import CustomDataReader


class AllenNLPClassifier:
    """
    Wrapper for custom BiMPM allennlp classification model along with preprocessors, saved in the same directory:
        [required]
        - model.tar.gz            : trained model
        
    Also requires:
        - models/customization_package/*            : scripts for the custom models
        
    Predicts labels and probabilities on the data with fields:
        - Left span tokens
        - Right span tokens
        - Additional features
    """

    def __init__(self, model_dir_path):
        self.model_dir_path = model_dir_path
        self._max_len = 100

        self._symbol_map = {
            'x': 'х',
            'X': 'X',
            'y': 'у',
            '—': '-',
            '“': '«',
            '‘': '«',
            '”': '»',
            '’': '»',
            '😆': '😄',
            '😊': '😄',
            '😑': '😄',
            '😔': '😄',
            '😉': '😄',
            '❗': '😄',
            '🤔': '😄',
            '😅': '😄',
            '⚓': '😄',
            'ε': 'α',
            'ζ': 'α',
            'η': 'α',
            'μ': 'α',
            'δ': 'α',
            'λ': 'α',
            'ν': 'α',
            'β': 'α',
            'γ': 'α',
            'と': '尋',
            'の': '尋',
            '神': '尋',
            '隠': '尋',
            'し': '尋',
        }

        self._left_dummy_placement = '-'
        self._right_dummy_placement = '###'

        self._model = CustomBiMPMPredictor.from_path(os.path.join(self.model_dir_path, 'model.tar.gz'),
                                                     predictor_name='custom_bimpm_predictor')

    def predict_proba(self, snippet_x, snippet_y, features):
        if len(snippet_x.split()) == 0 or len(snippet_y.split()) == 0:
            return [1., 0.]

        return self._model.predict(self._prepare_sequence(snippet_x, is_left_snippet=True),
                                   self._prepare_sequence(snippet_y, is_left_snippet=False),
                                   features)['probs']

    def predict_proba_batch(self, snippet_x, snippet_y, features):
        predictions = self._model.predict_batch_json([
            {'premise': self._prepare_sequence(snippet_x[i], is_left_snippet=True),
             'hypothesis': self._prepare_sequence(snippet_y[i], is_left_snippet=False),
             'metadata': features[i]} if 0 < len(
                snippet_x[i]) and 0 < len(snippet_y[i]) else
            {'premise': self._left_dummy_placement, 'hypothesis': self._right_dummy_placement, 'metadata': '0'}
            for i in range(len(snippet_x))])

        return [prediction['probs'] for prediction in predictions]

    def predict(self, snippet_x, snippet_y, features):
        return self._model.predict(self._prepare_sequence(snippet_x, is_left_snippet=True),
                                   self._prepare_sequence(snippet_y, is_left_snippet=False),
                                   features)['label']

    def predict_batch(self, snippet_x, snippet_y, features):
        predictions = self._model.predict_batch_json([
            {'premise': self._prepare_sequence(snippet_x[i], is_left_snippet=True),
             'hypothesis': self._prepare_sequence(snippet_y[i], is_left_snippet=False),
             'metadata': features[i]} if 0 < len(
                snippet_x[i]) <= self._max_len and 0 < len(
                snippet_y[i]) <= self._max_len else
            {'premise': self._left_dummy_placement, 'hypothesis': self._right_dummy_placement, 'metadata': '0'}
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
