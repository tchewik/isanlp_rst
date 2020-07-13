import os
import pickle

import numpy as np
import pandas as pd
from allennlp.predictors import Predictor
# from models.customization_package.model.custom_bimpm_predictor import CustomBiMPMPredictor
# from models.customization_package.dataset_readers.custom_reader import CustomDataReader
from models.bimpm_custom_package.model.custom_bimpm_predictor import CustomBiMPMPredictor
from models.bimpm_custom_package.dataset_readers.custom_reader import CustomDataReader


from .symbol_map import SYMBOL_MAP


##from models.customization_package2.model.contextual_bimpm_predictor import ContextualBiMpmPredictor
##from models.customization_package2.dataset_readers.contextual_reader import ContextualReader


class SimpleAllenNLPClassifier:
    def __init__(self, model_dir_path, cuda_device):
        self.model_dir_path = model_dir_path
        self._cuda_device = cuda_device

        file_labels = os.path.join(self.model_dir_path, 'vocabulary/labels.txt')
        self.labels = open(file_labels, 'r').readlines() if os.path.isfile(
            file_labels) else None
        self.labels = [label.strip() for label in self.labels]

        self._max_len = 100
        self._symbol_map = SYMBOL_MAP

        self._left_dummy_placement = '-'
        self._right_dummy_placement = '###'


class AllenNLPBiMPMClassifier(SimpleAllenNLPClassifier):
    """
    Wrapper for allennlp classification model along with preprocessors, saved in the same directory:
        [required]
        - model.tar.gz            : trained model

    Predicts labels and probabilities on the data with fields:
        - Left span tokens
        - Right span tokens
    """

    def __init__(self, model_dir_path, cuda_device=-1):
        SimpleAllenNLPClassifier.__init__(self, model_dir_path, cuda_device)

        self._model = Predictor.from_path(os.path.join(self.model_dir_path, 'model.tar.gz'),
                                          predictor_name='textual-entailment',
                                          cuda_device=self._cuda_device)

    def predict_proba(self, snippet_x, snippet_y, *args, **kwargs):
        _snippet_x = self._prepare_sequence(snippet_x, is_left_snippet=True)
        _snippet_y = self._prepare_sequence(snippet_y, is_left_snippet=True)

        if len(_snippet_x.split()) == 0 or len(_snippet_y.split()) == 0 or len(
                _snippet_x.split()) > self._max_len or len(
                _snippet_y.split()) > self._max_len:
            return [1., 0.]

        return self._model.predict(_snippet_x, _snippet_y)['probs']

    def predict_proba_batch(self, snippet_x, snippet_y, *args, **kwargs):
        predictions = self._model.predict_batch_json([
            {'premise': self._prepare_sequence(snippet_x[i], is_left_snippet=True),
             'hypothesis': self._prepare_sequence(snippet_y[i], is_left_snippet=False)} if 0 < len(
                snippet_x[i]) and 0 < len(snippet_y[i]) else
            {'premise': self._left_dummy_placement, 'hypothesis': self._right_dummy_placement}
            for i in range(len(snippet_x))])

        return [prediction['probs'] for prediction in predictions]

    def predict(self, snippet_x, snippet_y, *args, **kwargs):
        _snippet_x = self._prepare_sequence(snippet_x, is_left_snippet=True)
        _snippet_y = self._prepare_sequence(snippet_y, is_left_snippet=True)

        if len(_snippet_x.split()) == 0 or len(_snippet_y.split()) == 0 or len(
                _snippet_x.split()) > self._max_len or len(
                _snippet_y.split()) > self._max_len:
            return 'other_NN'

        return self._model.predict(_snippet_x, _snippet_y)['label']

    def predict_batch(self, snippet_x, snippet_y, *args, **kwargs):
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


class AllenNLPCustomBiMPMClassifier(SimpleAllenNLPClassifier):
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

    def __init__(self, model_dir_path, cuda_device=-1):
        SimpleAllenNLPClassifier.__init__(self, model_dir_path, cuda_device)

        self._model = CustomBiMPMPredictor.from_path(os.path.join(self.model_dir_path, 'model.tar.gz'),
                                                     predictor_name='custom_bimpm_predictor',
                                                     cuda_device=self._cuda_device)

    def predict_proba(self, snippet_x, snippet_y, same_sentence, same_paragraph, *args, **kwargs):
        _snippet_x = self._prepare_sequence(snippet_x, is_left_snippet=True)
        _snippet_y = self._prepare_sequence(snippet_y, is_left_snippet=True)

        if len(_snippet_x.split()) == 0 or len(_snippet_y.split()) == 0 or len(
                _snippet_x.split()) > self._max_len or len(
                _snippet_y.split()) > self._max_len:
            return [1., 0.]

        return self._model.predict(_snippet_x, _snippet_y, same_sentence, same_paragraph)['probs']

    def predict_proba_batch(self, snippet_x, snippet_y, same_sentence, same_paragraph, *args, **kwargs):
        predictions = self._model.predict_batch_json([
            {'premise': self._prepare_sequence(snippet_x[i], is_left_snippet=True),
             'hypothesis': self._prepare_sequence(snippet_y[i], is_left_snippet=False),
             'same_sentence': same_sentence[i],
             'same_paragraph': same_paragraph[i]} if 0 < len(
                snippet_x[i]) and 0 < len(snippet_y[i]) else
            {'premise': self._left_dummy_placement, 'hypothesis': self._right_dummy_placement, 'same_sentence': '0', 'same_paragraph': '0'}
            for i in range(len(snippet_x))])

        return [prediction['probs'] for prediction in predictions]

    def predict(self, snippet_x, snippet_y, same_sentence, same_paragraph, *args, **kwargs):
        _snippet_x = self._prepare_sequence(snippet_x, is_left_snippet=True)
        _snippet_y = self._prepare_sequence(snippet_y, is_left_snippet=True)

        if len(_snippet_x.split()) == 0 or len(_snippet_y.split()) == 0 or len(
                _snippet_x.split()) > self._max_len or len(
                _snippet_y.split()) > self._max_len:
            return 'other_NN'

        return self._model.predict(_snippet_x, _snippet_y, same_sentence, same_paragraph)['label']

    def predict_batch(self, snippet_x, snippet_y, same_sentence, same_paragraph, *args, **kwargs):
        predictions = self._model.predict_batch_json([
            {'premise': self._prepare_sequence(snippet_x[i], is_left_snippet=True),
             'hypothesis': self._prepare_sequence(snippet_y[i], is_left_snippet=False),
             'same_sentence': same_sentence[i],
             'same_paragraph': same_paragraph[i]} if 0 < len(
                snippet_x[i]) <= self._max_len and 0 < len(
                snippet_y[i]) <= self._max_len else
            {'premise': self._left_dummy_placement, 'hypothesis': self._right_dummy_placement, 'same_sentence': '0', 'same_paragraph': '0'}
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


class AllenNLPContextualBiMPMClassifier(SimpleAllenNLPClassifier):
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

    def __init__(self, model_dir_path, cuda_device=-1):
        SimpleAllenNLPClassifier.__init__(self, model_dir_path, cuda_device)

        self._model = ContextualBiMpmPredictor.from_path(os.path.join(self.model_dir_path, 'model.tar.gz'),
                                                         predictor_name='contextual_bimpm_predictor',
                                                         cuda_device=self._cuda_device)

    def predict_proba(self, snippet_x, snippet_y, features, left_context, right_context, *args, **kwargs):
        _snippet_x = self._prepare_sequence(snippet_x, is_left_snippet=True)
        _snippet_y = self._prepare_sequence(snippet_y, is_left_snippet=False)
        _left_context = self._prepare_sequence(left_context, is_left_snippet=True)
        _right_context = self._prepare_sequence(right_context[i], is_left_snippet=False)

        if len(_snippet_x.split()) == 0 or len(_snippet_y.split()) == 0 or len(
                _snippet_x.split()) > self._max_len or len(
                _snippet_y.split()) > self._max_len:
            return [1., 0.]

        return self._model.predict(_snippet_x, _snippet_y, features,
                                   left_context=_left_context,
                                   right_context=_right_context)['probs']

    def predict_proba_batch(self, snippet_x, snippet_y, features, left_context, right_context, *args, **kwargs):
        predictions = self._model.predict_batch_json([
            {'premise': self._prepare_sequence(snippet_x[i], is_left_snippet=True),
             'hypothesis': self._prepare_sequence(snippet_y[i], is_left_snippet=False),
             'left_context': self._prepare_sequence(left_context[i], is_left_snippet=True),
             'right_context': self._prepare_sequence(right_context[i], is_left_snippet=False),
             'metadata': features[i]} if 0 < len(
                snippet_x[i]) and 0 < len(snippet_y[i]) else
            {'premise': self._left_dummy_placement, 'hypothesis': self._right_dummy_placement, 'metadata': '0',
             'left_context': ' '.join([self._left_dummy_placement] * self._context_length),
             'right_context': ' '.join([self._right_dummy_placement] * self._context_length)}
            for i in range(len(snippet_x))])

        return [prediction['probs'] for prediction in predictions]

    def predict(self, snippet_x, snippet_y, features, left_context, right_context, *args, **kwargs):
        _snippet_x = self._prepare_sequence(snippet_x, is_left_snippet=True)
        _snippet_y = self._prepare_sequence(snippet_y, is_left_snippet=True)
        _left_context = self._prepare_sequence(left_context[i], is_left_snippet=True)
        _right_context = self._prepare_sequence(right_context[i], is_left_snippet=False)

        if len(_snippet_x.split()) == 0 or len(_snippet_y.split()) == 0 or len(
                _snippet_x.split()) > self._max_len or len(
                _snippet_y.split()) > self._max_len:
            return 'other_NN'

        return self._model.predict(_snippet_x, _snippet_y, features,
                                   left_context=_left_context,
                                   right_context=_right_context)['label']

    def predict_batch(self, snippet_x, snippet_y, features, left_context, right_context, *args, **kwargs):
        predictions = self._model.predict_batch_json([
            {'premise': self._prepare_sequence(snippet_x[i], is_left_snippet=True),
             'hypothesis': self._prepare_sequence(snippet_y[i], is_left_snippet=False),
             'left_context': self._prepare_sequence(left_context[i], is_left_snippet=True),
             'right_context': self._prepare_sequence(right_context[i], is_left_snippet=False),
             'metadata': features[i]} if 0 < len(
                snippet_x[i]) <= self._max_len and 0 < len(
                snippet_y[i]) <= self._max_len else
            {'premise': self._left_dummy_placement, 'hypothesis': self._right_dummy_placement, 'metadata': '0',
             'left_context': ' '.join([self._left_dummy_placement] * self._context_length),
             'right_context': ' '.join([self._right_dummy_placement] * self._context_length)}
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


class SklearnClassifier:
    """
    Wrapper for sklearn/catboost classification model along with preprocessors, saved in the same directory:
        [required]
        - model.pkl            : trained model
        [optional]
        - drop_columns.pkl     : list of names for columns to drop before prediction
        - categorical_cols.pkl : list of names for columns with categorical features
        - one_hot_encoder.pkl  : trained one-hot sklearn encoder model
        - scaler.pkl           : trained sklearn scaler model
        - label_encoder.pkl    : trained label encoder to decode predictions
    """

    def __init__(self, model_dir_path, *args, **kwargs):
        self.model_dir_path = model_dir_path

        file_drop_columns = os.path.join(self.model_dir_path, 'drop_columns.pkl')
        self._drop_columns = pickle.load(open(file_drop_columns, 'rb')) if os.path.isfile(
            file_drop_columns) else None
        if self._drop_columns:
            self._drop_columns = [value for value in self._drop_columns if
                                  not value in ('category_id' 'filename' 'order')]

        file_scaler = os.path.join(self.model_dir_path, 'scaler.pkl')
        self._scaler = pickle.load(open(file_scaler, 'rb')) if os.path.isfile(
            file_scaler) else None

        file_categorical_cols = os.path.join(self.model_dir_path, 'categorical_cols.pkl')
        self._categorical_cols = pickle.load(open(file_categorical_cols, 'rb')) if os.path.isfile(
            file_categorical_cols) else None

        file_one_hot_encoder = os.path.join(self.model_dir_path, 'one_hot_encoder.pkl')
        self._one_hot_encoder = pickle.load(open(file_one_hot_encoder, 'rb')) if os.path.isfile(
            file_one_hot_encoder) else None

        file_label_encoder = os.path.join(self.model_dir_path, 'label_encoder.pkl')
        self._label_encoder = pickle.load(open(file_label_encoder, 'rb')) if os.path.isfile(
            file_label_encoder) else None

        self._model = pickle.load(open(os.path.join(self.model_dir_path, 'model.pkl'), 'rb'))
        self.classes_ = self._model.classes_
        
        if self._label_encoder:
            self.labels = self._label_encoder.classes_
        else:
            self.labels = list(map(str, self.classes_))

    def predict_proba(self, features, *args, **kwargs):
        try:
            predictions = self._model.predict_proba(self._preprocess_features(features))[0]
        except AttributeError:
            predictions = self._model._predict_proba_lr(self._preprocess_features(features))[0]
        return predictions

    def predict_proba_batch(self, features, *args, **kwargs):
        try:
            predictions = self._model.predict_proba(self._preprocess_features(features))
        except AttributeError:
            predictions = self._model._predict_proba_lr(self._preprocess_features(features))
        return predictions

    def predict(self, features, *args, **kwargs):
        if self._label_encoder:
            return self._label_encoder.inverse_transform(self._model.predict(self._preprocess_features(features)))

        return self._model.predict(self._preprocess_features(features))

    def predict_batch(self, features, *args, **kwargs):
        return self.predict(self, features, *args, **kwargs)

    def _preprocess_features(self, _features):
        features = _features[:]

        if self._categorical_cols:
            if self._label_encoder:
                features[self._categorical_cols] = features[self._categorical_cols].apply(
                    lambda col: self._label_encoder.fit_transform(col))

            if self._one_hot_encoder:
                features_ohe = self._one_hot_encoder.transform(features[self._categorical_cols].values)
                features_ohe = pd.DataFrame(features_ohe, features.index,
                                            columns=self._one_hot_encoder.get_feature_names(self._categorical_cols))

                features = features.join(
                    pd.DataFrame(features_ohe, features.index).add_prefix('cat_'), how='right'
                ).drop(columns=self._categorical_cols)

        if self._drop_columns:
            for _column in self._drop_columns:
                try:
                    features = features.drop(columns=[_column])
                except KeyError as e:
                    pass

        if 'category_id' in features.keys():
            features = features.drop(columns=['category_id', 'filename', 'order'])

        if self._scaler:
            return self._scaler.transform(features.values)

        return features.values.astype('float64')


class EnsembleClassifier:
    """
    Wrapper for voting ensemble
    """

    def __init__(self, models, voting_type='soft'):
        """
        :param models: list of initialized models
        :param voting_type: type of voting {soft|hard}
        """
        for i in range(1, len(models)):
            assert set(models[i].labels) == set(models[i - 1].labels)

        self.models = models
        self.labels = models[0].labels

        self.voting_type = voting_type
        self.vote = np.max if self.voting_type == 'hard' else np.mean

    def predict_proba(self, snippet_x, snippet_y, features, *args, **kwargs):
        results = []

        for model in self.models:
            sample_prediction = model.predict_proba(snippet_x=snippet_x, snippet_y=snippet_y, features=features, *args, **kwargs)
            results.append(dict(zip(model.labels, sample_prediction)))

        ensembled_result = {key: self.vote([result[key] for result in results]) for key in self.labels}

        return [ensembled_result[key] for key in self.labels]

    def predict_proba_batch(self, snippet_x, snippet_y, features, *args, **kwargs):
        results = []

        for model in self.models:
            model_predictions = model.predict_proba_batch(snippet_x=snippet_x, snippet_y=snippet_y, features=features, *args, **kwargs)

            annot_predictions = []
            for sample_prediction in model_predictions:
                annot_predictions.append(dict(zip(model.labels, sample_prediction)))

            results.append(annot_predictions)

        ensembled_result = []

        for i in range(len(results[0])):        
            ensembled_result.append(
                {key: self.vote([result[i][key] for result in results]) for key in self.labels})

        return [[sample_result[key] for key in self.labels] for sample_result in ensembled_result]

    def predict(self, snippet_x, snippet_y, features, *args, **kwargs):
        proba = self.predict_proba(snippet_x=snippet_x, snippet_y=snippet_y, features=features, *args, **kwargs)

        return self.labels[np.argmax(proba)]

    def predict_batch(self, snippet_x, snippet_y, features, *args, **kwargs):
        result = []

        proba = self.predict_proba_batch(snippet_x=snippet_x, snippet_y=snippet_y, features=features, *args, **kwargs)
        for sample in proba:
            result.append(self.labels[np.argmax(sample)])

        return result
