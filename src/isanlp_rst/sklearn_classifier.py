import os
import pickle

import pandas as pd


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

    def __init__(self, model_dir_path):
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

    def predict_proba(self, features):
        return self._model.predict_proba(self._preprocess_features(features))

    def predict(self, features):
        if self._label_encoder:
            return self._label_encoder.inverse_transform(self._model.predict(self._preprocess_features(features)))

        return self._model.predict(self._preprocess_features(features))

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
            features = features.drop(columns=self._drop_columns)

        if 'category_id' in features.keys():
            features = features.drop(columns=['category_id', 'filename', 'order'])

        if self._scaler:
            return self._scaler.transform(features.values)

        return features.values.astype('float64')
