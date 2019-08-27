import os
import pickle

import pandas as pd


class SklearnClassifier:
    def __init__(self, model_dir_path):
        self.model_dir_path = model_dir_path

        file_drop_columns = os.path.join(self.model_dir_path, 'drop_columns.pkl')
        self._drop_columns = pickle.load(open(file_drop_columns, 'rb')) if os.path.isfile(
            file_drop_columns) else None

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
            file_one_hot_encoder) else None

        self._model = pickle.load(open(os.path.join(self.model_dir_path, 'model.pkl'), 'rb'))

    def predict_proba(self, features):
        return self._model.predict_proba(self._preprocess_features(features))

    def predict(self, features):
        if self._label_encoder:
            return self._label_encoder.inverse_transform(self._model.predict(self._preprocess_features(features)))

        return self._model.predict(self._preprocess_features(features))

    def _preprocess_features(self, features):
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

        if self._scaler:
            return pd.DataFrame(self._scaler.transform(features.values), index=features.index, columns=features.columns)

        return features.values.astype('float64')
