import numpy as np
import pandas as pd


def get_embeddings(embedder, X, maxlen=100):
    X_ = [text[:text.rfind('_')] for text in X.split()]
    result = np.zeros((embedder.vector_size, maxlen))

    for i in range(min(len(X_), maxlen)):
        try:
            result[i] = embedder[X_[i]]
        except KeyError:
            continue

    return result


class FeaturesExtractor:
    DROP_COLUMNS = ['snippet_x', 'snippet_y', 'snippet_x_tmp', 'snippet_y_tmp', 'postags_x', 'postags_y']

    def __init__(self, processor, scaler, categorical_cols, one_hot_encoder, label_encoder):
        self.processor = processor
        self.scaler = scaler
        self._categorical_cols = categorical_cols
        self.one_hot_encoder = one_hot_encoder
        self.label_encoder = label_encoder

    def __call__(self, df, annot_text, annot_tokens, annot_sentences, annot_postag, annot_morph, annot_lemma, annot_syntax_dep_tree):
        X = self.processor(df, annot_text, annot_tokens, annot_sentences, annot_postag, annot_morph, annot_lemma, annot_syntax_dep_tree)

        if self._categorical_cols:
            if self.label_encoder:
                X[self._categorical_cols] = X[self._categorical_cols].apply(lambda col: self.label_encoder.fit_transform(col))

            if self.one_hot_encoder:
                X_ohe = self.one_hot_encoder.transform(X[self._categorical_cols].values)
                X_ohe = pd.DataFrame(X_ohe, X.index, columns=self.one_hot_encoder.get_feature_names(self._categorical_cols))

                X = X.join(
                    pd.DataFrame(X_ohe, X.index).add_prefix('cat_'), how='right'
                ).drop(columns=self._categorical_cols).drop(columns=self.DROP_COLUMNS)

        if self.scaler:
            return pd.DataFrame(self.scaler.transform(X.values), index=X.index, columns=X.columns)

        return X
