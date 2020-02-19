import numpy as np
import pandas as pd


def get_embeddings(embedder, x, maxlen=100):
    x_ = [text[:text.rfind('_')] for text in x.split()]
    result = np.zeros((embedder.vector_size, maxlen))

    for i in range(min(len(x_), maxlen)):
        try:
            result[i] = embedder[x_[i]]
        except KeyError:
            continue

    return result


class FeaturesExtractor:
    DROP_COLUMNS = ['snippet_x', 'snippet_y', 'snippet_x_tmp', 'snippet_y_tmp', 'postags_x', 'postags_y']

    def __init__(self, processor, scaler=None, categorical_cols=None, one_hot_encoder=None, label_encoder=None):
        self.processor = processor
        self.scaler = scaler
        self._categorical_cols = categorical_cols
        self.one_hot_encoder = one_hot_encoder
        self.label_encoder = label_encoder

    def __call__(self, df, 
                 annot_text, annot_tokens, annot_sentences, 
                 annot_lemma, annot_morph, annot_postag, annot_syntax_dep_tree):
        x = self.processor(df, 
                           annot_text, annot_tokens, annot_sentences, 
                           annot_lemma, annot_morph, annot_postag, annot_syntax_dep_tree)

        if self._categorical_cols:
            if self.label_encoder:
                x[self._categorical_cols] = x[self._categorical_cols].apply(
                    lambda col: self.label_encoder.fit_transform(col))

            if self.one_hot_encoder:
                x_ohe = self.one_hot_encoder.transform(x[self._categorical_cols].values)
                x_ohe = pd.DataFrame(x_ohe, x.index,
                                     columns=self.one_hot_encoder.get_feature_names(self._categorical_cols))

                x = x.join(
                    pd.DataFrame(x_ohe, x.index).add_prefix('cat_'), how='right'
                ).drop(columns=self._categorical_cols).drop(columns=self.DROP_COLUMNS)

        if self.scaler:
            return pd.DataFrame(self.scaler.transform(x.values), index=x.index, columns=x.columns)

        return x
