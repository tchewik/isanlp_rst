from isanlp.annotation_rst import DiscourseUnit
import pandas as pd


class RSTTreePredictor:
    def __init__(self, features_processor, relation_predictor, label_predictor):
        self.features_processor = features_processor
        self.relation_predictor = relation_predictor
        self.label_predictor = label_predictor
        if self.label_predictor:
            self.labels = self.label_predictor.classes_
        self.genre = None

    def predict_label(self, features):
        if not self.label_predictor:
            return 'relation'

        return self.label_predictor.predict(features)


class GoldTreePredictor(RSTTreePredictor):
    def __init__(self, corpus):
        RSTTreePredictor.__init__(self, None, None, None)
        self.corpus = corpus

    def extract_features(self, *args):
        return [args[0].text, args[1].text]

    def predict_pair_proba(self, features):
        def _check_snippet_pair_in_dataset(left_snippet, right_snippet):
            return ((((self.corpus.snippet_x == left_snippet) & (self.corpus.snippet_y == right_snippet)).sum(
                axis=0) != 0)
                    or ((self.corpus.snippet_y == left_snippet) & (self.corpus.snippet_x == right_snippet)).sum(
                        axis=0) != 0)

        left_snippet, right_snippet = features
        return float(_check_snippet_pair_in_dataset(left_snippet, right_snippet))

    def predict_label(self, features):
        if not self.label_predictor:
            return 'relation'


class CustomTreePredictor(RSTTreePredictor):
    def __init__(self, features_processor, relation_predictor, label_predictor=None):
        RSTTreePredictor.__init__(self, features_processor, relation_predictor, label_predictor)

    def extract_features(self, left_node: DiscourseUnit, right_node: DiscourseUnit,
                         annot_text, annot_tokens, annot_sentences, annot_postag, annot_morph, annot_lemma,
                         annot_syntax_dep_tree):
        pair = pd.DataFrame({
            'snippet_x': [left_node.text.strip()],
            'snippet_y': [right_node.text.strip()],
            #'genre': self.genre
        })

        try:
            features = self.features_processor(pair, annot_text=annot_text,
                                               annot_tokens=annot_tokens, annot_sentences=annot_sentences,
                                               annot_postag=annot_postag, annot_morph=annot_morph,
                                               annot_lemma=annot_lemma, annot_syntax_dep_tree=annot_syntax_dep_tree)
            return features
        except IndexError:
            with open('errors.log', 'w+') as f:
                f.write(str(pair.values))
                f.write(annot_text)
            return -1

    def predict_pair_proba(self, features):
        return self.relation_predictor.predict_proba(features)[0][1]
