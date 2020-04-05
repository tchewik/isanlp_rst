import pandas as pd
from isanlp.annotation_rst import DiscourseUnit


class RSTTreePredictor:
    """
    Contains classifiers and processors needed for tree building.
    """

    def __init__(self, features_processor, relation_predictor, label_predictor, nuclearity_predictor):
        self.features_processor = features_processor
        self.relation_predictor = relation_predictor
        self.label_predictor = label_predictor

        self.nuclearity_predictor = nuclearity_predictor
        if self.nuclearity_predictor:
            self.nuclearities = self.nuclearity_predictor.classes_

        self.genre = None


class GoldTreePredictor(RSTTreePredictor):
    """
    Contains classifiers and processors needed for gold tree building from corpus.
    """

    def __init__(self, corpus):
        """
        :param pandas.DataFrame corpus:
            columns=['snippet_x', 'snippet_y', 'category_id']
            rows=[all the relations pairs from corpus]
        """
        RSTTreePredictor.__init__(self, None, None, None, None)
        self.corpus = corpus

    def extract_features(self, *args):
        return pd.DataFrame({
            'snippet_x': [args[0].text, ],
            'snippet_y': [args[1].text, ]
        })

    def initialize_features(self, *args):
        return pd.DataFrame({
            'snippet_x': [args[0][i].text for i in range(len(args[0]) - 1)],
            'snippet_y': [args[0][i].text for i in range(1, len(args[0]))]
        })

    def predict_pair_proba(self, features):
        def _check_snippet_pair_in_dataset(left_snippet, right_snippet):
            return float((((self.corpus.snippet_x == left_snippet) & (self.corpus.snippet_y == right_snippet)).sum(
                axis=0) != 0)
                         or ((self.corpus.snippet_y == left_snippet) & (self.corpus.snippet_x == right_snippet)).sum(
                axis=0) != 0)

        result = features.apply(lambda row: _check_snippet_pair_in_dataset(row.snippet_x, row.snippet_y), axis=1)
        return result.values.tolist()

    def predict_label(self, features):
        def _get_label(left_snippet, right_snippet):
            label = self.corpus[
                ((self.corpus.snippet_x == left_snippet) & (self.corpus.snippet_y == right_snippet))].category_id.values
            if label.size == 0:
                return 'relation'

            return label[0]

        if type(features) == pd.Series:
            result = _get_label(features.loc['snippet_x'], features.loc['snippet_y'])
            return result
        else:
            result = features.apply(lambda row: _get_label(row.snippet_x, row.snippet_y), axis=1)
            return result.values.tolist()

    def predict_nuclearity(self, features):
        def _get_nuclearity(left_snippet, right_snippet):
            nuclearity = self.corpus[
                ((self.corpus.snippet_x == left_snippet) & (self.corpus.snippet_y == right_snippet))].order.values
            if nuclearity.size == 0:
                return '_'

        if type(features) == pd.Series:
            result = _get_nuclearity(features.loc['snippet_x'], features.loc['snippet_y'])
            return result
        else:
            result = features.apply(lambda row: _get_nuclearity(row.snippet_x, row.snippet_y), axis=1)
            return result.values.tolist()


class CustomTreePredictor(RSTTreePredictor):
    """
    Contains trained classifiers and feature processors needed for tree prediction.
    """

    def __init__(self, features_processor, relation_predictor, label_predictor=None, nuclearity_predictor=None):
        RSTTreePredictor.__init__(self, features_processor, relation_predictor, label_predictor, nuclearity_predictor)

    def extract_features(self, left_node: DiscourseUnit, right_node: DiscourseUnit,
                         annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                         annot_syntax_dep_tree):
        pair = pd.DataFrame({
            'snippet_x': [left_node.text.strip()],
            'snippet_y': [right_node.text.strip()],
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

    def initialize_features(self, nodes,
                            annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                            annot_syntax_dep_tree):
        pairs = pd.DataFrame({
            'snippet_x': [node.text.strip() for node in nodes[:-1]],
            'snippet_y': [node.text.strip() for node in nodes[1:]]
        })

        try:
            features = self.features_processor(pairs, annot_text=annot_text,
                                               annot_tokens=annot_tokens, annot_sentences=annot_sentences,
                                               annot_postag=annot_postag, annot_morph=annot_morph,
                                               annot_lemma=annot_lemma, annot_syntax_dep_tree=annot_syntax_dep_tree)
            return features
        except IndexError:
            with open('feature_extractor_errors.log', 'w+') as f:
                f.write(str(pairs.values))
                f.write(annot_text)
            return -1

    def predict_pair_proba(self, features):
        _same_sentence_bonus = 0.5

        if type(features) == pd.DataFrame:
            probas = self.relation_predictor.predict_proba(features)

            same_sentence_bonus = list(map(lambda value: float(value) * _same_sentence_bonus,
                                           list(features['same_sentence'] == 1)))
            return [probas[i][1] + same_sentence_bonus[i] for i in range(len(probas))]

        if type(features) == pd.Series:
            return self.relation_predictor.predict_proba(features)[0][1] + (
                    features.loc['same_sentence'] == 1) * _same_sentence_bonus

        if type(features) == list:
            return self.relation_predictor.predict_proba([features])[0][1]

    def predict_label(self, features):
        if not self.label_predictor:
            return 'relation'

        if type(features) == pd.DataFrame:
            return self.label_predictor.predict(features)

        if type(features) == pd.Series:
            return self.label_predictor.predict(features.to_frame().T)[0]

    def predict_nuclearity(self, features):
        if not self.nuclearity_predictor:
            return 'unavail'

        if type(features) == pd.DataFrame:
            return self.nuclearity_predictor.predict(features)

        if type(features) == pd.Series:
            return self.nuclearity_predictor.predict(features.to_frame().T)[0]


class NNTreePredictor(CustomTreePredictor):
    """
    Contains trained classifiers and feature processors needed for tree prediction.
    """

    def initialize_features(self, nodes,
                            annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                            annot_syntax_dep_tree):
        features = super().initialize_features(nodes,
                                               annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph,
                                               annot_postag,
                                               annot_syntax_dep_tree)
        features['snippet_x'] = features['tokens_x'].map(lambda row: ' '.join(row)).values
        features['snippet_y'] = features['tokens_y'].map(lambda row: ' '.join(row)).values

        return features

    def predict_pair_proba(self, features):
        _same_sentence_bonus = 0.12

        if type(features) == pd.DataFrame:
            probas = self.relation_predictor.predict_proba_batch(features['snippet_x'].values.tolist(),
                                                                 features['snippet_y'].values.tolist())

            same_sentence_bonus = list(map(lambda value: float(value) * _same_sentence_bonus,
                                           list(features['same_sentence'] == 1)))

            return [probas[i][1] + same_sentence_bonus[i] for i in range(len(probas))]

        if type(features) == pd.Series:
            return self.relation_predictor.predict_proba(features.loc['snippet_x'],
                                                         features.loc['snippet_y'])[0][1] + (
                           features.loc['same_sentence'] == 1) * _same_sentence_bonus

        if type(features) == list:
            snippet_x = [feature['snippet_x'] for feature in features]
            snippet_y = [feature['snippet_y'] for feature in features]

            probas = self.relation_predictor.predict_proba_batch(snippet_x, snippet_y)

            return [proba[1] for proba in probas]  # self.relation_predictor.predict_proba([features])[0][1]

    def predict_label(self, features):
        _class_mapper = {
            'background_NS': 'elaboration_NS',
            'background_SN': 'preparation_SN',
            'evaluation_NS': 'elaboration_NS',
            'evidence_NS': 'elaboration_NS',
            'comparison_NN': 'contrast_NN',
            'restatement_NN': 'joint_NN',
            'sequence_NN': 'joint_NN'
        }

        result = 'relation'

        if not self.label_predictor:
            return result

        if type(features) == pd.DataFrame:
            result = self.label_predictor.predict_batch(features['snippet_x'].values.tolist(),
                                                        features['snippet_y'].values.tolist())

        if type(features) == pd.Series:
            result = self.label_predictor.predict(features.loc['snippet_x'],
                                                  features.loc['snippet_y'])

        if type(result) == list:
            return [_class_mapper.get(value) if _class_mapper.get(value) else value for value in result]

        if _class_mapper.get(result):
            return _class_mapper.get(result)

        return result
