import numpy as np
import sys

from isanlp.annotation_rst import DiscourseUnit


class GreedyRSTParser:
    def __init__(self, tree_predictor, forest_threshold=0.05):
        """
        :param RSTTreePredictor tree_predictor:
        :param float forest_threshold: minimum relation probability to append the pair into the tree
        """
        self.tree_predictor = tree_predictor
        self.forest_threshold = forest_threshold

    def __call__(self, edus, annot_text, annot_tokens, annot_sentences, annot_postag, annot_morph, annot_lemma,
                 annot_syntax_dep_tree, genre=None):
        """
        :param list edus: DiscourseUnit
        :param str annot_text: original text
        :param list annot_tokens: isanlp.annotation.Token
        :param list annot_sentences: isanlp.annotation.Sentence
        :param list annot_postag: lists of str for each sentence
        :param annot_lemma: lists of str for each sentence
        :param annot_syntax_dep_tree: list of isanlp.annotation.WordSynt for each sentence
        :return: list of DiscourseUnit containing each extracted tree
        """

        def to_merge(scores):
            return np.argmax(np.array(scores))

        self.tree_predictor.genre = genre

        nodes = edus
        
        for edu in nodes:
            print(edu, file=sys.stderr)
        
        max_id = edus[-1].id

        # initialize scores
        features = [
            self.tree_predictor.extract_features(nodes[i], nodes[i + 1], annot_text, annot_tokens,
                                                 annot_sentences,
                                                 annot_postag, annot_morph, annot_lemma,
                                                 annot_syntax_dep_tree)
            for i in range(len(nodes) - 1)]

        scores = [self.tree_predictor.predict_pair_proba(features[i]) for i in range(len(nodes) - 1)]

        while len(nodes) > 2 and any([score > self.forest_threshold for score in scores]):
            # select two nodes to merge
            j = to_merge(scores)  # position of the pair in list
            relation = self.tree_predictor.predict_label(features)

            # make the new node by merging node[j] + node[j+1]
            temp = DiscourseUnit(
                id=max_id + 1,
                left=nodes[j],
                right=nodes[j + 1],
                relation=relation,
                proba=scores[j],
                text=nodes[j].text + nodes[j + 1].text  #annot_text[nodes[j].start:nodes[j+1].end]
            )
            
            print(temp, file=sys.stderr)
            
            max_id += 1

            # modify the node list
            nodes = nodes[:j] + [temp] + nodes[j + 2:]

            # modify the scores list
            if j == 0:
                features = self.tree_predictor.extract_features(nodes[j], nodes[j + 1],
                                                                annot_text, annot_tokens, annot_sentences, annot_postag,
                                                                annot_morph, annot_lemma, annot_syntax_dep_tree)
                predicted = self.tree_predictor.predict_pair_proba(features)

                scores = [predicted] + scores[j + 2:]

            elif j + 1 < len(nodes):
                features_left = self.tree_predictor.extract_features(nodes[j - 1], nodes[j], annot_text, annot_tokens,
                                                                     annot_sentences, annot_postag, annot_morph,
                                                                     annot_lemma, annot_syntax_dep_tree)
                predicted_left = self.tree_predictor.predict_pair_proba(features_left)

                features_right = self.tree_predictor.extract_features(nodes[j], nodes[j + 1], annot_text, annot_tokens,
                                                                      annot_sentences, annot_postag, annot_morph,
                                                                      annot_lemma, annot_syntax_dep_tree)
                predicted_right = self.tree_predictor.predict_pair_proba(features_right)

                scores = scores[:j - 1] + [predicted_left] + [predicted_right] + scores[j + 2:]

            else:
                features = self.tree_predictor.extract_features(nodes[j - 1], nodes[j],
                                                                annot_text, annot_tokens, annot_sentences, annot_postag,
                                                                annot_morph, annot_lemma, annot_syntax_dep_tree)
                predicted = self.tree_predictor.predict_pair_proba(features)
                scores = scores[:j - 1] + [predicted]

        if len(scores) == 1 and scores[0] > self.forest_threshold:
            root = DiscourseUnit(
                id=max_id + 1,
                left=nodes[0],
                right=nodes[1],
                relation='root',
                proba=scores[0]
            )
            nodes = [root]

        return nodes
