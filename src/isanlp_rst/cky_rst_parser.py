from isanlp.annotation_rst import DiscourseUnit


class DiscourseUnitCreator:
    def __init__(self, id, text=""):
        self.id = id
        self.text = text

    def __call__(self, left_node, right_node, proba=1., text="", start=None, end=None, relation="", nuclearity=""):
        self.id += 1
        return DiscourseUnit(
            id=self.id,
            left=left_node,
            right=right_node,
            relation=relation,
            nuclearity=nuclearity,
            proba=proba,
            start=start,
            end=end,
            orig_text=self.text
        )


class CKYRSTParser:
    def __init__(self, tree_predictor, confidence_threshold=0.1, threshold_max=0.6, threshold_min=0.1,
                 threshold_decay=0.05):
        """
        :param RSTTreePredictor tree_predictor:
        :param float confidence_threshold: minimum relation probability to append the pair into the tree
        """
        self.tree_predictor = tree_predictor
        self._threshold_max = threshold_max
        self._threshold_min = threshold_min
        self._threshold_decay = threshold_decay
        self._confidence_threshold = confidence_threshold

    def __call__(self, edus, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree):
        tree = None
        threshold = self._threshold_max

        while True:
            tree = self._parse_cky(edus, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph,
                                   annot_postag, annot_syntax_dep_tree)
            if tree is not None:
                break

            threshold -= self._threshold_decay

        return tree

    def _parse_cky(self, edus, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                   annot_syntax_dep_tree):

        length = len(edus)
        parse_table = [[None for _ in range(length - y)] for y in range(length)]
        for i in range(length):
            parse_table[0][i] = edus[i]

        ds_creator = DiscourseUnitCreator(edus[-1].id, annot_text)
        threshold = self._confidence_threshold

        for words_to_consider in range(2, length + 1):
            for starting_cell in range(0, length - words_to_consider + 1):

                for left_size in range(1, words_to_consider):
                    right_size = words_to_consider - left_size

                    left_node = parse_table[left_size - 1][starting_cell]
                    right_node = parse_table[right_size - 1][starting_cell + left_size]

                    if (right_node is None) or (left_node is None):
                        continue

                    _features = self.tree_predictor.extract_features(left_node, right_node,
                                                                     annot_text, annot_tokens,
                                                                     annot_sentences,
                                                                     annot_lemma, annot_morph, annot_postag,
                                                                     annot_syntax_dep_tree)

                    score = self.tree_predictor.predict_pair_proba(_features)[0]
                    label = self.tree_predictor.predict_label(_features)
                    relation, nuclearity = label.split('_')

                    if score > threshold:
                        proba = left_node.proba * right_node.proba * score
                        if ((parse_table[words_to_consider - 1][starting_cell] is None) or
                                proba > parse_table[words_to_consider - 1][starting_cell].proba):
                            parse_table[words_to_consider - 1][starting_cell] = ds_creator(left_node, right_node, proba, 
                                                                                           relation=relation, nuclearity=nuclearity)

            threshold -= self._threshold_decay
            if threshold < self._threshold_min:
                return parse_table[words_to_consider - 1]

        return [parse_table[length - 1][0]]
