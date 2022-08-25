import pandas as pd
import razdel
import re
from isanlp.annotation_rst import DiscourseUnit
from td_rst_parser.predict_interactive import TrainedPredictor


def tokenize(text):
    result = ' '.join([tok.text for tok in razdel.tokenize(text)])
    return result


class Node:
    def __init__(self, left_id1, left_nuc, left_rel, left_id2, right_id1, right_nuc, right_rel, right_id2):
        self.left_nuc = left_nuc
        self.left_rel = left_rel
        self.right_nuc = right_nuc
        self.right_rel = right_rel

        self.left_id1, self.left_id2, self.right_id1, self.right_id2 = \
            list(map(int, [left_id1, left_id2, right_id1, right_id2]))

    def __str__(self):
        return str(vars(self))


class TopDownRSTParser:
    def __init__(self, rst_predictor, trained_model_path):
        self.tree_predictor = rst_predictor
        self.top_down_predictor = TrainedPredictor(trained_model_path)

    def __call__(self, edus, annot_text, annot_tokens, annot_sentences, annot_lemma, annot_morph, annot_postag,
                 annot_syntax_dep_tree, genre=None):
        """
        :param list edus: DiscourseUnit
        :param str annot_text: original text
        :param list annot_tokens: isanlp.annotation.Token
        :param list annot_sentences: isanlp.annotation.Sentence
        :param list annot_postag: lists of str for each sentence
        :param annot_lemma: lists of str for each sentence
        :param annot_syntax_dep_tree: list of isanlp.annotation.WordSynt for each sentence
        :return: one extracted tree as a DiscourseUnit
        """

        # 1. Predict tree structure on available EDUs using beam search
        document = {
            'InputDocs': [TopDownRSTParser.get_input_docs(annot_tokens)],
            'EduBreak_TokenLevel': [TopDownRSTParser.get_edu_breaks(edus, annot_tokens)],
            'SentBreak': [TopDownRSTParser.get_sentence_breaks(annot_sentences)],
        }
        document['Docs_structure'] = [TopDownRSTParser.construct_dummy_tree(len(document['EduBreak_TokenLevel'][0]))]
        string_descriptions = self.top_down_predictor.predict(document)
        string_descriptions['InputDocs'] = document['InputDocs']

        # 2. Correct the labels predicted with top-down model using our smart context-aware label predictor
        max_id = self._get_max_id(edus)
        self._id = max_id
        toks, nods = TopDownRSTParser.docs_structure_to_nodes(structure=string_descriptions['trees'][0],
                                                              tokens=document['InputDocs'][0])
        nods = self.predict_labels(nods, annot_text, annot_tokens,
                                   annot_sentences,
                                   annot_lemma, annot_morph, annot_postag,
                                   annot_syntax_dep_tree)

        # 3. Export into isanlp.DiscourseUnit
        du = self.docs_structure_to_du(nodes=nods, tokens=annot_tokens, text=annot_text,
                                       _tok_min=0, _tok_max=TopDownRSTParser.max_mentioned_token(nods))
        return [du]

    def _get_max_id(self, dus):
        max_id = dus[-1].id
        for du in dus[:-1]:
            if du.id > max_id:
                max_id = du.id

        return max_id

    def predict_labels(self, nodes, annot_text, annot_tokens,
                       annot_sentences,
                       annot_lemma, annot_morph, annot_postag,
                       annot_syntax_dep_tree):

        result = []
        for node in nodes:
            start = annot_tokens[node.left_id1].begin
            end = annot_tokens[node.left_id2].end
            left_node = DiscourseUnit(id=0,
                                      text=annot_text[start:end],
                                      start=start,
                                      end=end,
                                      )
            start = annot_tokens[node.right_id1].begin
            end = annot_tokens[node.right_id2].end
            right_node = DiscourseUnit(id=1,
                                       text=annot_text[start:end],
                                       start=start,
                                       end=end,
                                       )

            try:
                pair_feature = self.tree_predictor.extract_features(left_node, right_node,
                                                                    annot_text, annot_tokens,
                                                                    annot_sentences,
                                                                    annot_lemma, annot_morph, annot_postag,
                                                                    annot_syntax_dep_tree)
                relation = self._get_relation(pair_feature)
                relation, nuclearity = relation.split('_')

                left_nuclearity = 'Satellite' if nuclearity == 'SN' else 'Nucleus'
                right_nuclearity = 'Satellite' if nuclearity == 'NS' else 'Nucleus'

                left_relation = relation
                right_relation = relation

                if left_nuclearity == 'Satellite':
                    right_relation = 'span'

                if right_nuclearity == 'Satellite':
                    left_relation = 'span'

            except:
                print('Unknown error occured.')
                left_relation = node.left_rel
                left_nuclearity = node.left_nuc
                right_relation = node.right_rel
                right_nuclearity = node.right_nuc

            result.append(Node(left_id1=node.left_id1,
                               left_nuc=left_nuclearity,
                               left_rel=left_relation,
                               left_id2=node.left_id2,
                               right_id1=node.right_id1,
                               right_nuc=right_nuclearity,
                               right_rel=right_relation,
                               right_id2=node.right_id2))

        return result

    def _get_relation(self, pair_feature):
        relation = 'joint_NN'

        # try:
        relation = self.tree_predictor.predict_label(pair_feature)
        if type(relation) == list:
            relation = relation[0]

        return relation

    @staticmethod
    def get_input_docs(tokens):
        """ InputDocs : list of lists with plain tokens of each document """
        return [token.text for token in tokens]

    @staticmethod
    def get_edu_breaks(doc_trees: list, tokens: list):
        """ EduBreak_TokenLevel : list of lists with the token positions of right EDU ends of each document """

        def extr_edus(tree, begin):
            if tree.relation == 'elementary':
                return [(tree.start - begin, tree.end - begin)]
            else:
                tt = []
                tt += extr_edus(tree.left, begin=begin)
                tt += extr_edus(tree.right, begin=begin)
            return tt

        def map_offset_to_tokens(offset):
            begin, end = -1, -1
            i = 0
            for i, token in enumerate(tokens):
                if begin == -1 and token.begin > offset[0]:
                    begin = i - 1
                if begin != -1:
                    if token.end > offset[1]:
                        end = i - 1
                        return begin, end
            return begin, i

        edus = []
        for tree in doc_trees:
            edus += extr_edus(tree, begin=0)

        right_borders = sorted(list(set([map_offset_to_tokens(offset)[1] for offset in edus])))
        if not len(tokens) - 1 in right_borders:
            right_borders.append(len(tokens) - 1)
        return right_borders

    @staticmethod
    def construct_dummy_tree(number_of_edu):
        result = []
        for i in range(1, number_of_edu):
            result.append(f'(1:Nucleus=joint:{i},{i + 1}:Nucleus=joint:{i + 1})')
        return result

    @staticmethod
    def get_sentence_breaks(sentences):
        """ SentBreak for sentence breaks in terms of token offsets """
        return [sentence.end - 1 for sentence in sentences]

    @staticmethod
    def leftmostid(tree):
        if tree.left:
            return TopDownRSTParser.leftmostid(tree.left)
        return tree.id

    @staticmethod
    def rightmostid(tree):
        if tree.right:
            return TopDownRSTParser.rightmostid(tree.right)
        return tree.id

    @staticmethod
    def correct_relations(rel: str, nuc: str):
        target_map = {
            'relation': 'joint',
            'antithesis': 'contrast',
            'cause': 'cause-effect',
            'effect': 'cause-effect',
            'conclusion': 'restatement',
            'interpretation': 'interpretation-evaluation',
            'evaluation': 'interpretation-evaluation',
            'motivation': 'condition',
            'span': 'attribution'
        }

        if rel in target_map:
            rel = target_map.get(rel)

        relation_map = {
            'restatement_SN': 'restatement_NN',
            'restatement_NS': 'restatement_NN',
            'contrast_SN': 'contrast_NN',
            'contrast_NS': 'contrast_NN',
            'solutionhood_NS': 'elaboration_NS',
            'preparation_NS': 'elaboration_NS',
            'concession_SN': 'preparation_SN',
            'evaluation_SN': 'preparation_SN',
            'elaboration_SN': 'preparation_SN',
            'evidence_SN': 'preparation_SN',
            'background_SN': 'preparation_SN'
        }

        full_rel = rel + '_' + nuc
        if full_rel in relation_map:
            full_rel = relation_map.get(full_rel)
            rel, nuc = full_rel.split('_')

        return rel, nuc

    @staticmethod
    def collect_edus(docs_structure):
        edus_id = []
        for entry in docs_structure:
            left, right = entry.split(',')
            left = left.replace('(', '').split(':')
            du1, du2 = left[0], left[2]
            if du1 == du2:
                edus_id.append(int(du1))

            right = right.replace(')', '').split(':')
            du1, du2 = right[0], right[2]
            if du1 == du2:
                edus_id.append(int(du1))
        return edus_id

    @staticmethod
    def pad(tokens, max_len):
        return tokens + [''] * (max_len - len(tokens))

    @staticmethod
    def docs_structure_to_nodes(structure: str, tokens: list):
        if structure == 'NONE':
            return []

        _pred_tok_max = 0  # for padding
        nodes = []

        for line in structure.split():
            args = re.split(r'[:=,]', line[1:-1].strip())
            nodes.append(Node(*args))

            if nodes[-1].right_id2 > _pred_tok_max:
                _pred_tok_max = nodes[-1].right_id2

        tokens = TopDownRSTParser.pad(tokens, _pred_tok_max)
        return tokens, nodes

    @staticmethod
    def define_root(nodes, _tok_min, _tok_max):
        for node in nodes:
            if node.left_id1 == _tok_min and node.right_id2 == _tok_max:
                return node

    def docs_structure_to_du(self, nodes, tokens, text, _tok_min, _tok_max):
        du = None
        root = TopDownRSTParser.define_root(nodes, _tok_min, _tok_max)

        if root:
            nuc = 'NN'
            rel = root.left_rel

            if root.left_nuc == 'Satellite':
                nuc = 'SN'
                rel = root.left_rel
            elif root.right_nuc == 'Satellite':
                nuc = 'NS'
                rel = root.right_rel

            if root.left_id1 == root.left_id2:
                self._id += 1
                start = tokens[root.left_id1].begin
                end = tokens[root.left_id2].end
                left = DiscourseUnit(id=self._id,
                                     start=start,
                                     end=end,
                                     text=text[start:end],
                                     relation='elementary')
            else:
                left = self.docs_structure_to_du(nodes, tokens, text, root.left_id1, root.left_id2)

            if root.right_id1 == root.right_id2:
                self._id += 1
                start = tokens[root.right_id1].begin
                end = tokens[root.right_id2].end
                right = DiscourseUnit(id=self._id,
                                      start=start,
                                      end=end,
                                      text=text[start:end],
                                      relation='elementary')
            else:
                right = self.docs_structure_to_du(nodes, tokens, text, root.right_id1, root.right_id2)

            self._id += 1
            start = tokens[root.left_id1].begin
            end = tokens[root.right_id2].end
            new_du = DiscourseUnit(id=self._id,
                                   start=start,
                                   end=end,
                                   relation=rel,
                                   nuclearity=nuc,
                                   text=text[start:end],
                                   left=left,
                                   right=right)
            return new_du

        else:
            self._id += 1
            start = tokens[_tok_min].begin
            end = tokens[_tok_max].end
            return DiscourseUnit(id=self._id,
                                 start=start,
                                 end=end,
                                 text=text[start:end],
                                 relation='elementary')

    @staticmethod
    def max_mentioned_token(nodes):
        result = 0
        for node in nodes:
            if node.right_id2 > result:
                result = node.right_id2
        return result

    @staticmethod
    def relation_stats(nodes):
        all_rels = pd.Series([node.left_rel if node.left_nuc == 'Satellite' else node.right_rel for node in nodes])
        return all_rels.describe()

    @staticmethod
    def nuclearity_stats(nodes):
        all_rels = pd.Series(
            ['SN' if node.left_nuc == 'Satellite' else 'NS' if node.right_nuc == 'Satellite' else 'NN' for node in
             nodes])
        return all_rels.describe()
