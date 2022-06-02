def get_input_docs(doc_annot: dict):
    """ InputDocs : list of lists with plain tokens of each document """
    return [token.text for token in doc_annot.get('tokens')]


def get_edu_breaks(doc_trees: list, doc_annot: dict):
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
        for i, token in enumerate(doc_annot['tokens']):
            if begin == -1 and token.begin > offset[0]:
                begin = i - 1
            if begin != -1:
                if token.end > offset[1]:
                    end = i - 1
                    return begin, end
        return begin, i

    edus = []
    for tree in doc_trees:
        begin = tree.start
        edus += extr_edus(tree, begin=tree.start)

    return [map_offset_to_tokens(offset)[1] for offset in edus]


def get_sentence_breaks(doc_annot: dict):
    """ SentBreak for sentence breaks in terms of token offsets """
    return [sentence.end - 1 for sentence in doc_annot.get('sentences')]


def leftmostid(tree):
    if tree.left:
        return leftmostid(tree.left)
    return tree.id


def leftmoststart(tree):
    if tree.left:
        return leftmoststart(tree.left)
    return tree.start


def rightmostid(tree):
    if tree.right:
        return rightmostid(tree.right)
    return tree.id


def rightmostend(tree):
    if tree.right:
        return rightmostend(tree.right)
    return tree.end


true_relations = ['attribution_NS', 'attribution_SN', 'background_NS',
                  'cause-effect_NS', 'cause-effect_SN',
                  'comparison_NN', 'concession_NS', 'condition_NS', 'condition_SN',
                  'contrast_NN', 'elaboration_NS', 'evidence_NS',
                  'interpretation-evaluation_NS', 'interpretation-evaluation_SN',
                  'joint_NN', 'preparation_SN', 'purpose_NS', 'purpose_SN',
                  'restatement_NN', 'same-unit_NN', 'sequence_NN', 'solutionhood_SN']


def correct_relations(rel: str, nuc: str):
    target_map = {
        'antithesis': 'attribution',
        'cause': 'cause-effect',
        'conclusion': 'restatement',
        'interpretation': 'interpretation-evaluation',
        'evaluation': 'interpretation-evaluation',
        'motivation': 'condition',
    }

    relation_map = {
        'effect_SN': 'cause-effect_NS',  # In essays, they are reversed
        'effect_NS': 'cause-effect_SN',
        'evidence_SN': 'preparation_SN',
        'restatement_SN': 'condition_SN',
        'restatement_NS': 'elaboration_NS',
        'solutionhood_NS': 'elaboration_NS',
        'preparation_NS': 'elaboration_NS',
        'concession_SN': 'preparation_SN',
        'evaluation_SN': 'preparation_SN',
        'elaboration_SN': 'preparation_SN',
        'background_SN': 'preparation_SN',
    }

    if rel in target_map:
        rel = target_map.get(rel)

    full_rel = rel + '_' + nuc
    if full_rel in relation_map:
        full_rel = relation_map.get(full_rel)
        rel, nuc = full_rel.split('_')

    if not full_rel in true_relations:
        print(rel, nuc, full_rel)

    return rel, nuc


def du_to_docs_structure(tree: dict, du_counter: int, needs_preprocessing=True):
    if tree.relation != 'elementary':
        if needs_preprocessing:
            tree.relation, tree.nuclearity = correct_relations(tree.relation, tree.nuclearity)

        left_nuclearity = 'Satellite' if tree.nuclearity == 'SN' else 'Nucleus'
        right_nuclearity = 'Satellite' if tree.nuclearity == 'NS' else 'Nucleus'
        left_relation = tree.relation
        right_relation = tree.relation

        left_id_1 = leftmostid(tree.left) + du_counter
        left_id_2 = rightmostid(tree.left) + du_counter
        right_id_1 = leftmostid(tree.right) + du_counter
        right_id_2 = rightmostid(tree.right) + du_counter

        if left_nuclearity == 'Satellite':
            right_relation = 'span'

        if right_nuclearity == 'Satellite':
            left_relation = 'span'

        relstring_l = f'{left_id_1}:{left_nuclearity}={left_relation}:{left_id_2}'
        relstring_r = f'{right_id_1}:{right_nuclearity}={right_relation}:{right_id_2}'

        left_subtree_struct = du_to_docs_structure(tree.left, du_counter, needs_preprocessing) or []
        right_subtree_struct = du_to_docs_structure(tree.right, du_counter, needs_preprocessing) or []
        return [f'({relstring_l},{relstring_r})'] + left_subtree_struct + right_subtree_struct


def du_to_docs_structure_char(tree):
    if tree.relation != 'elementary':
        tree.relation, tree.nuclearity = correct_relations(tree.relation, tree.nuclearity)

        left_nuclearity = 'Satellite' if tree.nuclearity == 'SN' else 'Nucleus'
        right_nuclearity = 'Satellite' if tree.nuclearity == 'NS' else 'Nucleus'
        left_relation = tree.relation
        right_relation = tree.relation

        left_id_1 = leftmoststart(tree.left)
        left_id_2 = rightmostend(tree.left)
        right_id_1 = leftmoststart(tree.right)
        right_id_2 = rightmostend(tree.right)

        if left_nuclearity == 'Satellite':
            right_relation = 'span'

        if right_nuclearity == 'Satellite':
            left_relation = 'span'

        relstring_l = f'{left_id_1}:{left_nuclearity}={left_relation}:{left_id_2}'
        relstring_r = f'{right_id_1}:{right_nuclearity}={right_relation}:{right_id_2}'

        left_subtree_struct = du_to_docs_structure_char(tree.left) or []
        right_subtree_struct = du_to_docs_structure_char(tree.right) or []
        return [f'({relstring_l},{relstring_r})'] + left_subtree_struct + right_subtree_struct


def charsonly(text):
    return ''.join(text.replace('#', '').split())


def du_to_docs_structure_charsonly(tree, text, needs_preprocessing=True):
    if tree.relation != 'elementary':
        if needs_preprocessing:
            tree.relation, tree.nuclearity = correct_relations(tree.relation, tree.nuclearity)

        left_nuclearity = 'Satellite' if tree.nuclearity == 'SN' else 'Nucleus'
        right_nuclearity = 'Satellite' if tree.nuclearity == 'NS' else 'Nucleus'
        left_relation = tree.relation
        right_relation = tree.relation

        left_id_1 = text.find(charsonly(tree.left.text))
        left_id_2 = left_id_1 + len(charsonly(tree.left.text))
        right_id_1 = text.find(charsonly(tree.right.text))
        right_id_2 = right_id_1 + len(charsonly(tree.right.text))

        if left_nuclearity == 'Satellite':
            right_relation = 'span'

        if right_nuclearity == 'Satellite':
            left_relation = 'span'

        relstring_l = f'{left_id_1}:{left_nuclearity}={left_relation}:{left_id_2}'
        relstring_r = f'{right_id_1}:{right_nuclearity}={right_relation}:{right_id_2}'

        left_subtree_struct = du_to_docs_structure_charsonly(tree.left, text, needs_preprocessing) or []
        right_subtree_struct = du_to_docs_structure_charsonly(tree.right, text, needs_preprocessing) or []
        return [f'({relstring_l},{relstring_r})'] + left_subtree_struct + right_subtree_struct


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


def get_docs_structure(doc_trees: list):
    result = []
    du_counter = 0
    for tree in doc_trees:
        structure = du_to_docs_structure(tree, du_counter)
        if structure:
            result += structure
            du_counter += len(structure)
        else:
            du_counter += 1
    return result


def get_docs_structure_charsonly(doc_trees: list, needs_preprocessing=True):
    result = []
    text = charsonly(''.join([tree.text for tree in doc_trees]))
    for tree in doc_trees:
        structure = du_to_docs_structure_charsonly(tree, text, needs_preprocessing)
        if structure:
            result += structure
    return result
